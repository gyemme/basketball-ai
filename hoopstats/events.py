"""Stage 2 logic: ball possession, shots, makes/misses, rebounds, assists, turnovers.

All functions work on plain numpy arrays from extract.py plus a track -> team map, so they are
easy to unit-test with synthetic games (see tests/).

Conventions
  frame index fi -> time t[fi]
  team ids: "A" = team on the LEFT of the ScoreSync overlay, "B" = RIGHT
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .camera import Camera
from .court import Court


@dataclass
class Hold:
    tid: int
    team: str | None
    f0: int
    f1: int          # last frame the ball was with this player
    t0: float
    t1: float

    @property
    def dur(self) -> float:
        return self.t1 - self.t0


@dataclass
class Attempt:
    hold: Hold
    f_release: int
    t: float
    hoop: int | None
    near_rim_t: float | None
    video_made: bool | None
    xy: tuple[float, float] | None
    made: bool = False
    value: int | None = None
    matched: bool = False
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------- per-frame index

class FrameIndex:
    def __init__(self, ex: dict):
        self.t = ex["t"]
        self.N = len(self.t)
        self.people = ex["people"]
        self.by_frame: list[np.ndarray] = [np.zeros(0, int)] * self.N
        if len(self.people):
            order = np.argsort(self.people[:, 0], kind="stable")
            fis = self.people[order, 0].astype(int)
            bounds = np.searchsorted(fis, np.arange(self.N + 1))
            self.by_frame = [order[bounds[i]:bounds[i + 1]] for i in range(self.N)]
        self.ball = np.full((self.N, 4), np.nan)  # x, y, r, conf
        for fi, x, y, r, c in ex["balls"]:
            self.ball[int(fi)] = (x, y, r, c)
        self.H = ex["H"]
        self.Hconf = ex["Hconf"]

    def rows(self, fi):
        return self.people[self.by_frame[fi]]


def interpolate_ball(ball: np.ndarray, max_gap: int = 3) -> np.ndarray:
    b = ball.copy()
    ok = ~np.isnan(b[:, 0])
    idx = np.where(ok)[0]
    for a, c in zip(idx[:-1], idx[1:]):
        if 1 < c - a <= max_gap + 1:
            for k in range(a + 1, c):
                w = (k - a) / (c - a)
                b[k, :3] = (1 - w) * b[a, :3] + w * b[c, :3]
                b[k, 3] = 0
    return b


# ---------------------------------------------------------------------------- possession

def holders(fx: FrameIndex, ball: np.ndarray, team_of: dict[int, str | None]) -> np.ndarray:
    hold = np.full(fx.N, -1, int)
    for fi in range(fx.N):
        bx, by = ball[fi, 0], ball[fi, 1]
        if np.isnan(bx):
            continue
        rows = fx.rows(fi)
        best, bd = -1, 9e9
        for r in rows:
            tid = int(r[1])
            if team_of.get(tid) is None:
                continue
            x1, y1, x2, y2 = r[2:6]
            w, h = x2 - x1, y2 - y1
            if not (x1 - 0.25 * w <= bx <= x2 + 0.25 * w and y1 - 0.08 * h <= by <= y2 + 0.02 * h):
                continue
            d = abs(bx - (x1 + x2) / 2) / w + abs(by - (y1 + 0.5 * h)) / h
            if d < bd:
                best, bd = tid, d
        hold[fi] = best
    return hold


def smooth_holds(hold: np.ndarray, t: np.ndarray, team_of: dict, min_frames: int = 2, bridge: int = 3) -> list[Hold]:
    runs = []
    fi = 0
    N = len(hold)
    while fi < N:
        if hold[fi] < 0:
            fi += 1
            continue
        tid = hold[fi]
        f0 = fi
        while fi < N and hold[fi] == tid:
            fi += 1
        runs.append([int(tid), f0, fi - 1])
    merged: list[list[int]] = []
    for r in runs:
        if merged and merged[-1][0] == r[0] and r[1] - merged[-1][2] <= bridge + 1:
            merged[-1][2] = r[2]
        else:
            merged.append(r)
    out = []
    for tid, f0, f1 in merged:
        if f1 - f0 + 1 < min_frames:
            continue
        out.append(Hold(tid, team_of.get(tid), f0, f1, float(t[f0]), float(t[f1])))
    return out


# ---------------------------------------------------------------------------- rims

def rim_positions(fx: FrameIndex, court: Court, width: int, height: int) -> np.ndarray:
    """(N, 2, 3) array of (u, v, r_px) per hoop, NaN where unknown."""
    rims = np.full((fx.N, 2, 3), np.nan)
    fs = []
    for fi in range(fx.N):
        H = fx.H[fi]
        if np.isnan(H[0, 0]) or fx.Hconf[fi] < 0.2:
            continue
        try:
            cam = Camera(H, width, height)
        except Exception:
            continue
        fs.append(cam.f)
        for k, hp in enumerate(court.hoops):
            r = cam.rim(hp)
            if r and -0.2 * width < r[0] < 1.2 * width and -0.3 * height < r[1] < height:
                rims[fi, k] = r
    return rims


# ---------------------------------------------------------------------------- shots

def detect_attempts(fx: FrameIndex, ball: np.ndarray, holds: list[Hold], rims: np.ndarray, fps: float,
                    court: Court) -> list[Attempt]:
    out: list[Attempt] = []
    look = int(3.0 * fps)
    for i, h in enumerate(holds):
        if h.team is None:
            continue
        nxt = holds[i + 1] if i + 1 < len(holds) else None
        f_end = min(fx.N - 1, h.f1 + look)
        if nxt is not None:
            f_end = min(f_end, nxt.f0 + int(0.4 * fps))   # the ball can't fly through a catch
        # holder geometry at release
        rel = [r for r in fx.rows(h.f1) if int(r[1]) == h.tid]
        if not rel:
            continue
        x1, y1, x2, y2 = rel[0][2:6]
        bh = y2 - y1
        seg = ball[h.f1 + 1:f_end + 1]
        if len(seg) < 3 or np.isnan(seg[:, 1]).all():
            continue
        ys = seg[:, 1]
        peak = np.nanmin(ys)
        rose = peak < y1 - 0.25 * bh
        near_t, near_hoop, made_v = None, None, None
        for k in range(len(seg)):
            fi = h.f1 + 1 + k
            if np.isnan(seg[k, 0]):
                continue
            for hp in (0, 1):
                u, v, rr = rims[fi, hp]
                if np.isnan(u):
                    continue
                tol = max(3.5 * rr, 0.45 * bh)
                if abs(seg[k, 0] - u) < tol and abs(seg[k, 1] - v) < tol:
                    near_t, near_hoop = float(fx.t[fi]), hp
                    # did it drop through? above the rim, then below it, staying inside the rim's width
                    above = False
                    for k2 in range(max(0, k - int(0.6 * fps)), min(len(seg), k + int(1.2 * fps))):
                        fi2 = h.f1 + 1 + k2
                        bx, by = seg[k2, 0], seg[k2, 1]
                        if np.isnan(bx):
                            continue
                        u2, v2, r2 = rims[fi2, hp] if not np.isnan(rims[fi2, hp, 0]) else (u, v, rr)
                        if by < v2 - 0.5 * r2:
                            above = True
                        elif above and by > v2 + 1.0 * r2 and abs(bx - u2) < 1.6 * r2:
                            made_v = True
                            break
                    if made_v is None:
                        made_v = False
                    break
            if near_t is not None:
                break
        both_rims = (~np.isnan(rims[h.f1:f_end + 1, :, 0])).all(axis=0).all() if f_end > h.f1 else False
        if near_t is None and both_rims:
            continue                       # both rims visible and the ball went near neither: pass/dribble
        if near_t is None and not (rose and peak < y1 - 0.8 * bh):
            continue                       # no rim info: need a clear high arc
        xy = None
        for d in range(0, int(0.6 * fps)):
            for fi in (h.f1 - d, h.f1 + d):
                if 0 <= fi < fx.N and fx.Hconf[fi] >= 0.25:
                    rr = [r for r in fx.rows(fi) if int(r[1]) == h.tid and not np.isnan(r[7])]
                    if rr:
                        xy = (float(rr[0][7]), float(rr[0][8]))
                        break
            if xy:
                break
        hoop = near_hoop if near_hoop is not None else (court.nearest_hoop(*xy) if xy else None)
        out.append(Attempt(hold=h, f_release=h.f1, t=float(fx.t[h.f1]), hoop=hoop,
                           near_rim_t=near_t, video_made=made_v, xy=xy))
    return out


# ---------------------------------------------------------------------------- team <-> overlay side

def map_sides(attempts: list[Attempt], holds: list[Hold], baskets: list[dict], window: float) -> dict[str, str]:
    """Map raw team clusters ('0','1') to overlay sides ('A' left, 'B' right) by which cluster
    was shooting right before each score change."""
    votes = defaultdict(float)
    for b in baskets:
        side = "A" if b["side"] == "left" else "B"
        cands = [a for a in attempts if b["t"] - window <= a.t <= b["t"] - 0.3]
        latest = {}
        for a in cands:
            latest[a.hold.team] = a
        for team, a in latest.items():
            w = (2.0 if a.video_made else 1.0 if a.near_rim_t else 0.6) * math.exp(-(b["t"] - a.t) / 8.0)
            votes[(team, side)] += w
        if not cands:
            hs = [h for h in holds if b["t"] - window <= h.t1 <= b["t"] - 2 and h.team is not None]
            if hs:
                votes[(hs[-1].team, side)] += 0.2
    straight = votes[("0", "A")] + votes[("1", "B")]
    crossed = votes[("0", "B")] + votes[("1", "A")]
    return {"0": "B", "1": "A"} if crossed > straight else {"0": "A", "1": "B"}


# ---------------------------------------------------------------------------- periods / direction

def attack_direction(attempts: list[Attempt], duration: float, forced: list[float] | None = None):
    """Returns (period_starts, dir) where dir[(team, period)] = hoop index attacked."""
    pts = [(a.t, a.hold.team, a.hoop) for a in attempts if a.hoop is not None and a.hold.team in ("A", "B")]

    def score(split):
        tot = 0
        dirs = {}
        for p, (lo, hi) in enumerate(split):
            for team in ("A", "B"):
                hs = [hp for t, tm, hp in pts if tm == team and lo <= t < hi]
                if hs:
                    k = max((0, 1), key=lambda x: hs.count(x))
                    dirs[(team, p)] = k
                    tot += hs.count(k)
            # teams attack opposite hoops
            if ("A", p) in dirs and ("B", p) in dirs and dirs[("A", p)] == dirs[("B", p)]:
                tot -= 2
        return tot, dirs

    end = duration + 1
    if forced:
        starts = sorted(set([0.0] + list(forced)))
        split = list(zip(starts, starts[1:] + [end]))
        return starts, score(split)[1]
    base, d0 = score([(0, end)])
    best = (base, [0.0], d0)
    if len(pts) >= 8:
        for s in np.linspace(0.25 * duration, 0.75 * duration, 41):
            sc, d = score([(0, s), (s, end)])
            n1 = sum(1 for p in pts if p[0] < s)
            if sc > best[0] * 1.25 + 1 and n1 >= 3 and len(pts) - n1 >= 3:
                best = (sc, [0.0, float(s)], d)
    starts, dirs = best[1], best[2]
    for p in range(len(starts)):     # fill a missing team from the other one
        if ("A", p) in dirs and ("B", p) not in dirs:
            dirs[("B", p)] = 1 - dirs[("A", p)]
        if ("B", p) in dirs and ("A", p) not in dirs:
            dirs[("A", p)] = 1 - dirs[("B", p)]
    return starts, dirs


def period_of(t: float, starts: list[float]) -> int:
    p = 0
    for i, s in enumerate(starts):
        if t >= s:
            p = i
    return p


# ---------------------------------------------------------------------------- main event builder

def build(fx: FrameIndex, ex: dict, team_of: dict[int, str | None], player_of: dict[int, str],
          baskets: list[dict], gaps: list[dict], court: Court, fps: float, size: tuple[int, int],
          cfg: dict) -> dict:
    """Returns dict with shots, events, possessions, periods, directions, holds, attempts."""
    W, H = size
    window = float(cfg.get("shots", {}).get("make_window_s", 25))
    ball = interpolate_ball(fx.ball, max_gap=max(2, int(0.3 * fps)))
    hold_arr = holders(fx, ball, team_of)
    holds = smooth_holds(hold_arr, fx.t, team_of, min_frames=max(2, int(0.2 * fps)), bridge=max(2, int(0.35 * fps)))
    rims = rim_positions(fx, court, W, H)
    attempts = detect_attempts(fx, ball, holds, rims, fps, court)
    duration = float(fx.t[-1]) if fx.N else 0.0
    forced = cfg.get("periods")
    starts, dirs = attack_direction(attempts, duration, forced[1:] if forced else None)

    def attack_hoop(team: str, t: float) -> int | None:
        return dirs.get((team, period_of(t, starts)))

    # ---------------- match score changes to attempts
    shots = []
    sid = 0
    have_overlay = bool(baskets)
    for b in baskets:
        team = "A" if b["side"] == "left" else "B"
        pts = int(b["points"])
        if pts == 1:
            continue  # free throws handled below
        cands = [a for a in attempts if a.hold.team == team and not a.matched and b["t"] - window <= a.t <= b["t"] - 0.3]
        if cands:
            cands.sort(key=lambda a: (a.video_made is True, a.t))
            a = cands[-1]
            a.matched, a.made, a.value = True, True, pts
            continue
        # basket with no detected shot: credit the last player of that team who had the ball
        hs = [h for h in holds if h.team == team and b["t"] - window <= h.t1 <= b["t"] - 0.3]
        h = hs[-1] if hs else None
        xy = None
        if h is not None:
            for fi in range(h.f1, max(h.f0 - 1, h.f1 - int(fps)), -1):
                rr = [r for r in fx.rows(fi) if int(r[1]) == h.tid and not np.isnan(r[7])]
                if rr:
                    xy = (float(rr[0][7]), float(rr[0][8]))
                    break
        hoop = attack_hoop(team, b["t"])
        if xy is None and hoop is not None:   # unknown spot: put it at the rim side of the attacked hoop
            hx, hy = court.hoops[hoop]
            xy = (hx + (4 if hoop == 0 else -4), hy)
        sid += 1
        shots.append({"id": sid, "t": round(h.t1 if h else b["t"] - 5, 2), "player": player_of.get(h.tid) if h else f"{team}?",
                      "team": team, "x": round(xy[0], 1) if xy else None, "y": round(xy[1], 1) if xy else None,
                      "made": True, "free_throw": False, "kind": None, "assist": None,
                      "hoop": hoop, "loc_ok": False, "value_override": pts, "source": "scoreboard"})
    for a in attempts:
        if a.hold.team is None:
            continue
        made = a.made if have_overlay else bool(a.video_made)
        sid += 1
        xy = a.xy
        loc_ok = xy is not None
        if xy is None:
            hoop = a.hoop if a.hoop is not None else attack_hoop(a.hold.team, a.t)
            if hoop is None:
                continue
            hx, hy = court.hoops[hoop]
            xy = (hx + (8 if hoop == 0 else -8), hy)
        shots.append({"id": sid, "t": round(a.t, 2), "player": player_of.get(a.hold.tid, f"{a.hold.team}?"),
                      "team": a.hold.team, "x": round(xy[0], 1), "y": round(xy[1], 1), "made": bool(made),
                      "free_throw": False, "kind": None, "assist": None,
                      "hoop": a.hoop if a.hoop is not None else attack_hoop(a.hold.team, a.t),
                      "loc_ok": loc_ok, "value_override": a.value, "source": "video",
                      "_attempt": a})

    # ---------------- free throws (+1 on the overlay)
    for b in baskets:
        if int(b["points"]) != 1:
            continue
        team = "A" if b["side"] == "left" else "B"
        hoop = attack_hoop(team, b["t"])
        shooter, xy = f"{team}?", None
        if hoop is not None:
            hx, hy = court.hoops[hoop]
            fx_x = hx + (court.ft_line - court.hoop_offset) * (1 if hoop == 0 else -1)
            xy = (round(fx_x, 1), hy)
            best = defaultdict(list)
            f_lo = int(np.searchsorted(fx.t, b["t"] - 20))
            f_hi = int(np.searchsorted(fx.t, b["t"]))
            for fi in range(f_lo, f_hi, max(1, int(fps / 2))):
                for r in fx.rows(fi):
                    tid = int(r[1])
                    if team_of.get(tid) == team and not np.isnan(r[7]):
                        best[tid].append(math.hypot(r[7] - fx_x, r[8] - hy))
            ranked = sorted(((np.median(v), k) for k, v in best.items() if len(v) >= 3))
            if ranked and ranked[0][0] < 5:
                shooter = player_of.get(ranked[0][1], f"{team}?")
        sid += 1
        shots.append({"id": sid, "t": round(b["t"] - 3, 2), "player": shooter, "team": team,
                      "x": xy[0] if xy else None, "y": xy[1] if xy else None, "made": True, "free_throw": True,
                      "kind": None, "assist": None, "hoop": hoop, "loc_ok": xy is not None, "source": "scoreboard"})

    # ---------------- assists, shot kind
    hold_idx = {id(h): i for i, h in enumerate(holds)}
    for s in shots:
        a = s.pop("_attempt", None)
        if a is None:
            continue
        i = hold_idx[id(a.hold)]
        prev = holds[i - 1] if i > 0 else None
        passed = prev is not None and prev.team == a.hold.team and prev.tid != a.hold.tid and a.hold.t0 - prev.t1 <= 1.5
        s["kind"] = "catch" if passed and a.hold.dur <= 1.2 else "dribble"
        if s["made"] and passed and a.hold.dur <= 3.0:
            s["assist"] = player_of.get(prev.tid)
        if a.near_rim_t:
            s["rim_t"] = round(a.near_rim_t, 2)

    # ---------------- rebounds
    events = []
    for s in shots:
        if s["made"] or s["free_throw"]:
            continue
        t_r = s.get("rim_t", s["t"] + 1.0)
        nxt = [h for h in holds if h.t0 > t_r - 0.2 and h.t0 < t_r + 6 and h.team is not None and h.dur >= 0.25]
        if nxt:
            h = nxt[0]
            events.append({"t": round(h.t0, 2), "type": "rebound", "player": player_of.get(h.tid, f"{h.team}?"),
                           "team": h.team, "offensive": h.team == s["team"]})

    # ---------------- touches, turnovers, steals
    shot_times = sorted((s["t"], s["team"]) for s in shots)
    basket_times = [(b["t"], "A" if b["side"] == "left" else "B") for b in baskets]
    for h in holds:
        if h.team is None:
            continue
        events.append({"t": round(h.t0, 2), "type": "touch", "player": player_of.get(h.tid, f"{h.team}?"),
                       "team": h.team, "dur": round(h.dur + 1 / fps, 2)})
    for h1, h2 in zip(holds, holds[1:]):
        if h1.team is None or h2.team is None or h1.team == h2.team:
            continue
        gap = h2.t0 - h1.t1
        if gap > 3.0:
            continue
        if any(h1.t1 - 0.5 <= t <= h2.t0 + 0.5 for t, _ in shot_times):
            continue
        if any(h1.t1 - 3 <= t <= h2.t0 + 12 and tm == h1.team for t, tm in basket_times):
            continue
        events.append({"t": round(h1.t1, 2), "type": "turnover", "player": player_of.get(h1.tid, f"{h1.team}?"), "team": h1.team})
        if gap <= 1.0:
            events.append({"t": round(h2.t0, 2), "type": "steal", "player": player_of.get(h2.tid, f"{h2.team}?"), "team": h2.team})

    # ---------------- possessions (team with the ball)
    poss = []
    for h in holds:
        if h.team is None:
            continue
        if poss and poss[-1]["team"] == h.team and h.t0 - poss[-1]["t1"] <= 3:
            poss[-1]["t1"] = h.t1
        else:
            poss.append({"t0": h.t0, "t1": h.t1, "team": h.team})

    for s in shots:
        s["period"] = period_of(s["t"], starts) + 1
    shots.sort(key=lambda s: s["t"])
    for i, s in enumerate(shots, 1):
        s["id"] = i
    return {"shots": shots, "events": sorted(events, key=lambda e: e["t"]), "possessions": poss,
            "period_starts": starts, "directions": {f"{k[0]}{k[1]}": v for k, v in dirs.items()},
            "n_holds": len(holds), "n_attempts": len(attempts),
            "rims_known": float(np.mean((~np.isnan(rims[:, :, 0])).any(axis=1))) if fx.N else 0.0}

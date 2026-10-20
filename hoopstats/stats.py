"""Turn detected events into per-player and per-team statistics.

Input is the `game` dict produced by events.py (or by the sample generator):

    game = {
      "meta": {...}, "court": {...},
      "teams":   [{"id": "A", "name": "...", "color": "#hex"}, ...],
      "players": [{"id": "A10", "team": "A", "number": "10", "name": "..."}],
      "shots":   [{"id", "t", "period", "player", "team", "x", "y", "made",
                   "free_throw", "kind": "catch"|"dribble"|None, "assist": pid|None,
                   "hoop": 0|1|None, "loc_ok": bool}],
      "events":  [{"t", "type": "rebound"|"turnover"|"steal"|"touch", "player", "team",
                   "offensive": bool, "dur": s}],
      "possessions": [{"t0", "t1", "team"}],
      "positions": {"t": [...], "players": {pid: [[x, y] | None, ...]}},  # common time grid
      "tracked_s": {pid: seconds},
      "overlay": [{"t", "A": int, "B": int}],      # ScoreSync score readings (optional)
    }

`compute()` adds "classified" shot info and returns game with "stats" filled in.
"""
from __future__ import annotations

import math
from collections import defaultdict

from .court import Court, ZONES, AREAS

MPH_PER_FTPS = 0.681818


def _pct(m: int, a: int) -> float | None:
    return round(100.0 * m / a, 1) if a else None


def classify_shots(game: dict, court: Court) -> None:
    for s in game["shots"]:
        info = court.classify(s["x"], s["y"], s.get("hoop"), bool(s.get("free_throw")))
        s.update(info)
        vo = s.get("value_override")
        if vo in (2, 3) and not s.get("free_throw"):
            # the scoreboard knows better than an approximate location
            s["value"] = vo
            if vo == 3 and s["zone"] != "three":
                s["zone"] = "three"
            elif vo == 2 and s["zone"] == "three":
                s["zone"] = "mid"


def _shot_line(shots: list[dict]) -> dict:
    fg = [s for s in shots if not s.get("free_throw")]
    ft = [s for s in shots if s.get("free_throw")]
    three = [s for s in fg if s["value"] == 3]
    fgm = sum(s["made"] for s in fg)
    tpm = sum(s["made"] for s in three)
    ftm = sum(s["made"] for s in ft)
    pts = sum(s["value"] for s in shots if s["made"])
    zones = {}
    for z in ZONES:
        zs = [s for s in fg if s["zone"] == z]
        m = sum(s["made"] for s in zs)
        zones[z] = {"fga": len(zs), "fgm": m, "pct": _pct(m, len(zs)),
                    "share": _pct(len(zs), len(fg))}
    areas = {}
    for a in AREAS:
        as_ = [s for s in fg if s["area"] == a]
        m = sum(s["made"] for s in as_)
        areas[a] = {"fga": len(as_), "fgm": m, "pct": _pct(m, len(as_)), "share": _pct(len(as_), len(fg))}
    kinds = {}
    for k in ("catch", "dribble"):
        ks = [s for s in fg if s.get("kind") == k]
        m = sum(s["made"] for s in ks)
        kinds[k] = {"fga": len(ks), "fgm": m, "pct": _pct(m, len(ks))}
    left = sum(1 for s in fg if s["area"] in ("left_corner", "left_wing"))
    right = sum(1 for s in fg if s["area"] in ("right_corner", "right_wing"))
    dists = [s["dist_ft"] for s in fg if s.get("loc_ok", True)]
    return {
        "fga": len(fg), "fgm": fgm, "fg_pct": _pct(fgm, len(fg)),
        "fg3a": len(three), "fg3m": tpm, "fg3_pct": _pct(tpm, len(three)),
        "fg2a": len(fg) - len(three), "fg2m": fgm - tpm, "fg2_pct": _pct(fgm - tpm, len(fg) - len(three)),
        "fta": len(ft), "ftm": ftm, "ft_pct": _pct(ftm, len(ft)),
        "pts": pts,
        "zones": zones, "areas": areas, "kinds": kinds,
        "left_share": _pct(left, left + right), "right_share": _pct(right, left + right),
        "avg_dist_ft": round(sum(dists) / len(dists), 1) if dists else None,
    }


def _movement(game: dict, court: Court) -> dict[str, dict]:
    pos = game.get("positions") or {}
    ts = pos.get("t") or []
    out: dict[str, dict] = {}
    poss = sorted(game.get("possessions", []), key=lambda p: p["t0"])

    def team_with_ball(t: float) -> str | None:
        for p in poss:
            if p["t0"] <= t <= p["t1"]:
                return p["team"]
        return None

    team_of = {p["id"]: p["team"] for p in game["players"]}
    tracks = pos.get("players", {})
    spacing = defaultdict(list)     # offense: nearest teammate
    openness = defaultdict(list)    # offense: nearest defender
    pressure = defaultdict(list)    # defense: nearest opponent
    for i, t in enumerate(ts):
        on = {pid: tr[i] for pid, tr in tracks.items() if i < len(tr) and tr[i] is not None}
        if len(on) < 2:
            continue
        tb = team_with_ball(t)
        for pid, (x, y) in on.items():
            mates = [math.dist((x, y), q) for o, q in on.items() if o != pid and team_of.get(o) == team_of.get(pid)]
            opps = [math.dist((x, y), q) for o, q in on.items() if team_of.get(o) not in (None, team_of.get(pid))]
            if tb == team_of.get(pid):
                if mates:
                    spacing[pid].append(min(mates))
                if opps:
                    openness[pid].append(min(opps))
            elif tb is not None and opps:
                pressure[pid].append(min(opps))

    for pid, tr in tracks.items():
        dist = 0.0
        speeds = []
        paint = 0
        seen = 0
        prev = None
        for i, p in enumerate(tr):
            if p is None or i >= len(ts):
                prev = None
                continue
            seen += 1
            hoop = court.nearest_hoop(*p)
            if court.in_paint(p[0], p[1], hoop):
                paint += 1
            if prev is not None:
                dt = ts[i] - ts[prev[0]]
                if 0 < dt <= 1.5:
                    d = math.dist(p, prev[1])
                    v = d / dt
                    if v < 30:          # > ~20 mph is a registration glitch, not a kid
                        dist += d
                        speeds.append(v)
            prev = (i, p)
        tracked = game.get("tracked_s", {}).get(pid) or (seen * (ts[1] - ts[0]) if len(ts) > 1 else 0)
        speeds.sort()
        top = speeds[int(0.98 * (len(speeds) - 1))] if speeds else 0.0
        mean = lambda xs: round(sum(xs) / len(xs), 1) if xs else None
        out[pid] = {
            "tracked_min": round(tracked / 60, 1),
            "dist_ft": round(dist),
            "dist_per_min": round(dist / (tracked / 60), 1) if tracked else None,
            "avg_speed_mph": round((sum(speeds) / len(speeds)) * MPH_PER_FTPS, 1) if speeds else None,
            "top_speed_mph": round(top * MPH_PER_FTPS, 1),
            "paint_pct": _pct(paint, seen),
            "spacing_ft": mean(spacing[pid]),
            "open_ft": mean(openness[pid]),
            "pressure_ft": mean(pressure[pid]),
        }
    return out


def _heat(game: dict, court: Court, cell: float = 6.0) -> dict[str, list]:
    """Occupancy grid per player: list of [col, row, share%]."""
    pos = game.get("positions") or {}
    cols, rows = math.ceil(court.length / cell), math.ceil(court.width / cell)
    out = {}
    for pid, tr in (pos.get("players") or {}).items():
        grid = defaultdict(int)
        n = 0
        for p in tr:
            if p is None:
                continue
            c = min(int(p[0] // cell), cols - 1)
            r = min(int(p[1] // cell), rows - 1)
            grid[(c, r)] += 1
            n += 1
        out[pid] = [[c, r, round(100 * k / n, 2)] for (c, r), k in grid.items()] if n else []
    return out


def compute(game: dict) -> dict:
    court = Court.from_config(game.get("court"))
    game["court"] = court.to_dict()
    classify_shots(game, court)

    ev = defaultdict(lambda: defaultdict(float))
    for e in game.get("events", []):
        pid = e.get("player")
        if not pid:
            continue
        typ = e["type"]
        if typ == "rebound":
            ev[pid]["oreb" if e.get("offensive") else "dreb"] += 1
        elif typ == "turnover":
            ev[pid]["tov"] += 1
        elif typ == "steal":
            ev[pid]["stl"] += 1
        elif typ == "touch":
            ev[pid]["touches"] += 1
            ev[pid]["touch_s"] += e.get("dur", 0.0)
    for s in game["shots"]:
        if s.get("assist") and s["made"]:
            ev[s["assist"]]["ast"] += 1

    move = _movement(game, court)
    heat = _heat(game, court)

    players = {}
    for p in game["players"]:
        pid = p["id"]
        shots = [s for s in game["shots"] if s["player"] == pid]
        e = ev[pid]
        line = _shot_line(shots)
        touches = int(e["touches"])
        line.update({
            "oreb": int(e["oreb"]), "dreb": int(e["dreb"]), "reb": int(e["oreb"] + e["dreb"]),
            "ast": int(e["ast"]), "tov": int(e["tov"]), "stl": int(e["stl"]),
            "touches": touches,
            "avg_touch_s": round(e["touch_s"] / touches, 1) if touches else None,
            "tov_per_touch": _pct(int(e["tov"]), touches),
        })
        line.update(move.get(pid, {}))
        line["heat"] = heat.get(pid, [])
        players[pid] = line

    teams = {}
    for t in game["teams"]:
        tid = t["id"]
        shots = [s for s in game["shots"] if s["team"] == tid]
        line = _shot_line(shots)
        pids = [p["id"] for p in game["players"] if p["team"] == tid]
        for k in ("oreb", "dreb", "reb", "ast", "tov", "stl", "touches"):
            line[k] = sum(players[pid][k] for pid in pids)
        made_fg = [s for s in shots if s["made"] and not s.get("free_throw")]
        line["assisted_pct"] = _pct(sum(1 for s in made_fg if s.get("assist")), len(made_fg))
        opp_fg_misses = sum(1 for s in game["shots"] if s["team"] != tid and not s["made"] and not s.get("free_throw"))
        own_fg_misses = sum(1 for s in shots if not s["made"] and not s.get("free_throw"))
        line["oreb_pct"] = _pct(line["oreb"], own_fg_misses)
        line["dreb_pct"] = _pct(line["dreb"], opp_fg_misses)
        overlay = game.get("overlay") or []
        line["overlay_pts"] = overlay[-1].get(tid) if overlay else None
        teams[tid] = line

    # running score from our own detected makes, for the score-flow chart
    flow, run = [], {t["id"]: 0 for t in game["teams"]}
    for s in sorted(game["shots"], key=lambda s: s["t"]):
        if s["made"]:
            run[s["team"]] += s["value"]
            flow.append({"t": s["t"], **run})

    game["stats"] = {"players": players, "teams": teams, "flow": flow}
    return game

"""Generate a report.html filled with SAMPLE (made-up) data, to preview the report design.

    python scripts/make_sample_report.py --out runs/sample/report.html
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hoopstats.court import Court  # noqa: E402
from hoopstats import stats, insights  # noqa: E402
from hoopstats.report.build import build  # noqa: E402

TEAMS = [
    {"id": "A", "name": "BSB silver", "color": "#8ccfee", "mark": "#2a8fc9"},
    {"id": "B", "name": "east bay elite", "color": "#1f2a44", "mark": "#d0582f"},
]
ROSTERS = {
    "A": [("3", "Aarav"), ("5", "Ben"), ("10", "Caleb"), ("11", "Dev"), ("14", "Ethan"), ("22", "Felix"), ("23", "Gabe"), ("31", "Hari")],
    "B": [("1", "Ian"), ("2", "Jalen"), ("4", "Kai"), ("7", "Leo"), ("12", "Marcus"), ("15", "Nate"), ("20", "Omar"), ("24", "Rohan")],
}
# per-player tendencies: usage, zone mix (rim, paint, mid, three), skill, side bias (+ = left)
STYLE = {
    "A3": (1.6, (.35, .15, .30, .20), 1.05, -.6), "A5": (1.0, (.20, .10, .35, .35), .95, 0),
    "A10": (2.0, (.20, .10, .45, .25), 1.0, -.4), "A11": (.7, (.55, .30, .15, 0), .9, .2),
    "A14": (1.2, (.40, .25, .25, .10), 1.1, .5), "A22": (.6, (.30, .20, .30, .20), .8, 0),
    "A23": (.9, (.15, .10, .30, .45), 1.0, .3), "A31": (.4, (.60, .30, .10, 0), .85, 0),
    "B1": (1.8, (.30, .10, .25, .35), 1.1, .2), "B2": (1.2, (.45, .20, .25, .10), 1.0, -.3),
    "B4": (.8, (.20, .15, .40, .25), .9, 0), "B7": (1.5, (.40, .20, .30, .10), 1.05, .6),
    "B12": (.9, (.55, .35, .10, 0), 1.0, 0), "B15": (.6, (.25, .15, .30, .30), .85, -.2),
    "B20": (1.0, (.20, .10, .30, .40), .95, 0), "B24": (.5, (.50, .30, .20, 0), .9, 0),
}
BASE_P = {"rim": .52, "paint": .40, "mid": .31, "three": .24}
DUR = 2242.0


def attack_hoop(team: str, t: float) -> int:
    first_half = t < DUR / 2
    return (1 if team == "A" else 0) if first_half else (0 if team == "A" else 1)


def spot(court: Court, hoop: int, zone: str, side: float, rng: random.Random) -> tuple[float, float]:
    """Random court point in `zone` toward `hoop`; side>0 biases to the offense-left."""
    for _ in range(200):
        if zone == "rim":
            r = rng.uniform(1, 5.5)
        elif zone == "paint":
            r = rng.uniform(6.5, 12.5)
        elif zone == "mid":
            r = rng.uniform(9, court.three_pt - .8)
        else:
            r = rng.uniform(court.three_pt + .3, court.three_pt + 4)
        ang = rng.gauss(side * 45, 50)
        ang = max(-88, min(88, ang))
        depth, lat = r * math.cos(math.radians(ang)), r * math.sin(math.radians(ang))
        hx, hy = court.hoops[hoop]
        x, y = (hx - depth, hy + lat) if hoop == 1 else (hx + depth, hy - lat)
        if not court.contains(x, y, -.5):
            continue
        if court.classify(x, y, hoop)["zone"] == zone:
            return round(x, 1), round(y, 1)
    return court.hoops[hoop][0] + (-3 if hoop else 3), 25.0


def thumb(color: str, label: str) -> str:
    svg = (f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 112 150'><rect width='112' height='150' fill='#d9d4c8'/>"
           f"<circle cx='56' cy='34' r='15' fill='#8a6b55'/><rect x='30' y='52' width='52' height='56' rx='10' fill='{color}'/>"
           f"<rect x='36' y='106' width='16' height='38' fill='#333'/><rect x='60' y='106' width='16' height='38' fill='#333'/>"
           f"<text x='56' y='86' font-size='16' font-family='sans-serif' text-anchor='middle' fill='#fff'>{label}</text></svg>")
    import urllib.parse
    return "data:image/svg+xml;utf8," + urllib.parse.quote(svg)


def make_game(seed: int = 7) -> dict:
    rng = random.Random(seed)
    court = Court()
    players = [{"id": f"{t}{n}", "team": t, "number": n, "name": nm} for t, r in ROSTERS.items() for n, nm in r]
    team_pids = {t: [p["id"] for p in players if p["team"] == t] for t in ROSTERS}

    # rotations: 5 on court per team, sub every ~4 minutes, weighted by usage
    on_court = {}
    def lineup(team):
        pids = team_pids[team]
        w = [STYLE[p][0] + .6 for p in pids]
        chosen = set()
        while len(chosen) < 5:
            chosen.add(rng.choices(pids, w)[0])
        return sorted(chosen)

    shots, events, poss_list, overlay = [], [], [], []
    score = {"A": 0, "B": 0}
    t, team = 8.0, "A"
    next_sub = 0
    sid = 0
    while t < DUR - 10:
        if t >= next_sub:
            on_court = {k: lineup(k) for k in ("A", "B")}
            next_sub = t + rng.uniform(180, 300)
        dur = rng.uniform(9, 22)
        t0, t1 = t, min(t + dur, DUR - 5)
        poss_list.append({"t0": t0, "t1": t1, "team": team})
        off = on_court[team]
        defn = on_court["B" if team == "A" else "A"]
        # touches
        holders = []
        for k in range(rng.randint(2, 5)):
            h = rng.choices(off, [STYLE[p][0] for p in off])[0]
            if holders and holders[-1] == h:
                continue
            holders.append(h)
            d = max(.5, rng.gauss(2.2 * (1.5 if STYLE[h][0] > 1.4 else 1), 1.0))
            events.append({"t": t0 + k * 3, "type": "touch", "player": h, "team": team, "dur": round(d, 1)})
        r = rng.random()
        hoop = attack_hoop(team, t1)
        if r < .18:  # turnover
            events.append({"t": t1, "type": "turnover", "player": holders[-1], "team": team})
            if rng.random() < .6:
                events.append({"t": t1, "type": "steal", "player": rng.choice(defn), "team": "B" if team == "A" else "A"})
        elif r < .25:  # fouled -> 2 free throws
            shooter = holders[-1]
            for k in range(2):
                sid += 1
                made = rng.random() < .55 * STYLE[shooter][2]
                ftx = court.hoops[hoop][0] + (-13.75 if hoop else 13.75)
                shots.append({"id": sid, "t": round(t1 + k * 8, 1), "player": shooter, "team": team, "x": ftx, "y": 25.0,
                              "made": made, "free_throw": True, "kind": None, "assist": None, "hoop": hoop})
                if made:
                    score[team] += 1
                    overlay.append({"t": round(t1 + k * 8 + rng.uniform(3, 9), 1), **score})
        else:
            shooter = holders[-1]
            u, mix, skill, side = STYLE[shooter]
            zone = rng.choices(["rim", "paint", "mid", "three"], mix)[0]
            x, y = spot(court, hoop, zone, side, rng)
            p_make = min(.85, BASE_P[zone] * skill * (1.15 if len(holders) > 1 and rng.random() < .5 else 1))
            made = rng.random() < p_make
            kind = "catch" if len(holders) > 1 and rng.random() < .55 else "dribble"
            if kind == "catch":
                made = made or rng.random() < .04
            sid += 1
            ast = holders[-2] if made and len(holders) > 1 and kind == "catch" and rng.random() < .75 else None
            shots.append({"id": sid, "t": round(t1, 1), "player": shooter, "team": team, "x": x, "y": y, "made": made,
                          "free_throw": False, "kind": kind, "assist": ast, "hoop": hoop, "loc_ok": rng.random() > .07})
            info = court.classify(x, y, hoop)
            if made:
                score[team] += info["value"]
                overlay.append({"t": round(t1 + rng.uniform(3, 9), 1), **score})
            else:
                oreb = rng.random() < .3
                reb_team = team if oreb else ("B" if team == "A" else "A")
                pool = off if oreb else defn
                w = [(1 + STYLE[p][1][0] * 2) for p in pool]
                events.append({"t": t1 + 1.5, "type": "rebound", "player": rng.choices(pool, w)[0], "team": reb_team, "offensive": oreb})
                if oreb:
                    t = t1 + 2
                    continue
        team = "B" if team == "A" else "A"
        t = t1 + rng.uniform(2, 5)

    # The scoreboard is the truth; pretend detection missed two made baskets.
    made_idx = [i for i, s in enumerate(shots) if s["made"] and not s["free_throw"]]
    for i in sorted(rng.sample(made_idx, 2), reverse=True):
        shots.pop(i)

    # positions on a 1 s grid for movement / spacing / heatmaps
    ts = [float(i) for i in range(0, int(DUR))]
    pos = {p["id"]: [None] * len(ts) for p in players}
    cur = {p["id"]: [rng.uniform(10, 74), rng.uniform(5, 45)] for p in players}
    pi = 0
    lineups_at = []
    tsub, lu = 0, None
    rng2 = random.Random(seed + 1)
    for i, tt in enumerate(ts):
        while pi < len(poss_list) - 1 and poss_list[pi]["t1"] < tt:
            pi += 1
        if tt >= tsub:
            lu = {k: lineup(k) for k in ("A", "B")}
            tsub = tt + rng2.uniform(180, 300)
        pteam = poss_list[pi]["team"]
        hoop = attack_hoop(pteam, tt)
        hx = court.hoops[hoop][0]
        sgn = -1 if hoop else 1
        # 5-out slots (depth, lateral) re-shuffled each possession and nudged every few seconds
        slots = [(1, 21), (1, -21), (14, 15), (14, -15), (21, 0)]
        prng = random.Random(seed * 1000 + pi * 7 + int(tt // 4))
        prng.shuffle(slots)
        targets = {}
        for j, pid in enumerate(lu[pteam]):
            d, lat = slots[j]
            if STYLE[pid][1][0] > .45 and prng.random() < .5:   # bigs drift to the block
                d, lat = 4, prng.choice((-7, 7))
            targets[pid] = (hx + sgn * d, 25 + lat * (-sgn))
        dteam = "B" if pteam == "A" else "A"
        for j, pid in enumerate(lu[dteam]):
            mx, my = targets[lu[pteam][j]]
            targets[pid] = (mx + (hx - mx) * .2, my + (25 - my) * .2)
        for k in ("A", "B"):
            for pid in lu[k]:
                st = STYLE[pid]
                tx, ty = targets[pid]
                tx += rng2.gauss(0, 2)
                ty += rng2.gauss(0, 2)
                c = cur[pid]
                c[0] += (tx - c[0]) * .35 + rng2.gauss(0, .8) * (1.4 if st[0] > 1.2 else 1)
                c[1] += (ty - c[1]) * .35 + rng2.gauss(0, .8)
                c[0] = min(max(c[0], .5), court.length - .5)
                c[1] = min(max(c[1], .5), court.width - .5)
                if rng2.random() > .12:  # camera doesn't always see everyone
                    pos[pid][i] = [round(c[0], 1), round(c[1], 1)]
    tracked = {pid: sum(1 for p in tr if p is not None) for pid, tr in pos.items()}

    shots.sort(key=lambda s: s["t"])
    for s in shots:
        s["period"] = 1 if s["t"] < DUR / 2 else 2
    return {
        "meta": {
            "sample": True,
            "title": "BSB silver vs east bay elite",
            "date": "Oct 17, 2026",
            "venue": "",
            "period_label": "H",
            "direction_note": "1st half: BSB silver attacks the right basket",
            "source": {"device": "XbotGo Falcon", "resolution": "3840×2160", "fps": 30, "duration_s": DUR,
                       "file": "18:00:21.MP4"},
            "video_url": "../../videos/18%3A00%3A21.MP4",
        },
        "court": court.to_dict(),
        "teams": TEAMS,
        "players": players,
        "shots": shots,
        "events": events,
        "possessions": poss_list,
        "positions": {"t": ts, "players": pos},
        "tracked_s": tracked,
        "overlay": overlay,
        "quality": {"registration_pct": 91, "identified_pct": 84},
        "unidentified": [
            {"track": "track_57", "team": "A", "seconds": 212, "thumb": thumb("#8ccfee", "?")},
            {"track": "track_88", "team": "B", "seconds": 145, "thumb": thumb("#1f2a44", "1?")},
            {"track": "track_131", "team": "A", "seconds": 64, "thumb": thumb("#8ccfee", "2?")},
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/sample/report.html")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    game = make_game(a.seed)
    stats.compute(game)
    insights.attach(game)
    out = build(game, a.out)
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()

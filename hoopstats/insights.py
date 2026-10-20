"""Rule-based coaching insights. Every insight is backed by numbers from stats.py.

Each insight: {"kind": "strength"|"focus", "title": str, "detail": str, "drill": str|None}
Rules only fire with enough attempts to mean something; youth samples are small.
"""
from __future__ import annotations

from .court import ZONE_LABELS


def _f(v, suffix="%"):
    return "–" if v is None else f"{v:g}{suffix}"


def player_insights(st: dict, team_avg: dict) -> list[dict]:
    out: list[dict] = []
    z = st["zones"]
    fga = st["fga"]

    def add(kind, title, detail, drill=None, weight=1.0):
        out.append({"kind": kind, "title": title, "detail": detail, "drill": drill, "w": weight})

    # --- shot selection: mid-range heavy but finishes better at the rim
    if fga >= 6 and (z["mid"]["share"] or 0) >= 30 and z["rim"]["fga"] >= 2 \
            and (z["rim"]["pct"] or 0) - (z["mid"]["pct"] or 0) >= 15:
        add("focus", "Get to the rim more",
            f"{_f(z['rim']['pct'])} at the rim vs {_f(z['mid']['pct'])} from mid-range, "
            f"but mid-range is {_f(z['mid']['share'])} of shots.",
            "Drive-and-finish reps: 1-dribble attacks from the wing into a layup.", 3)

    if z["rim"]["fga"] >= 4 and (z["rim"]["pct"] or 0) < 40:
        add("focus", "Finishing at the rim",
            f"{z['rim']['fgm']}/{z['rim']['fga']} ({_f(z['rim']['pct'])}) within 6 ft.",
            "Mikan drill and contact layups, both hands, 50 makes a session.", 3)

    for zk in ("rim", "paint", "mid", "three"):
        zz = z[zk]
        if zz["fga"] >= 4 and (zz["pct"] or 0) >= 50:
            add("strength", f"Hot from {ZONE_LABELS[zk].split(' (')[0].lower()}",
                f"{zz['fgm']}/{zz['fga']} ({_f(zz['pct'])}).", None, 2)

    if st["fg3a"] >= 4 and (st["fg3_pct"] or 0) < 20:
        add("focus", "Three-point selection",
            f"{st['fg3m']}/{st['fg3a']} from three. Take a step in or attack the closeout.",
            "Shot-fake, one dribble, pull-up from 15 ft.", 2)

    # --- side tendency
    if fga >= 6 and st["left_share"] is not None:
        if st["right_share"] >= 70:
            add("focus", "Develop the left side",
                f"{_f(st['right_share'])} of side shots come from the right.",
                "Left-hand dribble series and left-side layups.", 2)
        elif st["left_share"] >= 70:
            add("focus", "Develop the right side",
                f"{_f(st['left_share'])} of side shots come from the left.",
                "Right-hand dribble series and right-side layups.", 2)

    # --- catch vs dribble
    c, d = st["kinds"]["catch"], st["kinds"]["dribble"]
    if c["fga"] >= 3 and d["fga"] >= 3 and c["pct"] is not None and d["pct"] is not None:
        if c["pct"] - d["pct"] >= 20:
            add("focus", "Better off the catch",
                f"Catch-and-shoot {_f(c['pct'])} vs off-dribble {_f(d['pct'])}.",
                "Move without the ball to get catch-and-shoot looks; work on 1-2 step pull-ups.", 1.5)
        elif d["pct"] - c["pct"] >= 20:
            add("strength", "Creates own shot",
                f"Off-dribble {_f(d['pct'])} vs catch-and-shoot {_f(c['pct'])}.", None, 1.5)

    if st["fta"] >= 4 and (st["ft_pct"] or 0) < 50:
        add("focus", "Free throws", f"{st['ftm']}/{st['fta']} at the line.",
            "Same routine every time; 25 free throws at the end of practice when tired.", 1.5)

    # --- ball handling
    if st.get("avg_touch_s") and st["touches"] >= 8 and st["avg_touch_s"] >= 3.5:
        add("focus", "Quicker decisions",
            f"Holds the ball {st['avg_touch_s']} s per touch (team avg {team_avg.get('avg_touch_s', '–')} s).",
            "0.5-second rule in practice: catch → shoot, pass or drive.", 2)
    if st["touches"] >= 8 and (st.get("tov_per_touch") or 0) >= 15:
        add("focus", "Protect the ball",
            f"{st['tov']} turnovers on {st['touches']} touches.",
            "Two-ball dribbling and pivot-and-pass under pressure.", 2)

    # --- movement / effort
    dpm, tdpm = st.get("dist_per_min"), team_avg.get("dist_per_min")
    if dpm and tdpm and st.get("tracked_min", 0) >= 5:
        if dpm < 0.8 * tdpm:
            add("focus", "Move more off the ball",
                f"Covers {dpm:g} ft/min vs team {tdpm:g} ft/min.",
                "Cut-and-replace drill; sprint the floor in transition.", 1.5)
        elif dpm > 1.15 * tdpm:
            add("strength", "High motor", f"Covers {dpm:g} ft/min vs team {tdpm:g} ft/min.", None, 1)

    if st.get("spacing_ft") is not None and st["spacing_ft"] < 9:
        add("focus", "Spacing on offense",
            f"Averages {st['spacing_ft']} ft from the nearest teammate on offense.",
            "5-out spacing: stay 12-15 ft apart, fill corners.", 1.5)

    pr, tpr = st.get("pressure_ft"), team_avg.get("pressure_ft")
    if pr is not None and tpr is not None and pr > tpr + 2:
        add("focus", "Closer on defense",
            f"Stays {pr} ft from the nearest opponent on defense (team {tpr} ft).",
            "Closeout drill: sprint, chop feet, hand up.", 1.5)

    if st["oreb"] >= 3:
        add("strength", "Crashes the offensive glass", f"{st['oreb']} offensive rebounds.", None, 1.5)
    if st["ast"] >= 3:
        add("strength", "Playmaker", f"{st['ast']} assists (estimated).", None, 1.5)
    if fga <= 2 and st.get("tracked_min", 0) >= 10:
        add("focus", "Be more aggressive",
            f"Only {fga} shot{'s' if fga != 1 else ''} in {st['tracked_min']:g} tracked minutes.",
            "Look for the shot first when open; attack closeouts.", 1)

    out.sort(key=lambda i: -i["w"])
    for i in out:
        i.pop("w")
    return out[:6]


def team_insights(team_name: str, st: dict) -> list[dict]:
    out = []
    z = st["zones"]
    best = max((k for k in z if z[k]["fga"] >= 5), key=lambda k: z[k]["pct"] or 0, default=None)
    worst = min((k for k in z if z[k]["fga"] >= 5), key=lambda k: z[k]["pct"] or 0, default=None)
    if best:
        out.append({"kind": "strength", "title": f"Best zone: {ZONE_LABELS[best]}",
                    "detail": f"{z[best]['fgm']}/{z[best]['fga']} ({_f(z[best]['pct'])}).", "drill": None})
    if worst and worst != best:
        out.append({"kind": "focus", "title": f"Coldest zone: {ZONE_LABELS[worst]}",
                    "detail": f"{z[worst]['fgm']}/{z[worst]['fga']} ({_f(z[worst]['pct'])}), "
                              f"{_f(z[worst]['share'])} of shots.", "drill": None})
    if st.get("assisted_pct") is not None:
        kind = "strength" if st["assisted_pct"] >= 50 else "focus"
        out.append({"kind": kind, "title": "Ball movement",
                    "detail": f"{_f(st['assisted_pct'])} of made field goals were assisted (est.).",
                    "drill": None if kind == "strength" else "Add a 3-pass rule in scrimmages."})
    if st.get("oreb_pct") is not None:
        kind = "strength" if st["oreb_pct"] >= 35 else "focus"
        out.append({"kind": kind, "title": "Offensive rebounding",
                    "detail": f"Rebounded {_f(st['oreb_pct'])} of own misses.",
                    "drill": None if kind == "strength" else "Assign 2 crashers on every shot."})
    return out


def team_averages(players: dict, pids: list[str]) -> dict:
    def avg(key):
        vals = [players[p][key] for p in pids if players[p].get(key) is not None]
        return round(sum(vals) / len(vals), 1) if vals else None
    return {k: avg(k) for k in ("dist_per_min", "avg_touch_s", "pressure_ft", "spacing_ft")}


def attach(game: dict) -> dict:
    players = game["stats"]["players"]
    for t in game["teams"]:
        pids = [p["id"] for p in game["players"] if p["team"] == t["id"]]
        avg = team_averages(players, pids)
        for pid in pids:
            players[pid]["insights"] = player_insights(players[pid], avg)
        game["stats"]["teams"][t["id"]]["insights"] = team_insights(t["name"], game["stats"]["teams"][t["id"]])
    return game

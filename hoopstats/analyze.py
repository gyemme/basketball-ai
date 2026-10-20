"""Stage 2: turn the cached extraction into players, events, stats, insights and the report."""
from __future__ import annotations

import base64
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from . import events as E
from . import identify as I
from . import insights, stats
from .court import Court
from .extract import load
from .report.build import build as build_report
from .scoreboard import clean_timeline, increments

FALLBACK_COLORS = {"A": "#2a8fc9", "B": "#d0582f"}


def _thumb(cache: Path, tid: int) -> str | None:
    p = cache / "thumbs" / f"{tid}.jpg"
    if not p.exists():
        return None
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode()


def _mark_color(lab) -> str:
    """A readable marker colour for the team: the jersey colour, pushed away from white/black."""
    hexc = I.lab_to_hex(lab)
    L = float(lab[0])
    if L > 200 or L < 60:   # near white / near black jerseys -> use defaults
        return None
    return hexc


def run(run_dir: Path, cfg: dict, overrides: Path | None = None, log=print) -> dict:
    cache = run_dir / "cache"
    ex, meta = load(cache)
    fps = float(meta["fps"])
    court = Court.from_config(cfg.get("court"))
    size = (int(meta["width"]), int(meta["height"]))
    t = ex["t"]
    N = len(t)
    log(f"  {N} analysed frames, {len(np.unique(ex['people'][:, 1])) if len(ex['people']) else 0} tracks")

    # ---------------- scoreboard
    ov = ex["overlay"]
    raw = [(float(r[0]), None if r[1] < 0 else int(r[1]), None if r[2] < 0 else int(r[2]), float(r[3])) for r in ov]
    timeline = clean_timeline(raw) if raw else []
    baskets, gaps = increments(timeline)
    log(f"  scoreboard: {len(timeline)} score changes"
        + (f", final {timeline[-1]['left']}-{timeline[-1]['right']}" if timeline else " (overlay not found)"))

    # ---------------- teams
    tracks = I.track_table(ex, court, fps)
    clusters, centers = I.assign_teams(tracks, cfg)
    ovr = I.load_overrides(overrides)
    team_of_c = {tid: (str(c) if c >= 0 else None) for tid, c in clusters.items()}

    # first pass with clusters to learn which cluster is which overlay side
    fx = E.FrameIndex(ex)
    if all(tm.get("jersey") for tm in cfg["teams"]):
        side = {"0": "A", "1": "B"}
    else:
        ball = E.interpolate_ball(fx.ball, max_gap=max(2, int(0.3 * fps)))
        h_arr = E.holders(fx, ball, team_of_c)
        holds = E.smooth_holds(h_arr, t, team_of_c, max(2, int(0.2 * fps)), max(2, int(0.35 * fps)))
        rims = E.rim_positions(fx, court, *size)
        atts = E.detect_attempts(fx, ball, holds, rims, fps, court)
        side = E.map_sides(atts, holds, baskets, float(cfg["shots"]["make_window_s"]))
    team_of = {tid: (side[c] if c is not None else None) for tid, c in team_of_c.items()}
    team_color = {side[str(k)]: c for k, c in enumerate(centers)}

    # ---------------- jersey numbers -> players
    team_tids = [tid for tid, tm in team_of.items() if tm and tracks[tid]["seconds"] >= 2]
    votes = I.read_numbers(cache, sorted(team_tids, key=lambda k: -tracks[k]["seconds"]),
                           cfg["analysis"].get("ocr", "auto"), log=log)
    rosters = {"A": set(cfg["teams"][0]["roster"]), "B": set(cfg["teams"][1]["roster"])}
    player_of: dict[int, str] = {}
    number_of: dict[str, str] = {}
    unidentified = []
    for tid in team_tids:
        tm = team_of[tid]
        num, share = I.best_number(votes.get(tid, Counter()), rosters[tm] or None)
        if num:
            pid = f"{tm}{num}"
            player_of[tid] = pid
            number_of[pid] = num
        else:
            player_of[tid] = f"{tm}?"
            if tracks[tid]["seconds"] >= 8:
                unidentified.append(tid)
    for tid, target in ovr.items():
        if target.lower() in ("ignore", "none", "ref", "referee"):
            team_of[tid] = None
            player_of.pop(tid, None)
            continue
        tm, num = target[0].upper(), target[1:]
        team_of[tid] = tm
        player_of[tid] = f"{tm}{num}"
        number_of[f"{tm}{num}"] = num
        if tid in unidentified:
            unidentified.remove(tid)
    for tid, tm in team_of.items():
        if tm and tid not in player_of:
            player_of[tid] = f"{tm}?"

    # ---------------- events
    ev = E.build(fx, ex, team_of, player_of, baskets, gaps, court, fps, size, cfg)
    log(f"  {ev['n_holds']} ball possessions, {ev['n_attempts']} shot attempts detected, "
        f"{len(ev['shots'])} shots in report")

    # ---------------- players list
    names = {"A": cfg["teams"][0]["roster"], "B": cfg["teams"][1]["roster"]}
    used = set(player_of.values()) | {s["player"] for s in ev["shots"]} | {e["player"] for e in ev["events"]}
    players = []
    for pid in sorted(used, key=lambda p: (p[0], 999 if p.endswith("?") else int(p[1:]) if p[1:].isdigit() else 998)):
        if not pid or pid[0] not in "AB":
            continue
        num = number_of.get(pid, "?")
        players.append({"id": pid, "team": pid[0], "number": num,
                        "name": names[pid[0]].get(num, "") if num != "?" else "Unidentified"})
    for tm, rr in names.items():   # roster players never seen still get a row
        for num, nm in rr.items():
            pid = f"{tm}{num}"
            if pid not in {p["id"] for p in players}:
                players.append({"id": pid, "team": tm, "number": num, "name": nm})

    # ---------------- positions on a 1 s grid, tracked time
    grid_t = np.arange(float(t[0]) if N else 0, float(t[-1]) + 1 if N else 1, 1.0)
    pos = {p["id"]: [None] * len(grid_t) for p in players}
    tracked = Counter()
    P = ex["people"]
    good = (ex["Hconf"][P[:, 0].astype(int)] >= 0.25) if len(P) else np.zeros(0, bool)
    buckets: dict[tuple[str, int], list] = {}
    for r, g in zip(P, good):
        pid = player_of.get(int(r[1]))
        if not pid:
            continue
        tracked[pid] += 1 / fps
        if not g or np.isnan(r[7]) or not court.contains(r[7], r[8], 3):
            continue
        k = int(t[int(r[0])] - grid_t[0])
        buckets.setdefault((pid, k), []).append((r[7], r[8]))
    for (pid, k), v in buckets.items():
        if pid in pos and 0 <= k < len(grid_t) and not pid.endswith("?"):
            a = np.median(np.array(v), axis=0)
            pos[pid][k] = [round(float(a[0]), 1), round(float(a[1]), 1)]

    # ---------------- shots: value override from the scoreboard, drop unknown locations
    shots = []
    for s in ev["shots"]:
        if s["x"] is None:
            continue
        shots.append(s)

    teams = []
    for i, tm in enumerate(("A", "B")):
        c = team_color.get(tm)
        jersey = cfg["teams"][i].get("jersey") or (I.lab_to_hex(c) if c is not None else FALLBACK_COLORS[tm])
        mark = (_mark_color(c) if c is not None else None) or FALLBACK_COLORS[tm]
        teams.append({"id": tm, "name": cfg["teams"][i]["name"], "color": jersey, "mark": mark})
    if teams[0]["mark"] == teams[1]["mark"]:
        teams[1]["mark"] = FALLBACK_COLORS["B"]

    overlay_tl = [{"t": e["t"], "A": e["left"], "B": e["right"]} for e in timeline]
    vids = meta["videos"]
    v0 = vids[0]
    import os
    import urllib.parse
    rel = os.path.relpath(Path(v0["path"]).resolve(), run_dir.resolve())
    video_url = urllib.parse.quote(rel.replace(os.sep, "/"))
    dur = sum(v["duration"] for v in vids)
    reg_pct = round(100 * float(np.mean(ex["Hconf"] >= 0.25)), 0) if N else 0
    on_time = sum(v for k, v in tracked.items())
    id_time = sum(v for k, v in tracked.items() if not k.endswith("?"))
    game = {
        "meta": {
            "title": cfg["game"].get("title") or f"{teams[0]['name']} vs {teams[1]['name']}",
            "date": cfg["game"].get("date") or (v0.get("creation_time") or "")[:10],
            "venue": cfg["game"].get("venue") or "",
            "period_label": "P" if len(ev["period_starts"]) != 2 else "H",
            "direction_note": _direction_note(ev, teams),
            "source": {"device": "XbotGo", "resolution": f"{v0['width']}×{v0['height']}",
                       "fps": round(v0["fps"]), "duration_s": dur, "file": Path(v0["path"]).name},
            "video_url": video_url,
            "analysis": {"fps": fps, "start": meta.get("start"), "end": meta.get("end")},
        },
        "court": court.to_dict(),
        "teams": teams,
        "players": players,
        "shots": shots,
        "events": ev["events"],
        "possessions": ev["possessions"],
        "positions": {"t": grid_t.tolist(), "players": pos},
        "tracked_s": dict(tracked),
        "overlay": overlay_tl,
        "quality": {
            "registration_pct": int(reg_pct),
            "identified_pct": int(round(100 * id_time / on_time)) if on_time else None,
            "shots_detected_video": ev["n_attempts"],
            "rim_visible_pct": int(round(100 * ev["rims_known"])),
            "score_gaps": gaps,
        },
        "unidentified": [{"track": f"track_{tid}", "team": team_of[tid], "seconds": round(tracks[tid]["seconds"]),
                          "thumb": _thumb(cache, tid)} for tid in sorted(unidentified, key=lambda k: -tracks[k]["seconds"])[:24]],
    }
    game["track_map"] = {str(tid): {"team": team_of.get(tid), "player": player_of.get(tid)} for tid in tracks}
    stats.compute(game)
    insights.attach(game)
    write_outputs(run_dir, game)
    return game


def _direction_note(ev, teams) -> str:
    d = ev["directions"]
    if "A0" not in d:
        return ""
    side = "right" if d["A0"] == 1 else "left"
    lab = "1st half" if len(ev["period_starts"]) == 2 else "Start"
    return f"{lab}: {teams[0]['name']} attacks the {side} basket"


def write_outputs(run_dir: Path, game: dict) -> None:
    (run_dir / "game.json").write_text(json.dumps(game, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    build_report({k: v for k, v in game.items() if k != "track_map"}, run_dir / "report.html")
    st = game["stats"]
    pname = {p["id"]: p for p in game["players"]}
    tname = {t["id"]: t["name"] for t in game["teams"]}
    with open(run_dir / "player_stats.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["team", "number", "name", "tracked_min", "pts", "fgm", "fga", "fg_pct", "fg3m", "fg3a", "ftm", "fta",
                "oreb", "dreb", "ast", "tov", "stl", "touches", "avg_touch_s", "dist_per_min", "top_speed_mph",
                "spacing_ft", "pressure_ft"]
        w.writerow(cols)
        for p in game["players"]:
            s = st["players"][p["id"]]
            w.writerow([tname[p["team"]], p["number"], p["name"]] + [s.get(c) for c in cols[3:]])
    with open(run_dir / "shots.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["id", "t", "period", "team", "player", "made", "value", "zone", "area", "dist_ft", "x", "y",
                "kind", "assist", "free_throw", "loc_ok", "source"]
        w.writerow(cols)
        for s in game["shots"]:
            w.writerow([s.get(c) for c in cols])
    with open(run_dir / "events.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "type", "team", "player", "offensive", "dur"])
        for e in game["events"]:
            w.writerow([e["t"], e["type"], e["team"], e["player"], e.get("offensive", ""), e.get("dur", "")])
    summary = {
        "game": game["meta"]["title"], "date": game["meta"]["date"],
        "teams": {t["name"]: {k: st["teams"][t["id"]].get(k) for k in
                              ("pts", "overlay_pts", "fgm", "fga", "fg_pct", "fg3m", "fg3a", "ftm", "reb", "ast", "tov",
                               "assisted_pct", "oreb_pct")} | {"insights": st["teams"][t["id"]].get("insights")}
                  for t in game["teams"]},
        "players": [{"team": tname[p["team"]], "number": p["number"], "name": p["name"],
                     **{k: st["players"][p["id"]].get(k) for k in
                        ("pts", "fgm", "fga", "fg3m", "fg3a", "reb", "ast", "tov", "tracked_min", "avg_touch_s",
                         "dist_per_min", "spacing_ft")},
                     "zones": {z: {k: v for k, v in d.items() if k in ("fga", "fgm", "pct")}
                               for z, d in st["players"][p["id"]]["zones"].items()},
                     "insights": st["players"][p["id"]].get("insights")}
                    for p in game["players"] if st["players"][p["id"]]["fga"] or st["players"][p["id"]]["touches"]],
        "notes": "Estimated from video by hoopstats. Ask an LLM: 'Write a short development plan for each player.'",
    }
    (run_dir / "coach_summary.json").write_text(json.dumps(summary, indent=1, default=str))

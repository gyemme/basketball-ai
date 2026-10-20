"""hoopstats command line.

    hoopstats analyze VIDEO          full pipeline -> runs/<name>/report.html
    hoopstats calibrate VIDEO        click court landmarks (once per gym / camera spot)
    hoopstats report RUN_DIR         re-build stats + report from cache (after editing config/overrides)
    hoopstats score VIDEO            read the ScoreSync score timeline only (fast)
    hoopstats probe VIDEO            show video info
    hoopstats annotate RUN_DIR       render an annotated check video
    hoopstats sample                 sample report with made-up data
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import webbrowser
from pathlib import Path

from . import config as C


def tsec(s: str | None) -> float | None:
    if s is None:
        return None
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    parts = [float(p) for p in s.split(":")]
    v = 0.0
    for p in parts:
        v = v * 60 + p
    return v


def slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-") or "game"


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- commands

def cmd_probe(a):
    from . import video
    for v in video.collect(a.video):
        print(json.dumps(v.to_dict(), indent=1))


def cmd_score(a):
    from . import video, scoreboard
    for v in video.collect(a.video):
        log(f"Reading the score overlay in {v.path.name} ...")
        tl, raw = scoreboard.scan(v, progress=None)
        if not raw:
            log("  overlay templates missing")
            return
        good = sum(1 for r in raw if r[1] is not None and r[2] is not None)
        log(f"  {good}/{len(raw)} readings")
        for e in tl:
            m, s = divmod(int(e["t"]), 60)
            log(f"  {m:3d}:{s:02d}  {e['left']:3d} - {e['right']:<3d}{'   (jump: recording paused or scorer catching up)' if e.get('jump') else ''}")


def _run_dir(a, videos) -> Path:
    if a.out:
        return Path(a.out)
    return Path("runs") / slug(videos[0].path.stem)


def _config_path(a, run_dir: Path, videos) -> Path:
    if a.config:
        return Path(a.config)
    p = run_dir / "game.yaml"
    if not p.exists():
        run_dir.mkdir(parents=True, exist_ok=True)
        v0 = videos[0]
        C.write_template(p, date=(v0.creation_time or "")[:10])
        log(f"  created {p}  (edit team names, jersey colours and rosters there)")
    return p


def _calib_path(a, run_dir: Path) -> Path | None:
    if getattr(a, "calib", None):
        return Path(a.calib)
    p = run_dir / "calib" / "calib.json"
    return p


def cmd_calibrate(a):
    from . import video
    from .calibrate import run
    from .court import Court
    videos = video.collect(a.video)
    run_dir = _run_dir(a, videos)
    cfg = C.load(_config_path(a, run_dir, videos))
    out = Path(a.calib).parent if a.calib else run_dir / "calib"
    run([v.path for v in videos], out, Court.from_config(cfg["court"]), width=int(cfg["analysis"]["width"]),
        start=tsec(a.start) or 0.0, end=tsec(a.end), open_browser=not a.no_browser, log=log)


def cmd_analyze(a):
    from . import video
    from .register import load_calib
    from . import extract, analyze
    t0 = time.time()
    videos = video.collect(a.video)
    run_dir = _run_dir(a, videos)
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"hoopstats: {', '.join(v.path.name for v in videos)} -> {run_dir}/")
    for v in videos:
        W, H = v.size
        log(f"  {v.path.name}: {W}x{H} {v.fps:.0f} fps {v.codec} {v.duration / 60:.1f} min"
            + ("  (variable frame rate)" if v.vfr else ""))
    cfg_path = _config_path(a, run_dir, videos)
    cfg = C.load(cfg_path)
    if a.fps:
        cfg["analysis"]["fps"] = a.fps
    if a.detector:
        cfg["analysis"]["detector"] = a.detector
    if a.model:
        cfg["analysis"]["model"] = a.model

    start, end = tsec(a.start) or 0.0, tsec(a.end)
    calib_p = _calib_path(a, run_dir)
    calib = load_calib(calib_p)
    if calib is None and not a.no_calibrate and sys.stdin.isatty():
        ans = input("No court calibration yet. Calibrate now in your browser? It takes a few minutes and "
                    "enables shot locations. [Y/n] ").strip().lower()
        if ans in ("", "y", "yes"):
            from .calibrate import run as calrun
            from .court import Court
            calrun([v.path for v in videos], calib_p.parent, Court.from_config(cfg["court"]),
                   width=int(cfg["analysis"]["width"]), start=start, end=end, log=log)
            calib = load_calib(calib_p)
    if calib is None:
        log("  ! no court calibration: stats and makes will work, shot locations will be approximate")

    cache = run_dir / "cache"
    meta_p = cache / "meta.json"
    reuse = False
    if meta_p.exists() and not a.force:
        m = json.loads(meta_p.read_text())
        reuse = (m.get("version") == extract.CACHE_VERSION and m.get("fps") == float(cfg["analysis"]["fps"])
                 and m.get("start") == start and m.get("end") == end
                 and m.get("calib_hash") == extract.calib_hash(calib))
    if reuse:
        log("Step 1/2: using cached detections (add --force to redo)")
    else:
        log("Step 1/2: detecting and tracking players, ball and court (the slow part) ...")
        extract.run(videos, cfg, cache, calib, start=start, end=end, log=log)
    log("Step 2/2: events, stats and report ...")
    overrides = run_dir / "overrides.yaml"
    game = analyze.run(run_dir, cfg, overrides if overrides.exists() else None, log=log)
    _summary(game, run_dir, time.time() - t0)
    if a.open:
        webbrowser.open((run_dir / "report.html").resolve().as_uri())


def cmd_report(a):
    from . import analyze
    run_dir = Path(a.run_dir)
    cfg = C.load(a.config or run_dir / "game.yaml")
    ov = Path(a.overrides) if a.overrides else run_dir / "overrides.yaml"
    game = analyze.run(run_dir, cfg, ov if ov.exists() else None, log=log)
    _summary(game, run_dir, None)
    if a.open:
        webbrowser.open((run_dir / "report.html").resolve().as_uri())


def cmd_annotate(a):
    from .annotate import run
    run(Path(a.run_dir), tsec(a.start) or 0.0, tsec(a.end), width=a.width, log=log)


def cmd_sample(a):
    from .report.build import build
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from make_sample_report import make_game  # type: ignore
    from . import stats, insights
    g = make_game()
    stats.compute(g)
    insights.attach(g)
    out = build(g, a.out)
    log(f"wrote {out}")


def _summary(game: dict, run_dir: Path, secs: float | None):
    st = game["stats"]
    log("")
    for t in game["teams"]:
        s = st["teams"][t["id"]]
        sb = f" (scoreboard {s['overlay_pts']})" if s.get("overlay_pts") is not None else ""
        log(f"  {t['name']:>20}: {s['pts']} pts{sb} | FG {s['fgm']}/{s['fga']} | 3PT {s['fg3m']}/{s['fg3a']} | "
            f"FT {s['ftm']} | REB {s['reb']} | AST {s['ast']} | TOV {s['tov']}")
    q = game.get("quality", {})
    log(f"  court registered: {q.get('registration_pct')}% of frames | players identified: {q.get('identified_pct')}%")
    log("")
    log(f"Report:  {run_dir / 'report.html'}")
    log(f"Data:    {run_dir / 'player_stats.csv'}, shots.csv, events.csv, coach_summary.json")
    if game.get("unidentified"):
        log(f"Fix unknown players: add lines like 'track_{game['unidentified'][0]['track'].split('_')[1]}: A10' to "
            f"{run_dir / 'overrides.yaml'} then run:  hoopstats report {run_dir}")
    if secs:
        log(f"Done in {secs / 60:.1f} min")


def main(argv=None):
    p = argparse.ArgumentParser(prog="hoopstats", description="Basketball stats from XbotGo game video")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("analyze", help="full pipeline: video -> report.html")
    s.add_argument("video", nargs="+", help="video file(s) or a folder of split files")
    s.add_argument("--config", help="game.yaml (default: <out>/game.yaml, created if missing)")
    s.add_argument("--out", help="output folder (default: runs/<video name>)")
    s.add_argument("--calib", help="calibration json (default: <out>/calib/calib.json)")
    s.add_argument("--start", help="start time, e.g. 5:00")
    s.add_argument("--end", help="end time, e.g. 12:30")
    s.add_argument("--fps", type=float, help="frames analysed per second (default 10)")
    s.add_argument("--detector", choices=["yolo", "rfdetr"])
    s.add_argument("--model", help="detector weights, e.g. yolo11l.pt")
    s.add_argument("--force", action="store_true", help="ignore cached detections")
    s.add_argument("--no-calibrate", action="store_true", help="don't ask to calibrate")
    s.add_argument("--open", action="store_true", help="open the report when done")
    s.set_defaults(fn=cmd_analyze)

    s = sub.add_parser("calibrate", help="mark court landmarks (once per gym / camera spot)")
    s.add_argument("video", nargs="+")
    s.add_argument("--out", help="run folder (default: runs/<video name>)")
    s.add_argument("--config")
    s.add_argument("--calib", help="where to write calib.json")
    s.add_argument("--start", help="only sample views from this time on, e.g. 5:00")
    s.add_argument("--end", help="only sample views up to this time")
    s.add_argument("--no-browser", action="store_true", help="print the URL instead of opening a browser")
    s.set_defaults(fn=cmd_calibrate)

    s = sub.add_parser("report", help="rebuild stats/report from cached detections")
    s.add_argument("run_dir")
    s.add_argument("--config")
    s.add_argument("--overrides")
    s.add_argument("--open", action="store_true")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("score", help="read the ScoreSync score overlay timeline")
    s.add_argument("video", nargs="+")
    s.set_defaults(fn=cmd_score)

    s = sub.add_parser("probe", help="show video info")
    s.add_argument("video", nargs="+")
    s.set_defaults(fn=cmd_probe)

    s = sub.add_parser("annotate", help="render an annotated video from a run (for checking)")
    s.add_argument("run_dir")
    s.add_argument("--start")
    s.add_argument("--end")
    s.add_argument("--width", type=int, default=1280)
    s.set_defaults(fn=cmd_annotate)

    s = sub.add_parser("sample", help="write a sample report with made-up data")
    s.add_argument("--out", default="runs/sample/report.html")
    s.set_defaults(fn=cmd_sample)

    a = p.parse_args(argv)
    if a.cmd in ("analyze", "calibrate", "score", "probe", "annotate") and not shutil.which("ffmpeg"):
        sys.exit("ffmpeg is required:  brew install ffmpeg")
    a.fn(a)


if __name__ == "__main__":
    main()

"""Render an annotated check video: player boxes + IDs, ball, rims, shot banners."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import video as vid
from .camera import Camera
from .court import Court
from .extract import load


def _bgr(hexc: str):
    h = hexc.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


def run(run_dir: Path, start: float = 0.0, end: float | None = None, width: int = 1280, log=print) -> Path:
    ex, meta = load(run_dir / "cache")
    game = json.loads((run_dir / "game.json").read_text())
    court = Court.from_config(game["court"])
    fps = meta["fps"]
    aw, ah = meta["width"], meta["height"]
    t = ex["t"]
    P = ex["people"]
    tmap = game.get("track_map", {})
    colors = {tm["id"]: _bgr(tm["mark"]) for tm in game["teams"]}
    pname = {p["id"]: f"#{p['number']}" for p in game["players"]}
    shots = game["shots"]
    by_f = {}
    for i, r in enumerate(P):
        by_f.setdefault(int(r[0]), []).append(r)
    ball = {int(r[0]): r for r in ex["balls"]}
    s = width / aw
    out = run_dir / "annotated.mp4"
    H_out = int(round(ah * s / 2) * 2)
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{width}x{H_out}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                             "-pix_fmt", "yuv420p", "-crf", "23", str(out)], stdin=subprocess.PIPE)
    end = end if end is not None else float(t[-1])
    infos = [vid.VideoInfo(**{**v, "path": Path(v["path"])}) for v in meta["videos"]]
    fi0 = int(np.searchsorted(t, start))
    n = 0
    for v in infos:
        a, b = max(start - v.offset, 0), min(end - v.offset, v.duration)
        if b <= 0 or a >= v.duration:
            continue
        for tt, frame in vid.frames(v, fps=fps, width=width, start=a, end=b):
            gt = v.offset + tt
            fi = int(np.argmin(np.abs(t - gt)))
            img = frame.copy()
            for r in by_f.get(fi, []):
                tid = str(int(r[1]))
                info = tmap.get(tid, {})
                col = colors.get(info.get("team"), (160, 160, 160))
                x1, y1, x2, y2 = (r[2:6] * s).astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
                lab = pname.get(info.get("player"), "?") if info.get("team") else ""
                if lab:
                    cv2.putText(img, lab, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            if fi in ball:
                _, bx, by, br, _ = ball[fi]
                cv2.circle(img, (int(bx * s), int(by * s)), max(4, int(br * s)), (0, 165, 255), 2)
            Hm = ex["H"][fi]
            if not np.isnan(Hm[0, 0]):
                try:
                    cam = Camera(Hm, aw, ah)
                    for hp in court.hoops:
                        rr = cam.rim(hp)
                        if rr:
                            cv2.circle(img, (int(rr[0] * s), int(rr[1] * s)), max(3, int(rr[2] * s)), (255, 0, 255), 2)
                    # court outline
                    pts = np.float32([[0, 0], [court.length, 0], [court.length, court.width], [0, court.width]])
                    q = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), Hm).reshape(-1, 2) * s
                    cv2.polylines(img, [np.int32(q)], True, (255, 255, 0), 1)
                except Exception:
                    pass
            for sh in shots:
                if 0 <= gt - sh["t"] <= 2.0:
                    txt = ("MAKE " if sh["made"] else "MISS ") + pname.get(sh["player"], sh["player"])
                    cv2.rectangle(img, (10, 10), (260, 44), (0, 0, 0), -1)
                    cv2.putText(img, txt, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (80, 220, 80) if sh["made"] else (80, 80, 240), 2)
            m, sec = divmod(int(gt), 60)
            cv2.putText(img, f"{m}:{sec:02d}", (width - 90, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            proc.stdin.write(img.tobytes())
            n += 1
    proc.stdin.close()
    proc.wait()
    log(f"wrote {out} ({n} frames)")
    return out

"""Stage 1: one pass over the video -> detections, tracks, ball, court registration, score overlay.

Everything expensive happens here and is cached in <run>/cache/, so the analysis stage
(events, stats, report) can be re-run in seconds.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import video as vid
from .detect import make_detector
from .register import Registrar, image_to_court
from .scoreboard import ScoreReader
from .track import Tracker, BallTracker

CACHE_VERSION = 4


def torso_feature(frame: np.ndarray, box) -> np.ndarray:
    """[L, a, b, stripe, sat] of the shirt area. stripe = % of pixels in the smaller of the dark/light groups."""
    x1, y1, x2, y2 = [int(v) for v in box[:4]]
    w, h = x2 - x1, y2 - y1
    cx1, cx2 = x1 + int(0.28 * w), x2 - int(0.28 * w)
    cy1, cy2 = y1 + int(0.18 * h), y1 + int(0.45 * h)
    crop = frame[max(cy1, 0):max(cy2, 1), max(cx1, 0):max(cx2, 1)]
    if crop.size < 12:
        return np.full(5, np.nan, np.float32)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    L = lab[:, 0]
    # referee shirts: lots of both dark and light pixels (black/white stripes)
    stripe = 100.0 * float(min(np.mean(L < 90), np.mean(L > 160)))
    med = np.median(lab, axis=0)
    return np.array([med[0], med[1], med[2], stripe, float(np.median(hsv[:, 1]))], np.float32)


def run(videos: list[vid.VideoInfo], cfg: dict, cache: Path, calib: dict | None,
        start: float = 0.0, end: float | None = None, log=print) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "crops").mkdir(exist_ok=True)
    (cache / "thumbs").mkdir(exist_ok=True)
    a = cfg["analysis"]
    fps = float(a["fps"])
    aw = int(a["width"])
    log(f"  loading the {cfg['analysis'].get('detector', 'yolo')} detector (first run downloads weights) ...")
    det = make_detector(cfg)
    log("  detector ready; reading video ...")
    reader = ScoreReader((cfg.get("overlay") or {}).get("layout")) if cfg.get("overlay", {}).get("enabled", True) else None
    ocr_every = max(1, int(round(float(a.get("ocr_every", 1.0)) * fps)))

    T, Hs, Hconf = [], [], []
    people, colors, balls, overlay = [], [], [], []
    crops_n: dict[int, int] = {}
    thumb_h: dict[int, float] = {}
    tid_offset = 0
    t0 = time.time()
    fi = 0
    for v in videos:
        W, H = v.size
        ah = int(round(H * aw / W / 2) * 2)
        reg = Registrar(calib, (aw, ah))
        if calib and not reg.calibrated:
            log("  ! calibration keyframes could not be loaded; court locations disabled")
        trk = Tracker(max_miss=int(2 * fps))
        btr = BallTracker(max_gap=int(0.8 * fps))
        s_start = max(0.0, start - v.offset)
        s_end = None if end is None else end - v.offset
        if s_end is not None and s_end <= 0:
            continue
        if s_start >= v.duration:
            continue
        scale = W / aw
        last_print = 0.0
        max_tid = 0
        for t, full in vid.frames(v, fps=fps, start=s_start, end=s_end):
            gt = v.offset + t
            frame = cv2.resize(full, (aw, ah), interpolation=cv2.INTER_AREA) if full.shape[1] != aw else full
            d = det(frame)
            Hc, conf, M = reg.update(frame, d.persons)
            tracks = trk.update(d.persons, M)
            ball = btr.update(d.balls, aw, M, d.persons)
            T.append(gt)
            Hs.append(Hc if Hc is not None else np.full((3, 3), np.nan))
            Hconf.append(conf)
            if tracks:
                boxes = np.stack([b for _, b, _ in tracks])
                feet = np.c_[(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]]
                court = image_to_court(Hc, feet) if Hc is not None else np.full((len(tracks), 2), np.nan)
                for k, (tid, b, c) in enumerate(tracks):
                    gid = tid + tid_offset
                    max_tid = max(max_tid, gid)
                    people.append([fi, gid, *b[:4], c, court[k, 0], court[k, 1]])
                    colors.append(torso_feature(frame, b))
                    bh = b[3] - b[1]
                    # jersey crops for number reading (from the full-resolution frame)
                    if fi % ocr_every == (gid % ocr_every) and bh > 40 and crops_n.get(gid, 0) < 60:
                        x1, y1, x2, y2 = (b[:4] * scale).astype(int)
                        hh = y2 - y1
                        cr = full[max(0, y1 + int(0.12 * hh)):max(0, y1 + int(0.62 * hh)), max(0, x1):max(0, x2)]
                        if cr.size:
                            cdir = cache / "crops" / str(gid)
                            cdir.mkdir(exist_ok=True)
                            cv2.imwrite(str(cdir / f"{fi:06d}.jpg"), cr, [cv2.IMWRITE_JPEG_QUALITY, 92])
                            crops_n[gid] = crops_n.get(gid, 0) + 1
                    if bh > thumb_h.get(gid, 0) and b[1] > 0:
                        x1, y1, x2, y2 = (b[:4] * scale).astype(int)
                        th = full[max(0, y1):y2, max(0, x1):x2]
                        if th.size:
                            th = cv2.resize(th, (112, int(112 * th.shape[0] / max(th.shape[1], 1))))
                            cv2.imwrite(str(cache / "thumbs" / f"{gid}.jpg"), th, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            thumb_h[gid] = bh
            if ball is not None:
                balls.append([fi, *ball])
            if reader is not None and reader.ready and (fi % max(1, int(fps)) == 0):
                r = reader.read(full)
                overlay.append([gt, -1 if r.left is None else r.left, -1 if r.right is None else r.right, r.conf])
            fi += 1
            if time.time() - last_print >= 15:
                last_print = time.time()
                el = time.time() - t0
                done = gt - start
                total = ((end or sum(x.duration for x in videos)) - start)
                eta = f", ~{el / done * (total - done) / 60:.0f} min left" if done >= 5 else ""
                log(f"  {vid_time(gt)} processed | {len(tracks)} people in view | "
                    f"court {'ok' if Hc is not None else '--'} | {el / 60:.1f} min elapsed{eta}")
        tid_offset = max_tid + 1
    np.savez_compressed(
        cache / "extract.npz",
        t=np.array(T, np.float64), H=np.array(Hs, np.float64).reshape(-1, 3, 3), Hconf=np.array(Hconf, np.float32),
        people=np.array(people, np.float64).reshape(-1, 9), colors=np.array(colors, np.float32).reshape(-1, 5),
        balls=np.array(balls, np.float64).reshape(-1, 5), overlay=np.array(overlay, np.float64).reshape(-1, 4),
    )
    meta = {
        "version": CACHE_VERSION, "fps": fps, "width": aw,
        "height": int(round(videos[0].size[1] * aw / videos[0].size[0] / 2) * 2),
        "start": start, "end": end, "videos": [v.to_dict() for v in videos],
        "calibrated": bool(calib), "calib_hash": calib_hash(calib), "seconds": round(time.time() - t0, 1),
    }
    (cache / "meta.json").write_text(json.dumps(meta, indent=1))
    return cache


def calib_hash(calib: dict | None) -> str | None:
    if not calib:
        return None
    import hashlib
    keep = [[kf.get("t"), kf.get("H")] for kf in calib.get("keyframes", [])]
    return hashlib.md5(json.dumps(keep).encode()).hexdigest()


def vid_time(s: float) -> str:
    s = int(s)
    return f"{s // 60}:{s % 60:02d}"


def load(cache: Path) -> tuple[dict, dict]:
    d = np.load(cache / "extract.npz")
    meta = json.loads((cache / "meta.json").read_text())
    return {k: d[k] for k in d.files}, meta

"""Synthetic game generator for testing everything after detection (no ML models needed).

Simulates a real camera with a known projection, 10 players + a referee, passes, shots,
makes/misses, rebounds and a lagging ScoreSync overlay, then writes the same cache files
that extract.py writes. analyze.run() can then be checked against the known truth.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

import cv2
import numpy as np

from hoopstats.court import Court
from hoopstats.extract import CACHE_VERSION

W, H = 1920, 1080
FPS = 10.0


def make_camera(target_x: float = 42.0, f: float = 1250.0):
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]])
    C = np.array([42.0, -18.0, 16.0])
    target = np.array([target_x, 25.0, 2.0])
    z = target - C
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    t = -R @ C
    P = K @ np.c_[R, t]
    return P


def proj(P, pts):
    X = np.c_[np.asarray(pts, float).reshape(-1, 3), np.ones(len(np.asarray(pts).reshape(-1, 3)))]
    q = X @ P.T
    return q[:, :2] / q[:, 2:3]


TEAM_LAB = {"A": (200, 115, 108), "B": (55, 132, 112)}   # light blue, navy (OpenCV Lab, 0-255)


def generate(out_dir: Path, seconds: float = 240.0, seed: int = 3, pan: bool = True) -> dict:
    rng = random.Random(seed)
    court = Court()
    cache = out_dir / "cache"
    (cache / "crops").mkdir(parents=True, exist_ok=True)
    (cache / "thumbs").mkdir(exist_ok=True)
    N = int(seconds * FPS)
    t = np.arange(N) / FPS
    numbers = {"A": ["3", "5", "10", "14", "22"], "B": ["1", "4", "7", "12", "20"]}
    pids = [(tm, n) for tm in "AB" for n in numbers[tm]]
    tid_of = {p: i + 1 for i, p in enumerate(pids)}
    ref_tid = 99
    pos = {p: np.array([rng.uniform(10, 74), rng.uniform(5, 45)]) for p in pids}
    pos["ref"] = np.array([42.0, 3.0])

    # ---- script the game: possessions alternate; each ends in a shot (made or missed)
    timeline = []   # list of (t0, kind, data)
    truth = {"shots": [], "score": {"A": 0, "B": 0}}
    ball3d = np.full((N, 3), np.nan)
    holder = [None] * N
    pos_at = [None] * N

    def snap(fr):
        if 0 <= fr < N:
            pos_at[fr] = {k: v.copy() for k, v in pos.items()}
    team = "A"
    f = int(2 * FPS)
    attack = {"A": 1, "B": 0}
    while f < N - int(8 * FPS):
        if f > N / 2 and attack["A"] == 1:
            attack = {"A": 0, "B": 1}       # halftime switch
        hx, hy = court.hoops[attack[team]]
        sgn = -1 if attack[team] == 1 else 1
        off = [p for p in pids if p[0] == team]
        # set offense positions around the attacked hoop
        for k, p in enumerate(off):
            ang = math.radians(-70 + 35 * k)
            r = rng.uniform(8, 21)
            pos[p] = np.array([hx + sgn * r * math.cos(ang), hy + r * math.sin(ang)])
        for k, p in enumerate([p for p in pids if p[0] != team]):
            q = pos[off[k]]
            pos[p] = q + 0.35 * (np.array([hx, hy]) - q)
        # 2-3 passes then a shot
        passes = rng.randint(1, 3)
        cur = rng.choice(off)
        for _ in range(passes):
            hold = int(rng.uniform(1.0, 2.2) * FPS)
            for k in range(hold):
                if f >= N:
                    break
                holder[f] = cur
                ball3d[f] = [*pos[cur], 3.5]
                snap(f)
                snap(f)
            f += 1
            nxt = rng.choice([p for p in off if p != cur])
            fl = int(0.6 * FPS)
            a, b = np.array([*pos[cur], 3.5]), np.array([*pos[nxt], 3.5])
            for k in range(fl):
                w = (k + 1) / (fl + 1)
                ball3d[f] = a + w * (b - a) + np.array([0, 0, 2 * math.sin(math.pi * w)])
                snap(f)
                snap(f)
            f += 1
            cur = nxt
        hold = int(rng.uniform(0.5, 1.0) * FPS)
        for k in range(hold):
            holder[f] = cur
            ball3d[f] = [*pos[cur], 3.5]
            snap(f)
            f += 1
        # shot arc to the rim
        made = rng.random() < 0.5
        d = float(np.linalg.norm(pos[cur] - np.array([hx, hy])))
        value = 3 if court.is_three(*pos[cur], attack[team]) else 2
        truth["shots"].append({"t": float(t[f - 1]), "team": team, "player": cur, "made": made, "value": value,
                               "xy": pos[cur].tolist()})
        a = np.array([*pos[cur], 6.5])
        b = np.array([hx, hy, 10.0])
        fl = int(1.3 * FPS)
        for k in range(fl):
            w = (k + 1) / fl
            p3 = a + w * (b - a)
            p3[2] += 6 * math.sin(math.pi * w)
            ball3d[f] = p3
            snap(f)
            f += 1
        if made:
            truth["score"][team] += value
            for k in range(int(0.6 * FPS)):
                ball3d[f] = [hx, hy, 10 - 6 * (k + 1) / (0.6 * FPS)]
                snap(f)
                snap(f)
            f += 1
            lag = rng.uniform(3, 9)
            timeline.append((float(t[f]) + lag, dict(truth["score"])))
            team = "B" if team == "A" else "A"
            for _ in range(int(rng.uniform(7, 12) * FPS)):   # inbound + bring it up
                snap(f)
                f += 1
        else:
            # bounce out and a rebound
            other = "B" if team == "A" else "A"
            reb = rng.choice([p for p in pids if p[0] == (team if rng.random() < 0.3 else other)])
            pos[reb] = np.array([hx + sgn * rng.uniform(3, 9), hy + rng.uniform(-8, 8)])
            a = np.array([hx, hy, 10.5])
            b = np.array([*pos[reb], 4.0])
            fl = int(0.9 * FPS)
            for k in range(fl):
                w = (k + 1) / fl
                ball3d[f] = a + w * (b - a) + np.array([0, 0, 3 * math.sin(math.pi * w)])
                snap(f)
                snap(f)
            f += 1
            truth["shots"][-1]["rebound"] = reb
            team = reb[0]
            for k in range(int(1.0 * FPS)):
                holder[f] = reb
                ball3d[f] = [*pos[reb], 3.5]
                snap(f)
                snap(f)
            f += 1

    # ---- render detections through a panning camera
    people, colors, balls, Hs, Hc = [], [], [], [], []
    overlay = []
    jitter = np.random.default_rng(seed)
    cam_x = 42.0
    for fi in range(N):
        if pan and not np.isnan(ball3d[fi, 0]):   # XbotGo-style: follow the ball, smoothly
            cam_x += 0.08 * (float(np.clip(ball3d[fi, 0], 18, 66)) - cam_x)
        P = make_camera(cam_x, 1250.0 + (150 if pan and abs(cam_x - 42) > 15 else 0))
        Hm = P[:, [0, 1, 3]] / P[2, 3]
        Hs.append(Hm)
        Hc.append(0.9)
        cur_pos = pos_at[fi] or pos_at[max(0, fi - 1)] or pos
        if pos_at[fi] is None:
            pos_at[fi] = cur_pos
        ents = [(p, cur_pos[p]) for p in pids] + [("ref", cur_pos["ref"])]
        for p, xy in ents:
            xy = xy + jitter.normal(0, 0.15, 2)
            foot = proj(P, [[*xy, 0]])[0]
            head = proj(P, [[*xy, 5.2]])[0]
            h = foot[1] - head[1]
            w = 0.38 * h
            if not (0 < foot[0] < W and 0 < foot[1] < H * 0.98):
                continue
            box = [foot[0] - w / 2, head[1], foot[0] + w / 2, foot[1]]
            tid = ref_tid if p == "ref" else tid_of[p]
            people.append([fi, tid, *box, 0.9, xy[0], xy[1]])
            if p == "ref":
                colors.append([120, 128, 128, 40, 5])
            else:
                L, A, B = TEAM_LAB[p[0]]
                colors.append([L + jitter.normal(0, 4), A + jitter.normal(0, 2), B + jitter.normal(0, 2), 5, 60])
        if not np.isnan(ball3d[fi, 0]) and jitter.random() > 0.1:
            u, v = proj(P, [ball3d[fi]])[0]
            if 0 < u < W and 0 < v < H:
                balls.append([fi, u, v, 7.0, 0.6])
        if fi % int(FPS) == 0:
            sc = {"A": 0, "B": 0}
            for tt, s in timeline:
                if tt <= t[fi]:
                    sc = s
            overlay.append([t[fi], sc["A"], sc["B"], 0.99])
    np.savez_compressed(cache / "extract.npz", t=t, H=np.array(Hs), Hconf=np.array(Hc, np.float32),
                        people=np.array(people), colors=np.array(colors, np.float32),
                        balls=np.array(balls), overlay=np.array(overlay))
    meta = {"version": CACHE_VERSION, "fps": FPS, "width": W, "height": H, "start": 0.0, "end": None,
            "videos": [{"path": str(out_dir / "synthetic.mp4"), "width": 3840, "height": 2160, "fps": 30,
                        "duration": seconds, "codec": "h264", "rotation": 0, "vfr": False,
                        "creation_time": "2026-10-17T18:00:21-07:00", "device": None, "offset": 0.0}],
            "calibrated": True, "seconds": 0}
    (cache / "meta.json").write_text(json.dumps(meta))
    # fake OCR results + crops so jersey numbers resolve
    store = {}
    for (tm, n), tid in tid_of.items():
        d = cache / "crops" / str(tid)
        d.mkdir(exist_ok=True)
        for k in range(3):
            name = f"{k:06d}.jpg"
            cv2.imwrite(str(d / name), np.full((40, 30, 3), 128, np.uint8))
            store[f"{tid}/{name}"] = [[n, 0.95]] if not (tm == "B" and n == "20") else []
    (cache / "ocr.json").write_text(json.dumps(store))
    truth["score_final"] = timeline[-1][1] if timeline else {"A": 0, "B": 0}
    return truth

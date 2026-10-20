"""Who is who: on-court filtering, referees, team colours, jersey numbers, track -> player."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .court import Court
from .ocr import make_ocr, prep


def hex_to_lab(h: str) -> np.ndarray:
    h = h.lstrip("#")
    rgb = np.array([[[int(h[i:i + 2], 16) for i in (4, 2, 0)]]], np.uint8)  # BGR
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float32)


def lab_to_hex(lab) -> str:
    px = np.clip(np.array([[lab]], np.float32), 0, 255).astype(np.uint8)
    b, g, r = cv2.cvtColor(px, cv2.COLOR_LAB2BGR).reshape(3)
    return f"#{r:02x}{g:02x}{b:02x}"


def track_table(ex: dict, court: Court, fps: float) -> dict[int, dict]:
    P, C = ex["people"], ex["colors"]
    out: dict[int, dict] = {}
    if len(P) == 0:
        return out
    for tid in np.unique(P[:, 1]).astype(int):
        m = P[:, 1] == tid
        rows, cols = P[m], C[m]
        xy = rows[:, 7:9]
        reg = ~np.isnan(xy[:, 0])
        on = np.array([court.contains(x, y, 2.5) for x, y in xy[reg]]) if reg.any() else np.zeros(0, bool)
        good = ~np.isnan(cols[:, 0])
        out[tid] = {
            "tid": tid, "n": int(m.sum()), "seconds": float(m.sum() / fps),
            "t0": int(rows[0, 0]), "t1": int(rows[-1, 0]),
            "color": np.nanmedian(cols[good, :3], axis=0) if good.any() else np.full(3, np.nan),
            "stripe": float(np.nanmedian(cols[good, 3])) if good.any() else 0.0,
            "sat": float(np.nanmedian(cols[good, 4])) if good.any() else 0.0,
            "on_court": float(on.mean()) if len(on) else None,
            "box_h": float(np.median(rows[:, 5] - rows[:, 3])),
        }
    return out


def is_referee(t: dict) -> bool:
    lab = t["color"]
    if np.any(np.isnan(lab)):
        return False
    chroma = float(np.hypot(lab[1] - 128, lab[2] - 128))
    return t["stripe"] > 25 and chroma < 15


def kmeans2(X: np.ndarray, w: np.ndarray, iters=30) -> tuple[np.ndarray, np.ndarray]:
    # init with the two most different points (by lightness), weighted Lloyd iterations
    i0 = int(np.argmin(X[:, 0]))
    i1 = int(np.argmax(np.linalg.norm(X - X[i0], axis=1)))
    C = np.stack([X[i0], X[i1]])
    for _ in range(iters):
        a = np.argmin(np.linalg.norm(X[:, None] - C[None], axis=2), axis=1)
        C2 = np.stack([np.average(X[a == k], axis=0, weights=w[a == k]) if (a == k).any() else C[k] for k in (0, 1)])
        if np.allclose(C, C2):
            break
        C = C2
    return C, a


def assign_teams(tracks: dict[int, dict], cfg: dict, min_seconds: float = 2.0) -> tuple[dict[int, int], list[np.ndarray]]:
    """Returns {tid: cluster 0/1 (or -1 = ref/other)} and the two cluster colours (Lab)."""
    cand = [t for t in tracks.values()
            if t["seconds"] >= min_seconds and not np.any(np.isnan(t["color"]))
            and (t["on_court"] is None or t["on_court"] >= 0.5)]
    refs = {t["tid"] for t in cand if is_referee(t)}
    players = [t for t in cand if t["tid"] not in refs]
    jerseys = [tm.get("jersey") for tm in cfg["teams"]]
    if all(jerseys):
        C = np.stack([hex_to_lab(j) for j in jerseys])
    elif len(players) >= 2:
        X = np.stack([t["color"] for t in players])
        w = np.array([t["seconds"] for t in players])
        C, _ = kmeans2(X, w)
    else:
        return {t["tid"]: -1 for t in tracks.values()}, []
    out = {}
    for t in tracks.values():
        if t["tid"] in refs or np.any(np.isnan(t["color"])):
            out[t["tid"]] = -1
            continue
        if t["on_court"] is not None and t["on_court"] < 0.5:
            out[t["tid"]] = -1
            continue
        d = np.linalg.norm(C - t["color"], axis=1)
        k = int(np.argmin(d))
        # far from both team colours -> spectator / coach / other
        out[t["tid"]] = k if d[k] < 45 else -1
    return out, [c for c in C]


def read_numbers(cache: Path, tids: list[int], kind: str, max_per_track: int = 20, log=print) -> dict[int, Counter]:
    """OCR jersey crops; results cached in cache/ocr.json."""
    store_p = cache / "ocr.json"
    store = json.loads(store_p.read_text()) if store_p.exists() else {}
    ocr = None
    votes: dict[int, Counter] = {}
    n_new = 0
    for i, tid in enumerate(tids):
        d = cache / "crops" / str(tid)
        files = sorted(d.glob("*.jpg")) if d.exists() else []
        if not files:
            continue
        # prefer the biggest crops
        files = sorted(files, key=lambda f: -f.stat().st_size)[:max_per_track]
        c = Counter()
        for f in files:
            key = f"{tid}/{f.name}"
            if key not in store:
                if ocr is None:
                    ocr = make_ocr(kind)
                    log(f"  reading jersey numbers with: {ocr.name}")
                    if ocr.name == "none":
                        break
                img = cv2.imread(str(f))
                store[key] = ocr(prep(img)) if img is not None else []
                n_new += 1
            for text, conf in store[key]:
                c[text] += conf
            if c and c.most_common(1)[0][1] >= 2.5:   # ~3 confident agreeing reads is enough
                break
        votes[tid] = c
    if n_new:
        store_p.write_text(json.dumps(store))
    return votes


def best_number(c: Counter, allowed: set[str] | None) -> tuple[str | None, float]:
    if not c:
        return None, 0.0
    items = [(k, v) for k, v in c.items() if not allowed or k in allowed]
    if not items:
        return None, 0.0
    items.sort(key=lambda kv: -kv[1])
    k, v = items[0]
    total = sum(c.values())
    share = v / total if total else 0
    if v >= 0.9 and share >= 0.5:
        return k, share
    return None, share


def load_overrides(path: Path | None) -> dict:
    if not path or not Path(path).exists():
        return {}
    import yaml
    d = yaml.safe_load(Path(path).read_text()) or {}
    return {int(str(k).replace("track_", "")): str(v) for k, v in d.items()}

"""Read the XbotGo ScoreSync score overlay that is burned into the video.

The overlay sits at a fixed spot in the frame and uses a fixed font, so digits are read
with template matching (no OCR model needed). Templates live in assets/scoresync_digits.npz.

Layout (normalised to frame size, measured on a 3840x2160 XbotGo Falcon recording):
    left score  : dark digits on white   (team shown on the left of the overlay)
    right score : white digits on black  (team shown on the right)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).with_name("assets")
TEMPLATES = ASSETS / "scoresync_digits.npz"
GLYPH_W, GLYPH_H = 16, 24

# (x0, y0, x1, y1) as fractions of frame width/height
DEFAULT_LAYOUT = {
    "left_score": (1600 / 3840, 1850 / 2160, 1790 / 3840, 1945 / 2160),
    "right_score": (2060 / 3840, 1850 / 2160, 2245 / 3840, 1945 / 2160),
}


def crop(frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box
    return frame[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def _gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return img[..., :3].astype(np.float32).mean(axis=2)
    return img.astype(np.float32)


def _resize(mask: np.ndarray, w: int, h: int) -> np.ndarray:
    """Area-ish resize of a float mask without cv2 (works where only numpy is present)."""
    H, W = mask.shape
    ys = (np.arange(h + 1) * H / h).astype(int)
    xs = (np.arange(w + 1) * W / w).astype(int)
    out = np.zeros((h, w), np.float32)
    for i in range(h):
        for j in range(w):
            block = mask[ys[i]:max(ys[i + 1], ys[i] + 1), xs[j]:max(xs[j + 1], xs[j] + 1)]
            out[i, j] = block.mean() if block.size else 0
    return out


def glyphs(region: np.ndarray) -> list[np.ndarray]:
    """Split a score region into normalised digit glyphs (left to right)."""
    g = _gray(region)
    bg = np.median(g)
    fg = (np.abs(g - bg) > 70).astype(np.float32)
    # ignore thin borders / anti-aliasing specks
    cols = fg.sum(axis=0) > max(2, 0.04 * fg.shape[0])
    out = []
    x = 0
    W = len(cols)
    while x < W:
        if not cols[x]:
            x += 1
            continue
        x0 = x
        while x < W and cols[x]:
            x += 1
        x1 = x
        sub = fg[:, x0:x1]
        rows = np.where(sub.sum(axis=1) > 0)[0]
        if len(rows) == 0:
            continue
        y0, y1 = rows[0], rows[-1] + 1
        gh, gw = y1 - y0, x1 - x0
        if gh < 0.35 * fg.shape[0] or gw < 2:  # too short to be a score digit
            continue
        if x0 == 0 or x1 == W:                  # cut by the crop edge
            continue
        sub = sub[y0:y1]
        # pad to the glyph aspect ratio so "1" keeps its thinness
        target_w = int(round(gh * GLYPH_W / GLYPH_H))
        if gw < target_w:
            pad = target_w - gw
            sub = np.pad(sub, ((0, 0), (pad // 2, pad - pad // 2)))
        out.append(_resize(sub, GLYPH_W, GLYPH_H))
    return out


@dataclass
class Reading:
    left: int | None
    right: int | None
    conf: float


class ScoreReader:
    def __init__(self, layout: dict | None = None, templates: Path | None = TEMPLATES):
        self.layout = {**DEFAULT_LAYOUT, **(layout or {})}
        self.labels = None
        self.tmpl = None
        if templates and Path(templates).exists():
            d = np.load(templates)
            self.tmpl = d["glyphs"].reshape(len(d["labels"]), -1).astype(np.float32)
            self.labels = d["labels"].astype(int)
            n = np.linalg.norm(self.tmpl, axis=1, keepdims=True)
            self.tmpl_n = self.tmpl / np.maximum(n, 1e-6)

    @property
    def ready(self) -> bool:
        return self.tmpl is not None

    def _digit(self, g: np.ndarray) -> tuple[int, float]:
        v = g.reshape(-1)
        v = v / max(np.linalg.norm(v), 1e-6)
        sims = self.tmpl_n @ v
        i = int(np.argmax(sims))
        return int(self.labels[i]), float(sims[i])

    def _number(self, region: np.ndarray) -> tuple[int | None, float]:
        gs = glyphs(region)
        if not gs or len(gs) > 3:
            return None, 0.0
        digits, confs = [], []
        for g in gs:
            d, c = self._digit(g)
            digits.append(d)
            confs.append(c)
        return int("".join(map(str, digits))), min(confs)

    def read(self, frame: np.ndarray) -> Reading:
        if not self.ready:
            return Reading(None, None, 0.0)
        l, cl = self._number(crop(frame, self.layout["left_score"]))
        r, cr = self._number(crop(frame, self.layout["right_score"]))
        return Reading(l, r, min(cl, cr))


def clean_timeline(raw: list[tuple[float, int | None, int | None, float]], min_conf: float = 0.85,
                   stable: int = 2, resync: int = 5) -> list[dict]:
    """Turn noisy per-frame readings into a score timeline.

    raw: [(t, left, right, conf), ...] sorted by t.
    Returns [{"t": time_first_seen, "left", "right", "jump": bool}], one entry per score change.
    Normal changes (each team +0..3) must be read `stable` times in a row. Anything else
    (a scorer correction, or a jump because the recording was paused) is accepted after
    `resync` identical readings and marked jump=True if it is not a correction.
    """
    out: list[dict] = []
    cur = (0, 0)
    cand, cand_n, cand_t = None, 0, None
    for t, l, r, c in raw:
        if l is None or r is None or c < min_conf:
            continue
        s = (l, r)
        if s == cur:
            cand, cand_n = None, 0
            continue
        if s == cand:
            cand_n += 1
        else:
            cand, cand_n, cand_t = s, 1, t
        dl, dr = l - cur[0], r - cur[1]
        normal = 0 <= dl <= 3 and 0 <= dr <= 3
        if cand_n >= (stable if normal else resync):
            jump = not normal and not (dl <= 0 and dr <= 0)
            cur = s
            out.append({"t": round(cand_t, 2), "left": l, "right": r, "jump": jump})
            cand, cand_n = None, 0
    return out


def increments(timeline: list[dict]) -> tuple[list[dict], list[dict]]:
    """Score changes as basket events.

    Returns (baskets, gaps):
      baskets: [{"t", "side": "left"|"right", "points": 1..3}] after removing baskets that the
               scorer later took back (a decrease cancels the latest basket(s) of that side).
      gaps:    [{"t", "left": +pts, "right": +pts}] points that appeared in one jump
               (recording paused / scorer catching up) and can't be tied to a shot.
    """
    baskets: list[dict] = []
    gaps: list[dict] = []
    prev = {"left": 0, "right": 0}
    for e in timeline:
        if e.get("jump"):
            gaps.append({"t": e["t"], "left": e["left"] - prev["left"], "right": e["right"] - prev["right"]})
        else:
            for side in ("left", "right"):
                d = e[side] - prev[side]
                if d > 0:
                    baskets.append({"t": e["t"], "side": side, "points": d})
                elif d < 0:
                    need = -d
                    for b in reversed(baskets):
                        if b["side"] == side and not b.get("void") and need > 0:
                            b["void"] = True
                            need -= b["points"]
        prev = {"left": e["left"], "right": e["right"]}
    return [b for b in baskets if not b.get("void")], gaps


def scan(info, layout: dict | None = None, every: float = 1.0, start: float = 0.0,
         end: float | None = None, progress=None) -> tuple[list[dict], list[tuple]]:
    """Fast pass over a video reading only the overlay band from keyframes.

    Returns (clean timeline, raw readings). ~2 min for a 37-min 4K game.
    """
    from . import video as _video
    reader = ScoreReader(layout)
    if not reader.ready:
        return [], []
    W, H = info.size
    lay = reader.layout
    keys = ("left_score", "right_score")
    bx0 = max(0, int(min(lay[k][0] for k in keys) * W) - 4) // 2 * 2
    by0 = max(0, int(min(lay[k][1] for k in keys) * H) - 4) // 2 * 2
    bx1 = min(W, int(max(lay[k][2] for k in keys) * W) + 4)
    by1 = min(H, int(max(lay[k][3] for k in keys) * H) + 4)
    band = (bx0, by0, (bx1 - bx0) // 2 * 2, (by1 - by0) // 2 * 2)
    raw = []
    for t, f in _video.frames(info, fps=1 / every, keyframes_only=True, crop=band, start=start, end=end):
        vals = []
        conf = 1.0
        for k in keys:
            x0, y0, x1, y1 = lay[k]
            sub = f[int(y0 * H) - by0:int(y1 * H) - by0, int(x0 * W) - bx0:int(x1 * W) - bx0]
            v, c = reader._number(sub)
            vals.append(v)
            conf = min(conf, c)
        raw.append((t, vals[0], vals[1], conf))
        if progress:
            progress(t)
    return clean_timeline(raw), raw

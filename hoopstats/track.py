"""Multi-person tracker (ByteTrack-style two-stage matching) with camera-motion compensation.

XbotGo pans to follow play, so before matching we move every track by the frame-to-frame
image motion estimated in register.py. That keeps IDs stable during pans.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between Nx4 and Mx4 boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def _assign(cost: np.ndarray, max_cost: float) -> list[tuple[int, int]]:
    if cost.size == 0:
        return []
    if linear_sum_assignment is not None:
        r, c = linear_sum_assignment(cost)
        return [(i, j) for i, j in zip(r, c) if cost[i, j] <= max_cost]
    pairs, used_r, used_c = [], set(), set()
    for k in np.argsort(cost, axis=None):
        i, j = divmod(int(k), cost.shape[1])
        if i in used_r or j in used_c or cost[i, j] > max_cost:
            continue
        pairs.append((i, j))
        used_r.add(i)
        used_c.add(j)
    return pairs


def warp_boxes(boxes: np.ndarray, M: np.ndarray | None) -> np.ndarray:
    """Move boxes by image homography M (prev -> cur), keeping their size scaled."""
    if M is None or len(boxes) == 0:
        return boxes
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    by = boxes[:, 3]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    P = np.c_[cx, by, np.ones(len(boxes))] @ M.T
    P = P[:, :2] / P[:, 2:3]
    s = np.sqrt(abs(np.linalg.det(M[:2, :2]))) if M is not None else 1.0
    return np.c_[P[:, 0] - w * s / 2, P[:, 1] - h * s, P[:, 0] + w * s / 2, P[:, 1]]


class Track:
    __slots__ = ("id", "box", "vel", "hits", "miss", "conf")

    def __init__(self, tid: int, box: np.ndarray, conf: float):
        self.id, self.box, self.conf = tid, box.astype(float), conf
        self.vel = np.zeros(2)
        self.hits, self.miss = 1, 0


class Tracker:
    def __init__(self, high: float = 0.5, low: float = 0.2, max_miss: int = 20, min_hits: int = 3):
        self.high, self.low = high, low
        self.max_miss, self.min_hits = max_miss, min_hits
        self.tracks: list[Track] = []
        self.next_id = 1

    def _predict(self, M):
        if not self.tracks:
            return np.zeros((0, 4))
        boxes = np.stack([t.box for t in self.tracks])
        boxes = warp_boxes(boxes, M)
        v = np.stack([t.vel for t in self.tracks])
        boxes = boxes + np.c_[v, v]
        return boxes

    def _cost(self, pred: np.ndarray, dets: np.ndarray) -> np.ndarray:
        c = 1 - iou(pred, dets[:, :4])
        # also allow matches by foot-point distance (fast movers / small boxes)
        pf = np.c_[(pred[:, 0] + pred[:, 2]) / 2, pred[:, 3]]
        df = np.c_[(dets[:, 0] + dets[:, 2]) / 2, dets[:, 3]]
        h = np.maximum((pred[:, 3] - pred[:, 1])[:, None], 1)
        d = np.linalg.norm(pf[:, None] - df[None], axis=2) / h
        return np.minimum(c, 0.4 + 0.6 * np.clip(d / 0.5, 0, 1) + (d > 0.5) * 1.0)

    def update(self, dets: np.ndarray, M: np.ndarray | None = None) -> list[tuple[int, np.ndarray, float]]:
        """dets Nx5 [x1,y1,x2,y2,conf]; M = image motion prev->cur. Returns confirmed tracks seen now."""
        dets = np.asarray(dets, float).reshape(-1, 5)
        pred = self._predict(M)
        hi = dets[dets[:, 4] >= self.high]
        lo = dets[(dets[:, 4] < self.high) & (dets[:, 4] >= self.low)]
        unmatched_t = list(range(len(self.tracks)))
        matched: dict[int, np.ndarray] = {}
        for pool, max_cost in ((hi, 0.75), (lo, 0.6)):
            if not len(pool) or not unmatched_t:
                rest = pool
            else:
                cost = self._cost(pred[unmatched_t], pool)
                pairs = _assign(cost, max_cost)
                used = set()
                for i, j in pairs:
                    matched[unmatched_t[i]] = pool[j]
                    used.add(j)
                unmatched_t = [t for k, t in enumerate(unmatched_t) if k not in {i for i, _ in pairs}]
                rest = np.array([pool[j] for j in range(len(pool)) if j not in used]).reshape(-1, 5)
            if pool is hi:
                new_dets = rest
        out = []
        for ti, det in matched.items():
            t = self.tracks[ti]
            old_c = np.array([(pred[ti, 0] + pred[ti, 2]) / 2, pred[ti, 3]])
            new_c = np.array([(det[0] + det[2]) / 2, det[3]])
            t.vel = 0.6 * t.vel + 0.4 * (new_c - old_c + t.vel)
            t.vel = np.clip(t.vel, -60, 60)
            t.box, t.conf = det[:4].copy(), float(det[4])
            t.hits += 1
            t.miss = 0
        for ti in unmatched_t:
            t = self.tracks[ti]
            t.box = pred[ti]
            t.miss += 1
            t.vel *= 0.7
        for d in new_dets:
            self.tracks.append(Track(self.next_id, d[:4], float(d[4])))
            self.next_id += 1
        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]
        for t in self.tracks:
            if t.miss == 0 and t.hits >= self.min_hits:
                out.append((t.id, t.box.copy(), t.conf))
        return out


class BallTracker:
    """Pick the one real ball among noisy 'sports ball' detections using motion continuity."""

    def __init__(self, max_jump_frac: float = 0.12, max_gap: int = 8):
        self.last = None          # (x, y, r)
        self.vel = np.zeros(2)
        self.gap = 0
        self.max_jump_frac, self.max_gap = max_jump_frac, max_gap

    def update(self, balls: np.ndarray, width: int, M: np.ndarray | None = None,
               persons: np.ndarray | None = None) -> tuple[float, float, float, float] | None:
        """Returns (x, y, radius, conf) or None."""
        balls = np.asarray(balls, float).reshape(-1, 5)
        if len(balls) and persons is not None and len(persons):
            # plausible size: ball diameter ~ 0.08-0.35 of a nearby player's height
            med_h = float(np.median(persons[:, 3] - persons[:, 1]))
            d = np.maximum(balls[:, 2] - balls[:, 0], balls[:, 3] - balls[:, 1])
            balls = balls[(d > 0.04 * med_h) & (d < 0.4 * med_h)]
        if self.last is not None and M is not None:
            p = np.array([self.last[0], self.last[1], 1.0]) @ M.T
            self.last = (p[0] / p[2], p[1] / p[2], self.last[2])
        cand = None
        if len(balls):
            c = np.c_[(balls[:, 0] + balls[:, 2]) / 2, (balls[:, 1] + balls[:, 3]) / 2]
            if self.last is not None and self.gap <= self.max_gap:
                pred = np.array(self.last[:2]) + self.vel * (self.gap + 1)
                dist = np.linalg.norm(c - pred, axis=1)
                lim = self.max_jump_frac * width * (1 + 0.5 * self.gap)
                score = balls[:, 4] - dist / lim
                k = int(np.argmax(score))
                if dist[k] <= lim:
                    cand = k
                elif balls[k, 4] > 0.5:
                    cand = k
            else:
                cand = int(np.argmax(balls[:, 4]))
        if cand is None:
            self.gap += 1
            if self.gap > self.max_gap:
                self.last, self.vel = None, np.zeros(2)
            return None
        x, y = c[cand]
        r = max(balls[cand, 2] - balls[cand, 0], balls[cand, 3] - balls[cand, 1]) / 2
        if self.last is not None:
            self.vel = 0.5 * self.vel + 0.5 * (np.array([x, y]) - np.array(self.last[:2])) / (self.gap + 1)
        self.last, self.gap = (x, y, r), 0
        return float(x), float(y), float(r), float(balls[cand, 4])

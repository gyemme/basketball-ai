"""Court registration for a moving (panning / zooming) XbotGo camera.

A handful of calibrated keyframes (court landmarks clicked once with `hoopstats calibrate`)
anchor the mapping. Every analysed frame is matched to the most similar keyframe using SIFT
features on the static background (walls, bleachers, floor lines). Between full matches the
camera motion is followed with optical flow. A camera that only rotates/zooms about a fixed
point maps frame-to-frame by a homography, so this is exact up to lens distortion.

Outputs per frame:
  H  : court (ft) -> image (analysis px) homography, or None
  conf: 0..1 registration confidence
  M  : image motion previous frame -> this frame (for the tracker), or None
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

FEAT_W = 960          # features are computed on frames scaled to this width
BOTTOM_MASK = 0.16    # ignore the bottom band (ScoreSync overlay, clock, watermark)


def _mask(shape, boxes=None, scale=1.0):
    h, w = shape[:2]
    m = np.full((h, w), 255, np.uint8)
    m[int(h * (1 - BOTTOM_MASK)):] = 0
    if boxes is not None:
        for b in boxes:
            x1, y1, x2, y2 = (np.asarray(b[:4]) * scale).astype(int)
            pad = int(0.15 * (y2 - y1))
            m[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = 0
    return m


class Features:
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.03)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)

    def small(self, frame):
        s = FEAT_W / frame.shape[1]
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return cv2.resize(g, (FEAT_W, int(round(frame.shape[0] * s))), interpolation=cv2.INTER_AREA), s

    def compute(self, gray_small, mask):
        kp, des = self.sift.detectAndCompute(gray_small, mask)
        pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
        return pts, des

    def match(self, a, b, min_inliers=25):
        """Homography mapping points of image A -> image B. Returns (H, inliers)."""
        pa, da = a
        pb, db = b
        if da is None or db is None or len(pa) < 8 or len(pb) < 8:
            return None, 0
        m = self.matcher.knnMatch(da, db, k=2)
        good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.75 * x[1].distance]
        if len(good) < min_inliers:
            return None, len(good)
        src = pa[[g.queryIdx for g in good]]
        dst = pb[[g.trainIdx for g in good]]
        H, inl = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        n = int(inl.sum()) if inl is not None else 0
        if H is None or n < min_inliers:
            return None, n
        return H, n


def load_calib(path: str | Path | None) -> dict | None:
    if not path or not Path(path).exists():
        return None
    with open(path) as f:
        c = json.load(f)
    c["_dir"] = str(Path(path).parent)
    return c


class Registrar:
    def __init__(self, calib: dict | None, size: tuple[int, int], refresh: int = 5):
        self.calib = calib
        self.iw, self.ih = size
        self.refresh = refresh
        self.feat = Features()
        self.keys = []   # (H_court_to_key_analysis_px, (pts, des))
        if calib:
            cw = calib.get("width", self.iw)
            for kf in calib["keyframes"]:
                img = cv2.imread(str(Path(calib["_dir"]) / kf["image"]))
                if img is None or kf.get("H") is None:
                    continue
                img = cv2.resize(img, (self.iw, self.ih)) if img.shape[1] != self.iw else img
                Hk = np.array(kf["H"], float)
                if cw != self.iw:
                    s = self.iw / cw
                    Hk = np.diag([s, s, 1.0]) @ Hk
                g, sc = self.feat.small(img)
                self.keys.append((Hk, self.feat.compute(g, _mask(g.shape))))
        self.prev_gray = None
        self.prev_pts = None
        self.H = None          # current court->analysis-px
        self.conf = 0.0
        self.since = 10 ** 9
        self.s = FEAT_W / self.iw

    @property
    def calibrated(self) -> bool:
        return bool(self.keys)

    def _to_small(self, H):
        S = np.diag([self.s, self.s, 1.0])
        return S @ H

    def _from_small(self, Hs):
        Si = np.diag([1 / self.s, 1 / self.s, 1.0])
        return Si @ Hs

    def update(self, frame: np.ndarray, persons: np.ndarray | None = None):
        g, _ = self.feat.small(frame)
        mask = _mask(g.shape, persons, self.s)
        M_small = None
        # 1) frame-to-frame motion with optical flow
        if self.prev_gray is not None and self.prev_pts is not None and len(self.prev_pts) >= 12:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, g, self.prev_pts, None,
                                                  winSize=(21, 21), maxLevel=3)
            ok = st.reshape(-1) == 1
            if ok.sum() >= 12:
                Mh, inl = cv2.findHomography(self.prev_pts[ok], nxt[ok], cv2.RANSAC, 2.0)
                if Mh is not None and inl is not None and inl.sum() >= 10:
                    M_small = Mh
        M = self._from_small(M_small) @ np.diag([self.s, self.s, 1.0]) if M_small is not None else None
        if M is not None and self.H is not None:
            self.H = M @ self.H
            self.conf *= 0.97
        elif M is None:
            self.conf *= 0.5
        self.since += 1
        # 2) re-anchor on a keyframe
        if self.keys and (self.since >= self.refresh or self.H is None or self.conf < 0.4):
            cur = self.feat.compute(g, mask)
            best = (None, 0, None)
            for Hk, kfeat in self.keys:
                Hm, n = self.feat.match(kfeat, cur)      # key_small -> cur_small
                if Hm is not None and n > best[1]:
                    best = (Hm, n, Hk)
            if best[0] is not None:
                Hc = self._from_small(best[0] @ self._to_small(best[2]))
                if self._sane(Hc):
                    self.H = Hc
                    self.conf = float(min(1.0, best[1] / 120))
                    self.since = 0
        if self.H is not None and not self._sane(self.H):
            self.H, self.conf = None, 0.0
        # 3) features to follow next frame
        pts = cv2.goodFeaturesToTrack(g, maxCorners=400, qualityLevel=0.01, minDistance=8, mask=mask)
        self.prev_pts = pts if pts is not None else None
        self.prev_gray = g
        return (None if self.H is None else self.H.copy()), self.conf, M

    def _sane(self, H) -> bool:
        """Plausible court mapping: the floor point at the image centre lies near the court,
        and a 1 ft step there covers a believable number of pixels."""
        if H is None or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-12:
            return False
        try:
            X = np.linalg.solve(H, np.array([self.iw / 2, self.ih * 0.6, 1.0]))
        except np.linalg.LinAlgError:
            return False
        if abs(X[2]) < 1e-12:
            return False
        x, y = X[0] / X[2], X[1] / X[2]
        if not (-60 < x < 160 and -60 < y < 110):
            return False
        p = np.array([[x, y, 1], [x + 1, y, 1], [x, y + 1, 1]]) @ H.T
        if np.any(np.sign(p[:, 2]) != np.sign(p[0, 2])):
            return False
        uv = p[:, :2] / p[:, 2:3]
        step = max(np.linalg.norm(uv[1] - uv[0]), np.linalg.norm(uv[2] - uv[0]))
        return bool(1.0 < step < 0.5 * self.iw)


def court_to_image(H, xy):
    p = np.c_[np.asarray(xy, float).reshape(-1, 2), np.ones(len(np.asarray(xy).reshape(-1, 2)))] @ H.T
    return p[:, :2] / p[:, 2:3]


def image_to_court(H, uv):
    Hi = np.linalg.inv(H)
    return court_to_image(Hi, uv)

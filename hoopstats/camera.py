"""Camera geometry from a court homography.

Given H (court feet -> image pixels) for one frame, recover a pinhole camera (focal length,
rotation, position) assuming square pixels and the principal point at the image centre.
That lets us project points ABOVE the floor, e.g. the rim (10 ft up), without a rim detector.
"""
from __future__ import annotations

import math

import numpy as np

RIM_HEIGHT_FT = 10.0
RIM_RADIUS_FT = 0.75


def focal_from_homography(H: np.ndarray, cx: float, cy: float) -> float | None:
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], float)
    Hc = T @ H
    h1, h2 = Hc[:, 0], Hc[:, 1]
    ests = []
    # orthogonality of r1, r2
    den = h1[2] * h2[2]
    if abs(den) > 1e-12:
        f2 = -(h1[0] * h2[0] + h1[1] * h2[1]) / den
        if f2 > 0:
            ests.append(f2)
    # equal norms of r1, r2
    den = h1[2] ** 2 - h2[2] ** 2
    if abs(den) > 1e-12:
        f2 = (h2[0] ** 2 + h2[1] ** 2 - h1[0] ** 2 - h1[1] ** 2) / den
        if f2 > 0:
            ests.append(f2)
    if not ests:
        return None
    f = math.sqrt(float(np.median(ests)))
    return f if 50 < f < 50000 else None


class Camera:
    def __init__(self, H: np.ndarray, width: int, height: int, focal: float | None = None):
        self.H = np.asarray(H, float)
        self.cx, self.cy = width / 2.0, height / 2.0
        f = focal or focal_from_homography(self.H, self.cx, self.cy)
        if f is None:
            f = width / (2 * math.tan(math.radians(70) / 2))  # ~70 deg horizontal FOV fallback
        self.f = f
        K = np.array([[f, 0, self.cx], [0, f, self.cy], [0, 0, 1]])
        self.K = K
        B = np.linalg.inv(K) @ self.H
        lam = 1.0 / max(np.linalg.norm(B[:, 0]), 1e-12)
        # the floor point seen at the image centre must be in front of the camera
        X = np.linalg.solve(self.H, np.array([self.cx, self.cy, 1.0]))
        X = X / X[2]
        s = (self.H @ X)[2]
        if s * lam < 0:
            lam = -lam
        r1, r2, t = B[:, 0] * lam, B[:, 1] * lam, B[:, 2] * lam
        r3 = np.cross(r1, r2)
        R = np.stack([r1, r2, r3], axis=1)
        U, _, Vt = np.linalg.svd(R)
        R = U @ Vt
        # make +z point UP (camera centre above the floor)
        C = -R.T @ t
        if C[2] < 0:
            R[:, 2] = -R[:, 2]
            C = -R.T @ t
        self.R, self.t, self.center = R, t, C

    def project(self, pts: np.ndarray) -> np.ndarray:
        """Project Nx3 court points (ft, z up) to Nx2 pixels."""
        P = np.asarray(pts, float).reshape(-1, 3)
        Xc = (self.R @ P.T).T + self.t
        uv = (self.K @ Xc.T).T
        return uv[:, :2] / uv[:, 2:3]

    def depth(self, pts: np.ndarray) -> np.ndarray:
        P = np.asarray(pts, float).reshape(-1, 3)
        return ((self.R @ P.T).T + self.t)[:, 2]

    def rim(self, hoop_xy: tuple[float, float]) -> tuple[float, float, float] | None:
        """(u, v, radius_px) of a rim; None if behind the camera."""
        hx, hy = hoop_xy
        c = np.array([[hx, hy, RIM_HEIGHT_FT]])
        if self.depth(c)[0] <= 0:
            return None
        ring = np.array([[hx + RIM_RADIUS_FT * math.cos(a), hy + RIM_RADIUS_FT * math.sin(a), RIM_HEIGHT_FT]
                         for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)])
        uv = self.project(ring)
        u, v = self.project(c)[0]
        r = float(np.max(np.abs(uv[:, 0] - u)))
        return float(u), float(v), max(r, 1.0)


def apply_h(H: np.ndarray, pts) -> np.ndarray:
    P = np.asarray(pts, float).reshape(-1, 2)
    q = np.c_[P, np.ones(len(P))] @ np.asarray(H, float).T
    return q[:, :2] / q[:, 2:3]

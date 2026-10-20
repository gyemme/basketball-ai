"""Court calibration in your web browser (one time per gym / camera position).

    hoopstats calibrate "videos/game.MP4"

1. Samples frames from the video (keyframes only, so it takes ~1-2 minutes for a full 4K game)
   and groups them into the distinct camera views XbotGo used (left end, right end, wide...).
2. Opens a page in your browser showing one frame per view next to a court diagram.
   Pick a landmark on the diagram (or follow the highlighted one), click where it is on the
   frame. 4+ well spread landmarks per view are enough; magenta court lines show the fit.
3. "Save calibration" writes calib.json and the analysis continues in the terminal.
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from .court import Court
from . import video as vid
from .register import Features, _mask

UI = Path(__file__).with_name("calibrate_ui.html")


def landmarks(c: Court) -> list[tuple[str, float, float]]:
    L, W, cy = c.length, c.width, c.width / 2
    lw = c.lane_width / 2
    r = c.three_pt or 0
    pts = [
        ("Near-left corner (baseline meets near sideline)", 0, 0),
        ("Far-left corner", 0, W),
        ("Near-right corner", L, 0),
        ("Far-right corner", L, W),
        ("Half-court line meets near sideline", L / 2, 0),
        ("Half-court line meets far sideline", L / 2, W),
        ("Center circle - near point", L / 2, cy - 6),
        ("Center circle - far point", L / 2, cy + 6),
    ]
    for side, x0, sgn in (("Left", 0.0, 1), ("Right", L, -1)):
        pts += [
            (f"{side} lane - baseline corner, near", x0, cy - lw),
            (f"{side} lane - baseline corner, far", x0, cy + lw),
            (f"{side} free-throw line - near end", x0 + sgn * c.ft_line, cy - lw),
            (f"{side} free-throw line - far end", x0 + sgn * c.ft_line, cy + lw),
            (f"{side} free-throw circle - top", x0 + sgn * (c.ft_line + 6), cy),
        ]
        if r:
            hx = x0 + sgn * c.hoop_offset
            pts += [
                (f"{side} 3-pt line meets baseline, near", x0, cy - (c.corner_three or r)),
                (f"{side} 3-pt line meets baseline, far", x0, cy + (c.corner_three or r)),
                (f"{side} 3-pt arc - top", hx + sgn * r, cy),
            ]
    return pts


def degenerate(court_pts) -> bool:
    """True if the landmarks can't define a homography (fewer than 4, or 3 of 4 on one line)."""
    P = np.asarray(court_pts, float)
    if len(P) < 4:
        return True
    if len(P) > 4:
        # spread: the points must not all lie close to a single line
        c = P - P.mean(0)
        sv = np.linalg.svd(c, compute_uv=False)
        return bool(sv[-1] < 0.08 * sv[0])
    from itertools import combinations
    for a, b, c in combinations(range(4), 3):
        u, v = P[b] - P[a], P[c] - P[a]
        area = abs(u[0] * v[1] - u[1] * v[0]) / 2
        if area < 8.0:   # square feet
            return True
    return False


def crossed(court_pts, img_pts) -> bool:
    """A floor homography keeps the left/right order of every point triple (the court is a plane
    in front of the camera), so a triple whose orientation flips means a mislabeled point."""
    from itertools import combinations
    C = np.asarray(court_pts, float)
    P = np.asarray(img_pts, float)
    signs = []
    for a, b, c in combinations(range(len(C)), 3):
        u, v = C[b] - C[a], C[c] - C[a]
        ca = u[0] * v[1] - u[1] * v[0]
        u, v = P[b] - P[a], P[c] - P[a]
        ia = u[0] * v[1] - u[1] * v[0]
        if abs(ca) < 8 or abs(ia) < 50:   # nearly collinear triples say nothing
            continue
        signs.append(np.sign(ca) * np.sign(ia))
    return bool(signs) and len(set(signs)) > 1


def fit_h(pairs) -> tuple[np.ndarray | None, float]:
    if len(pairs) < 4:
        return None, float("inf")
    src = np.float32([p[0] for p in pairs])
    dst = np.float32([p[1] for p in pairs])
    H, _ = cv2.findHomography(src, dst, 0)
    if H is None:
        return None, float("inf")
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = float(np.sqrt(((proj - dst) ** 2).sum(1)).mean())
    return H, err


# --------------------------------------------------------------------------- views

def sample_frames(videos, every: float, width: int, start: float = 0.0, end: float | None = None, log=print):
    out = []
    total = sum(v.duration for v in videos)
    t_last = time.time()
    for v in videos:
        a = max(0.0, start - v.offset)
        b = None if end is None else end - v.offset
        if b is not None and b <= 0 or a >= v.duration:
            continue
        for t, f in vid.frames(v, fps=1 / every, width=width, start=a, end=b, keyframes_only=True):
            out.append((v.offset + t, f.copy()))
            if time.time() - t_last > 3:
                t_last = time.time()
                log(f"  ... {int(v.offset + t) // 60}:{int(v.offset + t) % 60:02d} / {int(total) // 60} min")
    return out


def group_views(samples, min_inliers: int = 40, max_views: int = 6, log=print):
    """Greedy grouping of sampled frames into camera views that register to each other."""
    feat = Features()
    feats = []
    for _, f in samples:
        g, _ = feat.small(f)
        feats.append(feat.compute(g, _mask(g.shape)))
    left = list(range(len(samples)))
    views = []
    while left and len(views) < max_views:
        # seed: the frame in the middle of what is left (avoids warm-up / end-of-game oddities)
        seed = left[len(left) // 2]
        covered = [i for i in left if i == seed or feat.match(feats[seed], feats[i])[1] >= min_inliers]
        views.append({"seed": seed, "covers": covered})
        left = [i for i in left if i not in covered]
    views.sort(key=lambda v: -len(v["covers"]))
    n = max(1, len(samples))
    for v in views:
        v["share"] = round(100 * len(v["covers"]) / n)
    uncovered = round(100 * len(left) / n)
    return views, uncovered


# --------------------------------------------------------------------------- web UI

def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def run(video_path, out, court: Court, width: int = 1920, sample_every: float = 20.0,
        start: float = 0.0, end: float | None = None, open_browser: bool = True, log=print) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    videos = vid.collect([video_path] if isinstance(video_path, (str, Path)) else video_path)
    log("Sampling frames to find the different camera views (keyframes only) ...")
    t0 = time.time()
    samples = sample_frames(videos, sample_every, width, start, end, log=log)
    if not samples:
        sys.exit("Could not read frames from the video.")
    log(f"  {len(samples)} frames sampled in {time.time() - t0:.0f}s; grouping camera views ...")
    views, uncovered = group_views(samples, log=log)
    log(f"  {len(views)} camera views" + (f" ({uncovered}% of frames match none of them)" if uncovered else ""))

    imgs = {}
    for i, v in enumerate(views):
        ok, buf = cv2.imencode(".jpg", samples[v["seed"]][1], [cv2.IMWRITE_JPEG_QUALITY, 90])
        imgs[i] = buf.tobytes()
    lms = [{"id": k, "name": n, "x": x, "y": y} for k, (n, x, y) in enumerate(landmarks(court))]
    cfg = {"court": court.to_dict(), "landmarks": lms,
           "views": [{"id": i, "t": round(samples[v["seed"]][0], 1), "share": v["share"],
                      "img": f"/img/{i}.jpg"} for i, v in enumerate(views)],
           "width": width}
    old = out / "calib.json"
    if old.exists():   # pre-fill points from an earlier session for the same views
        try:
            cfg["previous"] = json.loads(old.read_text()).get("keyframes", [])
        except Exception:
            pass
    page = UI.read_text(encoding="utf-8").replace("__CONFIG__", json.dumps(cfg))
    done = threading.Event()
    result: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, page.encode(), "text/html; charset=utf-8")
            if self.path.startswith("/img/"):
                try:
                    i = int(self.path.split("/")[-1].split(".")[0])
                    return self._send(200, imgs[i], "image/jpeg")
                except Exception:
                    pass
            self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/save":
                return self._send(404, b"", "text/plain")
            n = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(n) or b"{}")
            keyframes, report, problems = [], [], []
            for vw in data.get("views", []):
                pts = vw.get("points", [])
                if not pts:
                    continue
                i0 = int(vw["id"]) + 1
                if len(pts) < 4:
                    problems.append(f"View {i0}: only {len(pts)} points (need 4+, or clear the view to skip it).")
                    continue
                if degenerate([p["court"] for p in pts]):
                    problems.append(f"View {i0}: the landmarks are (almost) on one line. Add points away from that line, "
                                    "e.g. on the other side of the lane or on the 3-point arc.")
                    continue
                if crossed([p["court"] for p in pts], [p["img"] for p in pts]):
                    problems.append(f"View {i0}: some points are flipped relative to the court diagram, so a point is "
                                    "probably on the wrong landmark (e.g. near and far ends swapped).")
                    continue
                H, err = fit_h([(tuple(p["court"]), tuple(p["img"])) for p in pts])
                if H is None or err > 25:
                    problems.append(f"View {i0}: points don't fit a court (average error {err:.0f} px). "
                                    "A point is probably on the wrong landmark, or the court size in game.yaml is off.")
                    continue
                i = int(vw["id"])
                name = f"keyframe_{i:02d}.jpg"
                (out / name).write_bytes(imgs[i])
                keyframes.append({"t": cfg["views"][i]["t"], "image": name, "points": pts,
                                  "H": H.tolist(), "fit_error_px": round(err, 2)})
                report.append({"id": i, "error": round(err, 1), "points": len(pts)})
            if problems or not keyframes:
                msg = " ".join(problems) or "Place at least 4 landmarks on at least one view."
                return self._send(200, json.dumps({"ok": False, "message": msg}).encode(), "application/json")
            calib = {"video": [str(v.path) for v in videos], "width": width,
                     "height": int(round(videos[0].size[1] * width / videos[0].size[0])),
                     "court": court.to_dict(), "keyframes": keyframes}
            (out / "calib.json").write_text(json.dumps(calib, indent=1))
            result["calib"] = calib
            result["report"] = report
            self._send(200, json.dumps({"ok": True, "views": report}).encode(), "application/json")
            done.set()

    port = _free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    url = f"http://127.0.0.1:{port}/"
    log("")
    log(f"Calibration page:  {url}")
    log("  Mark 4+ court landmarks on each view, then click 'Save calibration'. (Ctrl-C to cancel)")
    if open_browser:
        webbrowser.open(url)
    try:
        while not done.wait(0.5):
            pass
    except KeyboardInterrupt:
        srv.shutdown()
        log("Calibration cancelled.")
        return out / "calib.json"
    time.sleep(0.3)
    srv.shutdown()
    for r in result.get("report", []):
        log(f"  view {r['id'] + 1}: {r['points']} points, average fit error {r['error']} px")
    log(f"Saved {out / 'calib.json'}")
    return out / "calib.json"

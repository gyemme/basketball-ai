"""One-off helper: harvest ScoreSync overlay digit glyphs from a video and cluster them.

Step 1:  python scripts/build_digit_templates.py harvest VIDEO --out runs/_glyphs
         -> runs/_glyphs/clusters.npz + montage.png (one row per cluster)
Step 2:  look at montage.png and write the digit for each row:
         python scripts/build_digit_templates.py label runs/_glyphs "3,8,0,1,..."
         -> hoopstats/assets/scoresync_digits.npz
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hoopstats import video, scoreboard  # noqa: E402


def kmeans(X: np.ndarray, k: int, iters: int = 50, seed: int = 0):
    rng = np.random.default_rng(seed)
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        a = d.argmin(1)
        C2 = np.stack([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
        if np.allclose(C, C2):
            break
        C = C2
    return C, a


def harvest(path: str, out: str, every: float = 4.0, k: int = 16):
    info = video.probe(path)
    W, H = info.size
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gl = []
    lay = scoreboard.DEFAULT_LAYOUT
    # only pipe the overlay band, not the whole 4K frame
    bx0 = int(min(b[0] for b in lay.values()) * W) - 4
    by0 = int(min(b[1] for b in lay.values()) * H) - 4
    bx1 = int(max(b[2] for b in lay.values()) * W) + 4
    by1 = int(max(b[3] for b in lay.values()) * H) + 4
    bx0, by0 = bx0 // 2 * 2, by0 // 2 * 2
    band = (bx0, by0, (bx1 - bx0) // 2 * 2, (by1 - by0) // 2 * 2)
    for t, f in video.frames(info, fps=1 / every, keyframes_only=True, hwaccel=False, crop=band):
        for key in ("left_score", "right_score"):
            x0, y0, x1, y1 = lay[key]
            sub = f[int(y0 * H) - by0:int(y1 * H) - by0, int(x0 * W) - bx0:int(x1 * W) - bx0]
            gl += scoreboard.glyphs(sub)
        if int(t) % 300 == 0:
            print(f"  t={t:.0f}s glyphs={len(gl)}", flush=True)
    X = np.stack([g.reshape(-1) for g in gl])
    C, a = kmeans(X, min(k, len(X)))
    np.savez(out_dir / "clusters.npz", centers=C, assign=a, glyphs=X)
    # montage: each row = cluster, up to 12 examples
    from PIL import Image
    gw, gh = scoreboard.GLYPH_W, scoreboard.GLYPH_H
    rows = []
    for j in range(len(C)):
        ex = X[a == j][:12]
        row = np.zeros((gh, gw * 13), np.float32)
        row[:, :gw] = C[j].reshape(gh, gw)
        for i, e in enumerate(ex):
            row[:, gw * (i + 1):gw * (i + 2)] = e.reshape(gh, gw)
        rows.append(np.pad(row, ((2, 2), (0, 0)), constant_values=0.5))
        print(f"cluster {j}: {int((a == j).sum())} glyphs")
    m = (np.vstack(rows) * 255).astype(np.uint8)
    Image.fromarray(m).resize((m.shape[1] * 4, m.shape[0] * 4), Image.NEAREST).save(out_dir / "montage.png")
    print("wrote", out_dir / "montage.png")


def label(out: str, labels: str):
    d = np.load(Path(out) / "clusters.npz")
    lab = [x.strip() for x in labels.split(",")]
    X, a = d["glyphs"], d["assign"]
    G, L = [], []
    for j, l in enumerate(lab):
        if l in ("", "x", "?"):
            continue
        ex = X[a == j]
        # keep several real examples per digit (nearest-neighbour is more robust than the mean)
        for e in ex[:: max(1, len(ex) // 8)][:8]:
            G.append(e.reshape(scoreboard.GLYPH_H, scoreboard.GLYPH_W))
            L.append(int(l))
    scoreboard.ASSETS.mkdir(exist_ok=True)
    np.savez_compressed(scoreboard.TEMPLATES, glyphs=np.stack(G).astype(np.float16), labels=np.array(L))
    print("wrote", scoreboard.TEMPLATES, "digits:", sorted(set(L)))


if __name__ == "__main__":
    if sys.argv[1] == "harvest":
        harvest(sys.argv[2], sys.argv[4] if len(sys.argv) > 4 else "runs/_glyphs")
    else:
        label(sys.argv[2], sys.argv[3])

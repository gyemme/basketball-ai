"""Video probing and frame reading via ffmpeg (handles XbotGo .MP4/.MOV, H.264/HEVC, rotation,
odd filenames like '18:00:21.MP4', and games split over several files)."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}


def _ff(path: Path | str) -> str:
    # 'file:' stops ffmpeg treating '18:00:21.MP4' as a protocol '18:'
    return "file:" + str(Path(path).resolve())


def need(tool: str) -> str:
    exe = shutil.which(tool)
    if not exe:
        sys.exit(f"{tool} not found. Install it (macOS: brew install ffmpeg).")
    return exe


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    duration: float
    codec: str
    rotation: int = 0
    vfr: bool = False
    creation_time: str | None = None
    device: str | None = None
    offset: float = 0.0      # start time of this file on the combined game timeline

    @property
    def size(self) -> tuple[int, int]:
        if self.rotation in (90, 270):
            return self.height, self.width
        return self.width, self.height

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["path"] = str(self.path)
        return d


def _rate(s: str | None) -> float:
    if not s or s in ("0/0",):
        return 0.0
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b) if float(b) else 0.0
    return float(s)


def probe(path: Path | str) -> VideoInfo:
    need("ffprobe")
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", _ff(path)],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {out.stderr.strip()}")
    d = json.loads(out.stdout)
    vs = [s for s in d["streams"] if s.get("codec_type") == "video"]
    if not vs:
        raise RuntimeError(f"No video stream in {path}")
    v = vs[0]
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"]) % 360
    if "rotate" in (v.get("tags") or {}):
        rot = int(v["tags"]["rotate"]) % 360
    rot = (360 - rot) % 360 if rot < 0 else rot
    r, avg = _rate(v.get("r_frame_rate")), _rate(v.get("avg_frame_rate"))
    fmt = d.get("format", {})
    tags = fmt.get("tags", {}) or {}
    return VideoInfo(
        path=Path(path), width=int(v["width"]), height=int(v["height"]),
        fps=avg or r, duration=float(fmt.get("duration") or v.get("duration") or 0),
        codec=v.get("codec_name", "?"), rotation=rot,
        vfr=bool(r and avg and abs(r - avg) / r > 0.01),
        creation_time=tags.get("com.apple.quicktime.creationdate") or tags.get("creation_time"),
        device=tags.get("com.apple.quicktime.model"),
    )


def collect(inputs: list[str | Path]) -> list[VideoInfo]:
    """Expand files/folders into an ordered list of videos on one game timeline."""
    files: list[Path] = []
    for i in inputs:
        p = Path(i).expanduser()
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT and not f.name.startswith("."))
        elif p.exists():
            files.append(p)
        else:
            sys.exit(f"Video not found: {p}")
    if not files:
        sys.exit("No video files found.")
    infos = [probe(f) for f in files]
    infos.sort(key=lambda v: (v.creation_time or "", v.path.name))
    t = 0.0
    for v in infos:
        v.offset = t
        t += v.duration
    return infos


def frames(info: VideoInfo, fps: float, width: int | None = None, start: float = 0.0,
           end: float | None = None, gray: bool = False, crop: tuple[int, int, int, int] | None = None,
           keyframes_only: bool = False, hwaccel: bool = True) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (time_s, frame) at `fps`, optionally scaled to `width` (aspect kept) or cropped.

    Frames are BGR uint8 (or single-channel gray). Time is relative to this file's start.
    crop = (x, y, w, h) in source pixels, applied before scaling.
    """
    need("ffmpeg")
    W, H = info.size
    if crop:
        # yuv420 sources can only be cropped to even sizes/offsets
        crop = (crop[0] // 2 * 2, crop[1] // 2 * 2, crop[2] // 2 * 2, crop[3] // 2 * 2)
        W, H = crop[2], crop[3]
    if width and width < W:
        h = int(round(H * width / W / 2) * 2)
        W, H = width, h
    filters = []
    if crop:
        filters.append(f"crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]}")
    if (W, H) != ((crop[2], crop[3]) if crop else info.size):
        filters.append(f"scale={W}:{H}:flags=area")
    filters.append(f"fps={fps}")
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if hwaccel and sys.platform == "darwin":
        cmd += ["-hwaccel", "videotoolbox"]
    if keyframes_only:
        cmd += ["-skip_frame", "nokey"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", _ff(info.path)]
    if end is not None:
        cmd += ["-t", f"{max(end - start, 0):.3f}"]
    cmd += ["-an", "-vf", ",".join(filters), "-pix_fmt", "gray" if gray else "bgr24", "-f", "rawvideo", "-"]
    ch = 1 if gray else 3
    nbytes = W * H * ch
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=nbytes * 2)
    i = 0
    try:
        while True:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            arr = np.frombuffer(buf, np.uint8).reshape((H, W) if gray else (H, W, 3))
            yield start + i / fps, arr
            i += 1
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()
        if proc.returncode not in (0, -9, None) and i == 0:
            err = proc.stderr.read().decode(errors="ignore")
            if hwaccel and sys.platform == "darwin":
                # retry without hardware decoding
                yield from frames(info, fps, width, start, end, gray, crop, keyframes_only, hwaccel=False)
                return
            raise RuntimeError(f"ffmpeg failed: {err.strip()[:500]}")


def grab(info: VideoInfo, t: float, width: int | None = None) -> np.ndarray | None:
    for _, f in frames(info, fps=1, width=width, start=t, end=t + 1.0):
        return f
    return None

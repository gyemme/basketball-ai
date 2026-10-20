"""Game / pipeline configuration (game.yaml)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "game": {"title": None, "date": None, "venue": None},
    # Teams in the same order as the ScoreSync overlay: first = LEFT score, second = RIGHT score.
    # jersey: main shirt colour (hex). Leave null to detect automatically.
    # roster: {"10": "Caleb", "3": "Aarav"} – names optional; numbers help jersey reading.
    "teams": [
        {"name": "Left team", "jersey": None, "roster": {}},
        {"name": "Right team", "jersey": None, "roster": {}},
    ],
    "court": {"length": 84.0, "width": 50.0, "three_pt": 19.75},
    "analysis": {
        "fps": 10,                 # frames analysed per second of video
        "width": 1920,             # detection resolution (frames are scaled to this width)
        "detector": "yolo",        # yolo | rfdetr
        "model": None,             # e.g. yolo11m.pt / yolo11l.pt ; rfdetr: base | medium | large
        "imgsz": 1280,             # detector input size (bigger finds the ball better, slower)
        "device": "auto",          # auto | mps | cpu | cuda
        "person_conf": 0.35,
        "ball_conf": 0.15,
        "ocr": "auto",             # auto | vision (macOS) | rapidocr | none
        "ocr_every": 1.0,          # seconds between jersey-number reads per player
    },
    "overlay": {"enabled": True, "layout": None},   # ScoreSync score overlay
    "shots": {
        "make_window_s": 25,       # scorer may update the overlay up to this long after a basket
    },
    "periods": None,               # e.g. [0, 1250] = 2nd period starts at 1250 s. null = auto
}


def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load(path: str | Path | None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path and Path(path).exists():
        with open(path) as f:
            cfg = _merge(cfg, yaml.safe_load(f) or {})
    # YAML turns `date: 2026-10-17` into a date object; keep everything in `game` as text
    for k, v in list((cfg.get("game") or {}).items()):
        if v is not None and not isinstance(v, str):
            cfg["game"][k] = str(v)
    # normalise rosters to {str(number): name}
    for t in cfg["teams"]:
        r = t.get("roster") or {}
        if isinstance(r, list):
            r = {str(x): "" for x in r}
        t["roster"] = {str(k): (v or "") for k, v in r.items()}
    return cfg


TEMPLATE = """\
# hoopstats game config. Edit, then re-run:  hoopstats analyze <video> --config this-file
game:
  title: {title}
  date: {date}
  venue:

# Order matters: the first team is the one on the LEFT of the XbotGo score overlay.
teams:
  - name: {left}
    jersey:            # e.g. "#8ccfee" (light blue). Leave empty = detect automatically
    roster:            # jersey number: name   (names optional)
      # "10": Caleb
  - name: {right}
    jersey:
    roster:

court:
  length: 84          # feet. High school 84, NBA/college 94, many youth gyms 74
  width: 50
  three_pt: 19.75     # arc radius in feet; set to null if your league has no 3-point line

analysis:
  fps: 10
  detector: yolo      # yolo (Ultralytics) or rfdetr
  ocr: auto           # auto uses Apple Vision on macOS
"""


def write_template(path: Path, title="", date="", left="Left team", right="Right team") -> Path:
    path.write_text(TEMPLATE.format(title=title or "", date=date or "", left=left, right=right))
    return path

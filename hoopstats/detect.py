"""Person + ball detection with open-source models.

Backends:
  yolo   - Ultralytics YOLO11 (COCO: person=0, sports ball=32). Weights download automatically.
  rfdetr - Roboflow RF-DETR (Apache-2.0, COCO). pip install rfdetr

Both return Detections: persons Nx5 [x1,y1,x2,y2,conf], balls Mx5, in input-frame pixels.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np


@dataclass
class Detections:
    persons: np.ndarray
    balls: np.ndarray


def pick_device(pref: str = "auto") -> str:
    if pref and pref != "auto":
        return pref
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class YoloDetector:
    PERSON, BALL = 0, 32

    def __init__(self, model: str | None = None, imgsz: int = 1280, device: str = "auto",
                 person_conf: float = 0.35, ball_conf: float = 0.15):
        from ultralytics import YOLO
        self.model = YOLO(model or "yolo11m.pt")
        self.imgsz = imgsz
        self.device = pick_device(device)
        self.pc, self.bc = person_conf, ball_conf

    def __call__(self, frame_bgr: np.ndarray) -> Detections:
        r = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=min(self.pc, self.bc),
                               classes=[self.PERSON, self.BALL], device=self.device, verbose=False)[0]
        b = r.boxes
        if b is None or len(b) == 0:
            return Detections(np.zeros((0, 5)), np.zeros((0, 5)))
        xyxy = b.xyxy.cpu().numpy()
        conf = b.conf.cpu().numpy()
        cls = b.cls.cpu().numpy().astype(int)
        rows = np.c_[xyxy, conf]
        p = rows[(cls == self.PERSON) & (conf >= self.pc)]
        ball = rows[(cls == self.BALL) & (conf >= self.bc)]
        return Detections(p, ball)


class RFDetrDetector:
    # rfdetr uses COCO category ids (1-based, with gaps): person=1, sports ball=37
    PERSON, BALL = 1, 37

    def __init__(self, model: str | None = None, imgsz: int = 1280, device: str = "auto",
                 person_conf: float = 0.35, ball_conf: float = 0.15):
        import rfdetr
        name = (model or "medium").lower()
        cls = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium",
               "base": "RFDETRBase", "large": "RFDETRLarge"}.get(name, "RFDETRMedium")
        Model = getattr(rfdetr, cls, None) or getattr(rfdetr, "RFDETRBase")
        self.model = Model()
        try:
            self.model.optimize_for_inference()
        except Exception:
            pass
        self.pc, self.bc = person_conf, ball_conf

    def __call__(self, frame_bgr: np.ndarray) -> Detections:
        from PIL import Image
        img = Image.fromarray(frame_bgr[..., ::-1])
        d = self.model.predict(img, threshold=min(self.pc, self.bc))
        if d is None or len(d) == 0:
            return Detections(np.zeros((0, 5)), np.zeros((0, 5)))
        rows = np.c_[d.xyxy, d.confidence]
        cls = np.asarray(d.class_id).astype(int)
        p = rows[(cls == self.PERSON) & (rows[:, 4] >= self.pc)]
        b = rows[(cls == self.BALL) & (rows[:, 4] >= self.bc)]
        return Detections(p, b)


def make_detector(cfg: dict):
    a = cfg["analysis"]
    kind = (a.get("detector") or "yolo").lower()
    kw = dict(model=a.get("model"), imgsz=a.get("imgsz", 1280), device=a.get("device", "auto"),
              person_conf=a.get("person_conf", 0.35), ball_conf=a.get("ball_conf", 0.15))
    try:
        if kind == "rfdetr":
            return RFDetrDetector(**kw)
        return YoloDetector(**kw)
    except ImportError as e:
        alt = "yolo" if kind == "rfdetr" else "rfdetr"
        print(f"[detect] {kind} backend not installed ({e}); trying {alt}", file=sys.stderr)
        try:
            return RFDetrDetector(**kw) if alt == "rfdetr" else YoloDetector(**kw)
        except ImportError:
            sys.exit("No detector installed. Run:  pip install -e '.[yolo]'   (or '.[rfdetr]')")

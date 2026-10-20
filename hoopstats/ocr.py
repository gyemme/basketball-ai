"""Jersey-number reading. Uses Apple Vision on macOS (built in, no model download) or
RapidOCR (PaddleOCR models in ONNX) elsewhere. Returns [(digits, confidence)]."""
from __future__ import annotations

import re
import sys

import numpy as np

DIGITS = re.compile(r"^\d{1,2}$")


class NullOCR:
    name = "none"

    def __call__(self, img_bgr: np.ndarray) -> list[tuple[str, float]]:
        return []


class VisionOCR:
    name = "vision"

    def __init__(self):
        from ocrmac import ocrmac  # noqa: F401
        self._ocrmac = ocrmac

    def __call__(self, img_bgr):
        from PIL import Image
        im = Image.fromarray(img_bgr[..., ::-1])
        try:
            res = self._ocrmac.OCR(im, recognition_level="accurate", language_preference=["en-US"]).recognize()
        except Exception:
            return []
        out = []
        for r in res:
            text, conf = str(r[0]), float(r[1])
            text = text.replace(" ", "").replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")
            if DIGITS.match(text):
                out.append((text, conf))
        return out


class RapidOCR:
    name = "rapidocr"

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR as _R
        self.engine = _R()

    def __call__(self, img_bgr):
        try:
            res, _ = self.engine(img_bgr)
        except Exception:
            return []
        out = []
        for r in res or []:
            text, conf = str(r[1]).replace(" ", ""), float(r[2])
            text = text.replace("O", "0").replace("o", "0")
            if DIGITS.match(text):
                out.append((text, conf))
        return out


def make_ocr(kind: str = "auto"):
    kind = (kind or "auto").lower()
    order = {"auto": (["vision"] if sys.platform == "darwin" else []) + ["rapidocr"],
             "vision": ["vision"], "rapidocr": ["rapidocr"], "none": []}[kind]
    for k in order:
        try:
            return VisionOCR() if k == "vision" else RapidOCR()
        except Exception as e:  # not installed
            print(f"[ocr] {k} unavailable: {e}", file=sys.stderr)
    return NullOCR()


def prep(crop: np.ndarray) -> np.ndarray:
    """Upscale small crops so OCR engines see digits at a readable size."""
    import cv2
    h = crop.shape[0]
    if h < 160:
        f = 160 / max(h, 1)
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return crop

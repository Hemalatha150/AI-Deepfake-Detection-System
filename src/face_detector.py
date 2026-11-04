"""Face detection and cropping.

Backends are tried in order MTCNN -> dlib -> OpenCV Haar cascade, so the
system keeps working even when an optional library is not installed.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

import config

log = logging.getLogger(__name__)


class FaceDetector:
    BACKENDS = ("mtcnn", "dlib", "haar")

    def __init__(self, backend: str = "auto", min_confidence: float = config.MIN_FACE_CONFIDENCE):
        self.min_confidence = min_confidence
        # A requested backend is tried first; the others remain as fallbacks.
        order = (backend, *(b for b in self.BACKENDS if b != backend)) \
            if backend != "auto" else self.BACKENDS
        last_error = None
        for name in order:
            try:
                getattr(self, f"_init_{name}")()
                self.backend = name
                break
            except Exception as exc:  # library missing or failed to load
                last_error = exc
                log.warning("Face detector backend '%s' unavailable: %s", name, exc)
        else:
            raise RuntimeError(f"No face detector backend could be loaded: {last_error}")

    @staticmethod
    def available_backends() -> list[str]:
        """Backends whose library is installed (cheap check, no models loaded)."""
        import importlib.util
        found = [m for m in ("mtcnn", "dlib") if importlib.util.find_spec(m)]
        if hasattr(cv2, "CascadeClassifier"):  # removed from OpenCV 5
            found.append("haar")
        return found

    # ------------------------------------------------------------------ init
    def _init_mtcnn(self):
        from mtcnn import MTCNN
        self._mtcnn = MTCNN()

    def _init_dlib(self):
        import dlib
        self._dlib = dlib.get_frontal_face_detector()

    def _init_haar(self):
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(path)
        if self._haar.empty():
            raise RuntimeError(f"Could not load Haar cascade from {path}")

    # ------------------------------------------------------------- detection
    def detect(self, rgb: np.ndarray) -> list[dict]:
        """Detect faces in an RGB uint8 image.

        Returns a list of {"box": (x, y, w, h), "confidence": float,
        "keypoints": {name: (x, y)}} in original-image coordinates,
        sorted by face area (largest first).
        """
        h, w = rgb.shape[:2]
        scale = min(1.0, config.DETECT_MAX_SIDE / max(h, w))
        small = cv2.resize(rgb, (int(w * scale), int(h * scale))) if scale < 1.0 else rgb
        small = np.ascontiguousarray(small)

        faces = getattr(self, f"_detect_{self.backend}")(small)

        out = []
        for f in faces:
            x, y, bw, bh = (v / scale for v in f["box"])
            if bw < 20 or bh < 20:
                continue
            kps = {k: (px / scale, py / scale) for k, (px, py) in f.get("keypoints", {}).items()}
            out.append({"box": (int(x), int(y), int(bw), int(bh)),
                        "confidence": float(f["confidence"]), "keypoints": kps})
        out.sort(key=lambda f: f["box"][2] * f["box"][3], reverse=True)
        return out

    def _detect_mtcnn(self, rgb):
        try:
            results = self._mtcnn.detect_faces(rgb)
        except ValueError:
            # mtcnn 1.0 raises instead of returning [] when no candidate survives a stage.
            return []
        return [r for r in results if r["confidence"] >= self.min_confidence]

    def _detect_dlib(self, rgb):
        rects, scores, _ = self._dlib.run(rgb, 1, 0)
        return [{"box": (r.left(), r.top(), r.width(), r.height()),
                 "confidence": min(1.0, 0.5 + s / 4)} for r, s in zip(rects, scores)]

    def _detect_haar(self, rgb):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        boxes = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(40, 40))
        return [{"box": tuple(int(v) for v in b), "confidence": 1.0} for b in boxes]

    # ------------------------------------------------------------- cropping
    @staticmethod
    def crop_face(rgb: np.ndarray, box, margin: float = config.FACE_MARGIN) -> np.ndarray:
        """Square crop around `box`, enlarged by `margin`, clamped to the image."""
        h, w = rgb.shape[:2]
        x, y, bw, bh = box
        cx, cy = x + bw / 2, y + bh / 2
        side = max(bw, bh) * (1 + margin)
        x1, y1 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        x2, y2 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        return rgb[y1:y2, x1:x2]

    def extract_faces(self, rgb: np.ndarray, max_faces: int = 5,
                      margin: float = config.FACE_MARGIN) -> list[dict]:
        """Detect faces and return them with their crops."""
        faces = self.detect(rgb)[:max_faces]
        for f in faces:
            f["crop"] = self.crop_face(rgb, f["box"], margin)
        return [f for f in faces if f["crop"].size > 0]


def draw_faces(rgb: np.ndarray, faces: list[dict], labels: list[str] | None = None,
               colors: list[tuple] | None = None) -> np.ndarray:
    """Draw bounding boxes, landmarks and optional labels on a copy of the image."""
    img = rgb.copy()
    thick = max(2, img.shape[1] // 300)
    for i, f in enumerate(faces):
        x, y, w, h = f["box"]
        color = colors[i] if colors else (0, 200, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thick)
        for px, py in f.get("keypoints", {}).values():
            cv2.circle(img, (int(px), int(py)), thick + 1, (255, 255, 0), -1)
        if labels:
            scale = max(0.5, img.shape[1] / 1200)
            (tw, th), _ = cv2.getTextSize(labels[i], cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            ty = max(th + 4, y - 4)
            cv2.rectangle(img, (x, ty - th - 4), (x + tw + 4, ty + 2), color, -1)
            cv2.putText(img, labels[i], (x + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        scale, (255, 255, 255), thick)
    return img

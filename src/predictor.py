"""High-level inference API used by the web app and the CLI."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import config
from src.face_detector import FaceDetector
from src.gradcam import GradCAM, overlay
from src.model import load_model
from src.video import sample_frames, video_info


def verdict(fake_prob: float, threshold: float) -> dict:
    """Label + confidence. Confidence is 50% at the threshold and 100% at P=0 or P=1,
    so it stays meaningful when the calibrated threshold is not 0.5."""
    is_fake = fake_prob >= threshold
    if is_fake:
        confidence = 0.5 + 0.5 * (fake_prob - threshold) / max(1 - threshold, 1e-6)
    else:
        confidence = 0.5 + 0.5 * (threshold - fake_prob) / max(threshold, 1e-6)
    return {
        "label": config.DISPLAY_NAMES[int(is_fake)],
        "is_fake": bool(is_fake),
        "fake_probability": float(fake_prob),
        "confidence": float(confidence),
    }


def calibrate(p, threshold: float):
    """Map a model's raw P(fake) onto a common scale on which its own calibrated
    threshold sits at 0.5, so scores from different models can be compared."""
    p = np.asarray(p, dtype=np.float64)
    t = float(np.clip(threshold, 1e-6, 1 - 1e-6))
    return np.where(p < t, 0.5 * p / t, 0.5 + 0.5 * (p - t) / (1 - t))


def square_crops(rgb: np.ndarray) -> list[np.ndarray]:
    """Start, centre and end square crops along the image's long side."""
    h, w = rgb.shape[:2]
    s = min(h, w)
    if h >= w:
        return [rgb[y:y + s] for y in (0, (h - s) // 2, h - s)]
    return [rgb[:, x:x + s] for x in (0, (w - s) // 2, w - s)]


class DeepfakeDetector:
    """Two complementary detectors:

    * face model  - classifies each detected face crop (GAN / diffusion / face-swap);
    * scene model - classifies the whole image, catching AI-generated pictures whose
      faces are too small to analyse, or that contain no face at all.

    Scores are calibrated so each model's own threshold maps to 0.5; the media is
    flagged as a deepfake if either detector is confident.
    """

    def __init__(self, model_path=config.MODEL_PATH, detector_backend: str = "auto",
                 scene_model_path=config.SCENE_MODEL_PATH):
        self.model, self.meta = load_model(model_path)
        self.img_size = int(self.meta["img_size"])
        self.face_threshold = float(self.meta.get("threshold", config.DEFAULT_THRESHOLD))
        self.threshold = 0.5          # decisions are made on calibrated scores
        self.scene_model = self.scene_meta = None
        if scene_model_path and Path(scene_model_path).exists():
            self.scene_model, self.scene_meta = load_model(scene_model_path)
        self.face_detector = FaceDetector(detector_backend)
        self._gradcam = None

    # -------------------------------------------------------------- helpers
    def _prepare(self, crop: np.ndarray) -> np.ndarray:
        return cv2.resize(crop, (self.img_size, self.img_size),
                          interpolation=cv2.INTER_AREA).astype(np.float32)

    @staticmethod
    def _run(model, size: int, images: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
        if not images:
            return np.zeros(0)
        batch = np.stack([cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
                          for im in images]).astype(np.float32)
        return np.concatenate([model(batch[i:i + batch_size], training=False).numpy()[:, 0]
                               for i in range(0, len(batch), batch_size)])

    def predict_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        """Raw face-model P(fake) for face crops."""
        return self._run(self.model, self.img_size, crops)

    def face_scores(self, crops: list[np.ndarray]) -> np.ndarray:
        return calibrate(self.predict_crops(crops), self.face_threshold)

    def scene_scores(self, images: list[np.ndarray]) -> np.ndarray | None:
        if self.scene_model is None:
            return None
        # The scene model is trained on centre-square crops; average three square
        # crops along the long side so nothing at the edges is ignored.
        crops = [c for im in images for c in square_crops(im)]
        raw = self._run(self.scene_model, int(self.scene_meta["img_size"]), crops)
        raw = raw.reshape(len(images), 3).mean(axis=1)
        return calibrate(raw, float(self.scene_meta["threshold"]))

    def explain(self, crop: np.ndarray) -> np.ndarray:
        """Grad-CAM overlay for one face crop."""
        if self._gradcam is None:
            self._gradcam = GradCAM(self.model, self.meta["gradcam_layer"])
        resized = self._prepare(crop)
        return overlay(resized, self._gradcam.heatmap(resized))

    # ---------------------------------------------------------------- image
    def predict_image(self, rgb: np.ndarray, threshold: float | None = None,
                      max_faces: int = 5, explain: bool = False) -> dict:
        threshold = self.threshold if threshold is None else threshold
        faces = self.face_detector.extract_faces(rgb, max_faces=max_faces)
        scene = self.scene_scores([rgb])
        scene_score = None if scene is None else float(scene[0])

        used_full_image = not faces
        if used_full_image:
            faces = [{"box": (0, 0, rgb.shape[1], rgb.shape[0]), "confidence": 0.0,
                      "keypoints": {}, "crop": rgb}]
        scores = self.face_scores([f["crop"] for f in faces])
        for f, s in zip(faces, scores):
            f.update(verdict(s, threshold))
            if explain:
                f["gradcam"] = self.explain(f["crop"])

        # Manipulated if any face is, or if the image as a whole is AI-generated.
        # Without a face, the face model is only a weak fallback.
        face_score = None if used_full_image and scene_score is not None else float(scores.max())
        overall = max(s for s in (face_score, scene_score) if s is not None)
        result = verdict(overall, threshold)
        result.update(faces=faces, used_full_image=used_full_image, threshold=threshold,
                      face_score=face_score, scene_score=scene_score,
                      detector=self.face_detector.backend)
        return result

    # ---------------------------------------------------------------- video
    def predict_video(self, path, num_frames: int = config.FRAMES_PER_VIDEO,
                      threshold: float | None = None, progress_cb=None) -> dict:
        threshold = self.threshold if threshold is None else threshold
        info = video_info(path)

        frames, crops, owners = [], [], []
        for n, (idx, ts, rgb) in enumerate(sample_frames(path, num_frames)):
            faces = self.face_detector.extract_faces(rgb, max_faces=config.MAX_FACES_PER_FRAME)
            # Drop small secondary detections (often background false positives).
            if faces:
                main_area = faces[0]["box"][2] * faces[0]["box"][3]
                faces = [f for f in faces if f["box"][2] * f["box"][3] >= 0.25 * main_area]
            frames.append({"frame": idx, "time": ts, "n_faces": len(faces),
                           "faces": faces, "rgb": rgb})
            for f in faces:
                crops.append(f["crop"])
                owners.append(len(frames) - 1)
            if progress_cb:
                progress_cb((n + 1) / num_frames)
        if not frames:
            raise ValueError("No frames could be read from the video.")

        used_full_frames = not crops
        if used_full_frames:  # no faces anywhere: classify whole frames
            crops = [f["rgb"] for f in frames]
            owners = list(range(len(frames)))

        probs = self.face_scores(crops)
        frame_probs = {}
        for owner, p in zip(owners, probs):
            frame_probs[owner] = max(frame_probs.get(owner, 0.0), float(p))
        face_median = float(np.median(list(frame_probs.values())))
        scene = self.scene_scores([f["rgb"] for f in frames])
        scene_median = None if scene is None else float(np.median(scene))
        if scene is not None:
            for i, s in enumerate(scene):
                frame_probs[i] = max(frame_probs.get(i, 0.0), float(s))

        timeline, samples = [], []
        for i, fr in enumerate(frames):
            p = frame_probs.get(i)
            timeline.append({"frame": fr["frame"], "time": round(fr["time"], 2),
                             "faces": fr["n_faces"], "fake_probability": p})
        for crop, owner, p in zip(crops, owners, probs):
            samples.append({"crop": crop, "frame": frames[owner]["frame"],
                            "fake_probability": float(p)})

        scored = np.array([p for p in frame_probs.values()])
        # Median rather than mean: a few motion-blurred or badly lit frames
        # should not decide the verdict for the whole video.
        result = verdict(float(np.median(scored)), threshold)
        result.update(
            threshold=threshold,
            face_score=face_median,
            scene_score=scene_median,
            video=info,
            frames_analyzed=len(frames),
            frames_with_faces=sum(1 for f in frames if f["n_faces"]),
            used_full_frames=used_full_frames,
            fake_frame_ratio=float((scored >= threshold).mean()),
            # Temporal consistency: real videos tend to give stable per-frame
            # scores; flicker between frames is itself a manipulation cue.
            temporal_std=float(scored.std()),
            temporal_jitter=float(np.abs(np.diff(scored)).mean()) if len(scored) > 1 else 0.0,
            timeline=timeline,
            samples=samples,
            detector=self.face_detector.backend,
        )
        return result


def load_rgb(path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)   # handles non-ASCII paths on Windows
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def is_video(path) -> bool:
    return Path(path).suffix.lower() in config.VIDEO_EXTS

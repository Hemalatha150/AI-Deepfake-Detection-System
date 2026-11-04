"""Video frame sampling."""
from __future__ import annotations

import cv2
import numpy as np


def video_info(path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    info = {
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    info["duration"] = info["frames"] / info["fps"] if info["fps"] else 0.0
    return info


def sample_frames(path, num_frames: int = 20):
    """Yield (frame_index, timestamp_sec, rgb_frame) for evenly spaced frames."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    try:
        if total <= 0:
            # Frame count unknown (some streams/containers): read sequentially.
            idx = 0
            while idx < num_frames * 10:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % 10 == 0:
                    yield idx, idx / fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                idx += 1
            return
        indices = np.unique(np.linspace(0, total - 1, min(num_frames, total)).astype(int))
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                yield int(idx), idx / fps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()

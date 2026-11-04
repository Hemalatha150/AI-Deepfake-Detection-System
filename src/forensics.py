"""Classical image-forensics visualisations shown alongside the CNN verdict.

These are supporting evidence for the user, not part of the decision:
  * Error Level Analysis (ELA) - regions edited or synthesised separately
    often re-compress differently from the rest of the image.
  * Frequency spectrum - GAN/diffusion upsampling layers frequently leave
    periodic grid-like peaks in the 2-D Fourier spectrum.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, ImageChops


def error_level_analysis(rgb: np.ndarray, quality: int = 90) -> np.ndarray:
    original = Image.fromarray(rgb.astype(np.uint8)).convert("RGB")
    buf = io.BytesIO()
    original.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf)
    diff = np.asarray(ImageChops.difference(original, recompressed)).astype(np.float32)
    # Scale by a high percentile rather than the max so a few outlier pixels
    # don't leave the rest of the map black.
    ref = np.percentile(diff, 99.5) or 1.0
    return np.clip(diff * (255.0 / ref), 0, 255).astype(np.uint8)


def frequency_spectrum(rgb: np.ndarray, size: int = 256) -> np.ndarray:
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, (size, size)).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(spectrum))
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    colored = cv2.applyColorMap(magnitude, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

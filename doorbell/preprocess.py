"""Face preprocessing shared by training and recognition."""
from __future__ import annotations

import cv2
import numpy as np


def to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def normalize_face(gray: np.ndarray, size: int = 100, equalize: bool = True) -> np.ndarray:
    """Resize to a fixed square; optionally equalize the histogram.

    `size` is the main accuracy-vs-latency knob (smaller = faster on a Pi).
    `equalize` is the main lighting-robustness knob.
    """
    face = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.equalizeHist(face) if equalize else face


def adjust_gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
    """Simulate lighting: gamma > 1 darkens the image, gamma < 1 brightens it."""
    table = ((np.arange(256) / 255.0) ** gamma * 255).astype(np.uint8)
    return cv2.LUT(gray, table)

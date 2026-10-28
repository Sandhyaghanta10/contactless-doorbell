"""Face detection behind a tiny interface, so tests and benchmarks can swap it out."""
from __future__ import annotations

from typing import List, Protocol, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]  # x, y, w, h


class FaceDetector(Protocol):
    def detect(self, gray: np.ndarray) -> List[Box]: ...


class HaarFaceDetector:
    """OpenCV Haar cascade: cheap enough for a Raspberry Pi."""

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5,
                 min_size: Tuple[int, int] = (60, 60)) -> None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(path)
        if self._cascade.empty():
            raise FileNotFoundError(
                f"Haar cascade not found at {path}. Install opencv-python(-headless) 4.x.")
        self._params = dict(scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size)

    def detect(self, gray: np.ndarray) -> List[Box]:
        return [tuple(int(v) for v in box)
                for box in self._cascade.detectMultiScale(gray, **self._params)]

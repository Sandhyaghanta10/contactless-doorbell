"""LBPH face recognition (classic, tiny, trains in milliseconds - good for a Pi)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

import cv2
import numpy as np

from .preprocess import normalize_face


@dataclass(frozen=True)
class Match:
    name: Optional[str]  # None = unknown visitor
    distance: float      # LBPH distance: LOWER means more similar


class FaceRecognizer:
    """`threshold` is the largest distance still accepted as a registered visitor.
    Use float('inf') to always return the nearest identity (closed-set evaluation)."""

    def __init__(self, size: int = 100, equalize: bool = True, threshold: float = 80.0) -> None:
        self.size, self.equalize, self.threshold = size, equalize, threshold
        self._names: List[str] = []
        self._model = None

    def train(self, faces_by_name: Mapping[str, Sequence[np.ndarray]]) -> None:
        if not faces_by_name:
            raise ValueError("Need at least one registered visitor")
        self._names = sorted(faces_by_name)
        images, labels = [], []
        for label, name in enumerate(self._names):
            for face in faces_by_name[name]:
                images.append(normalize_face(face, self.size, self.equalize))
                labels.append(label)
        self._model = cv2.face.LBPHFaceRecognizer_create()
        self._model.train(images, np.array(labels, dtype=np.int32))

    def identify(self, gray_face: np.ndarray) -> Match:
        if self._model is None:
            raise RuntimeError("Call train() first")
        label, distance = self._model.predict(normalize_face(gray_face, self.size, self.equalize))
        name = self._names[label] if distance <= self.threshold else None
        return Match(name, float(distance))

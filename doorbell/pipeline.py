"""The doorbell brain: frame in -> detect -> recognize -> (rate-limited) notify."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .detector import Box, FaceDetector
from .notifier import Notifier
from .preprocess import to_gray
from .recognizer import FaceRecognizer

UNKNOWN = "unknown visitor"


@dataclass(frozen=True)
class Visit:
    name: Optional[str]
    distance: float
    box: Box


class Doorbell:
    def __init__(self, detector: FaceDetector, recognizer: FaceRecognizer, notifier: Notifier,
                 cooldown_s: float = 30.0, clock: Callable[[], float] = time.monotonic) -> None:
        self._detector, self._recognizer, self._notifier = detector, recognizer, notifier
        self._cooldown, self._clock = cooldown_s, clock
        self._last_alert: Dict[str, float] = {}

    def process(self, frame: np.ndarray) -> List[Visit]:
        gray = to_gray(frame)
        visits = []
        for box in self._detector.detect(gray):
            x, y, w, h = box
            match = self._recognizer.identify(gray[y:y + h, x:x + w])
            visit = Visit(match.name, match.distance, box)
            visits.append(visit)
            self._alert_once(visit)
        return visits

    def _alert_once(self, visit: Visit) -> None:
        """A visitor standing at the door for 30 s must not cause 900 emails."""
        key = visit.name or UNKNOWN
        now = self._clock()
        last = self._last_alert.get(key)
        if last is not None and now - last < self._cooldown:
            return
        self._last_alert[key] = now
        who = f"Registered visitor: {visit.name}" if visit.name else "Unknown visitor at the door"
        self._notifier.notify(f"Doorbell: {key}", f"{who} (match distance {visit.distance:.1f})")

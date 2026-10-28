"""Synthetic 'faces': one distinct smooth random texture per identity, plus per-photo noise.

Good enough to test the plumbing deterministically. Real accuracy numbers come from
running doorbell.benchmark on a real dataset.
"""
import cv2
import numpy as np
import pytest


def make_template(identity_seed: int, size: int = 100) -> np.ndarray:
    rng = np.random.default_rng(1000 + identity_seed)
    field = cv2.GaussianBlur(rng.integers(0, 256, (size, size), dtype=np.uint8), (0, 0), 4)
    return cv2.normalize(field, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def make_photo(template: np.ndarray, photo_seed: int) -> np.ndarray:
    rng = np.random.default_rng(photo_seed)
    noisy = template.astype(np.float32) + rng.normal(0, 10, template.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


@pytest.fixture
def make_gallery():
    def build(identities, photos=8, seed_offset=0):
        return {f"person{i}": [make_photo(make_template(i), 10 * i + p + seed_offset) for p in range(photos)]
                for i in identities}
    return build

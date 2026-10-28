"""Load a folder-per-person image dataset (e.g. the AT&T / Olivetti 'Database of Faces')."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".pgm", ".bmp"}
Gallery = Dict[str, List[np.ndarray]]


def load_gallery(root: Path) -> Gallery:
    """root/<person>/<image files> -> {person: [grayscale images]}."""
    gallery: Gallery = {}
    for person_dir in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        images = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                  for f in sorted(person_dir.iterdir()) if f.suffix.lower() in IMAGE_SUFFIXES]
        images = [img for img in images if img is not None]
        if images:
            gallery[person_dir.name] = images
    if not gallery:
        raise ValueError(f"No person folders with images found under {root}")
    return gallery


def split_gallery(gallery: Gallery, train_per_person: int, seed: int = 0) -> Tuple[Gallery, Gallery]:
    """Deterministic per-person train/test split (same seed = same split)."""
    rng = random.Random(seed)
    train: Gallery = {}
    test: Gallery = {}
    for name in sorted(gallery):
        images = list(gallery[name])
        if len(images) <= train_per_person:
            raise ValueError(f"{name} has {len(images)} images; need more than {train_per_person}")
        rng.shuffle(images)
        train[name], test[name] = images[:train_per_person], images[train_per_person:]
    return train, test

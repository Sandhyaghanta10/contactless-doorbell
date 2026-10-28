"""Accuracy-vs-latency-vs-lighting benchmark for the recognizer.

    python -m doorbell.benchmark --dataset dataset/att_faces --device-label "MacBook M-series"

Experiments (all numbers are measured by this script on YOUR data/hardware):
  1. Closed-set accuracy + per-face latency over face size x histogram-equalization x lighting (gamma).
  2. Open-set threshold sweep: how often registered visitors are accepted (TAR) vs strangers
     wrongly accepted (FAR) as the 'unknown visitor' threshold moves.
"""
from __future__ import annotations

import argparse
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from .dataset import Gallery, load_gallery, split_gallery
from .preprocess import adjust_gamma
from .recognizer import FaceRecognizer, Match

SIZES = (32, 64, 100)
GAMMAS = (0.4, 1.0, 2.5)  # <1 bright, 1 normal, >1 dark
THRESHOLDS = tuple(range(20, 141, 10))
Known = List[Tuple[str, Match]]


@dataclass(frozen=True)
class Result:
    size: int
    equalize: bool
    gamma: float
    accuracy: float
    p50_ms: float
    p95_ms: float


def _trained(train: Gallery, size: int, equalize: bool) -> FaceRecognizer:
    recognizer = FaceRecognizer(size, equalize, threshold=float("inf"))  # always nearest identity
    recognizer.train(train)
    return recognizer


def evaluate(train: Gallery, test: Gallery, size: int, equalize: bool, gamma: float) -> Result:
    recognizer = _trained(train, size, equalize)
    correct, total, latencies = 0, 0, []
    for name, faces in test.items():
        for face in faces:
            probe = adjust_gamma(face, gamma)
            start = time.perf_counter()
            match = recognizer.identify(probe)
            latencies.append((time.perf_counter() - start) * 1000)
            correct += match.name == name
            total += 1
    latencies.sort()
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))]
    return Result(size, equalize, gamma, correct / total, statistics.median(latencies), p95)


def score_open_set(train: Gallery, known_test: Gallery, strangers: Gallery,
                   size: int, equalize: bool) -> Tuple[Known, List[float]]:
    recognizer = _trained(train, size, equalize)
    known = [(name, recognizer.identify(face)) for name, faces in known_test.items() for face in faces]
    stranger_distances = [recognizer.identify(face).distance
                          for faces in strangers.values() for face in faces]
    return known, stranger_distances


def threshold_sweep(known: Known, stranger_distances: Sequence[float],
                    thresholds: Sequence[float]) -> List[Tuple[float, float, float]]:
    """(threshold, true-accept rate, false-accept rate) per threshold."""
    rows = []
    for t in thresholds:
        tar = sum(m.name == true and m.distance <= t for true, m in known) / len(known)
        far = sum(d <= t for d in stranger_distances) / len(stranger_distances)
        rows.append((t, tar, far))
    return rows


def format_report(results: Sequence[Result], sweep: Sequence[Tuple[float, float, float]],
                  device: str, summary: str) -> str:
    lines = [f"Device: **{device}**. {summary}\n",
             "### 1. Accuracy and latency (closed-set)\n",
             "| Face size | Hist. equalization | Lighting (gamma) | Accuracy | p50 ms | p95 ms |",
             "|---:|:---:|---:|---:|---:|---:|"]
    lines += [f"| {r.size} | {'on' if r.equalize else 'off'} | {r.gamma} | {r.accuracy:.1%} "
              f"| {r.p50_ms:.2f} | {r.p95_ms:.2f} |" for r in results]
    lines += ["", "### 2. Unknown-visitor threshold sweep (size 100, equalization on, normal light)\n",
              "| Threshold | Registered accepted (TAR) | Strangers accepted (FAR) |", "|---:|---:|---:|"]
    lines += [f"| {t:.0f} | {tar:.1%} | {far:.1%} |" for t, tar, far in sweep]
    return "\n".join(lines) + "\n"


def run(dataset: Path, train_per_person: int, strangers: int, seed: int, device: str) -> str:
    gallery = load_gallery(dataset)
    names = sorted(gallery)
    random.Random(seed).shuffle(names)
    stranger_names, enrolled_names = names[:strangers], names[strangers:]
    enrolled: Gallery = {n: gallery[n] for n in enrolled_names}
    stranger_gallery: Gallery = {n: gallery[n] for n in stranger_names}
    train, test = split_gallery(enrolled, train_per_person, seed)

    results = [evaluate(train, test, size, equalize, gamma)
               for size in SIZES for equalize in (False, True) for gamma in GAMMAS]
    known, stranger_distances = score_open_set(train, test, stranger_gallery, 100, True)
    sweep = threshold_sweep(known, stranger_distances, THRESHOLDS)
    summary = (f"{len(enrolled)} enrolled people ({train_per_person} training photos each), "
               f"{len(stranger_gallery)} held-out strangers, seed {seed}.")
    return format_report(results, sweep, device, summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--train-per-person", type=int, default=5)
    parser.add_argument("--strangers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-label", default="unspecified device")
    parser.add_argument("--out", type=Path, default=Path("results/results.md"))
    args = parser.parse_args()
    report = run(args.dataset, args.train_per_person, args.strangers, args.seed, args.device_label)
    print(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)


if __name__ == "__main__":
    main()

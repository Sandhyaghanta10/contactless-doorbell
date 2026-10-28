"""Camera loop + CLI.  python -m doorbell run --gallery faces/ [--dry-run]"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from .dataset import load_gallery
from .detector import HaarFaceDetector
from .notifier import EmailNotifier, PrintNotifier
from .pipeline import Doorbell
from .recognizer import FaceRecognizer


def run_camera(doorbell: Doorbell, camera_index: int, max_fps: float) -> None:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise SystemExit(f"Cannot open camera {camera_index}")
    delay = 1.0 / max_fps  # cap the frame rate: a Pi has no CPU to burn
    try:
        while True:
            ok, frame = capture.read()
            if ok:
                for visit in doorbell.process(frame):
                    print(f"saw {visit.name or 'UNKNOWN'} (distance {visit.distance:.1f})")
            time.sleep(delay)
    except KeyboardInterrupt:
        print("bye")
    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Contactless doorbell")
    parser.add_argument("command", choices=["run"])
    parser.add_argument("--gallery", type=Path, required=True,
                        help="folder with one sub-folder of photos per registered visitor")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--size", type=int, default=64, help="face size; smaller = faster")
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument("--cooldown", type=float, default=30.0, help="seconds between repeat alerts")
    parser.add_argument("--max-fps", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true", help="print alerts instead of emailing")
    args = parser.parse_args()

    recognizer = FaceRecognizer(args.size, threshold=args.threshold)
    recognizer.train(load_gallery(args.gallery))
    notifier = PrintNotifier() if args.dry_run else EmailNotifier.from_env()
    doorbell = Doorbell(HaarFaceDetector(), recognizer, notifier, cooldown_s=args.cooldown)
    run_camera(doorbell, args.camera, args.max_fps)


if __name__ == "__main__":
    main()

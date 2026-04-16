"""ByteTrack-style tracker adapter with an IoU fallback implementation."""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = sys.path[0]
if SCRIPT_DIR.endswith("tracker"):
    sys.path.pop(0)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import List

import numpy as np

from core.types import BBox, Detection, Track
from tracker.base_tracker import BaseTracker
from tracker.utils import IoUTrackerCore, split_detections_by_confidence


class ByteTrackTracker(BaseTracker):
    """Lightweight ByteTrack-compatible tracker.

    This is not a full external ByteTrack dependency. It follows ByteTrack's
    practical idea of matching high-confidence detections first and using
    low-confidence detections to recover existing tracks. Track output uses the
    project's unified Track dataclass.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        low_iou_threshold: float = 0.2,
        high_conf_threshold: float = 0.5,
        low_conf_threshold: float = 0.1,
        max_missing: int = 30,
        return_missing: bool = False,
    ) -> None:
        if low_conf_threshold > high_conf_threshold:
            raise ValueError("low_conf_threshold must be <= high_conf_threshold")
        self.high_conf_threshold = high_conf_threshold
        self.low_conf_threshold = low_conf_threshold
        self.low_iou_threshold = low_iou_threshold
        self.return_missing = return_missing
        self.core = IoUTrackerCore(iou_threshold=iou_threshold, max_missing=max_missing)

    def update(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        frame_index: int,
    ) -> List[Track]:
        high_detections, low_detections = split_detections_by_confidence(
            detections,
            high_threshold=self.high_conf_threshold,
            low_threshold=self.low_conf_threshold,
        )
        return self.core.update_byte_style(
            high_detections=high_detections,
            low_detections=low_detections,
            frame_index=frame_index,
            low_iou_threshold=self.low_iou_threshold,
            return_missing=self.return_missing,
        )

    def clear(self) -> None:
        self.core.clear()


# Backward-compatible alias for common spelling.
BYTETracker = ByteTrackTracker


def _demo() -> None:
    tracker = ByteTrackTracker(high_conf_threshold=0.5, low_conf_threshold=0.1)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frames = [
        [Detection(BBox(10, 10, 30, 30), 0.90, 2, "car")],
        [Detection(BBox(12, 10, 32, 30), 0.88, 2, "car")],
        [Detection(BBox(14, 10, 34, 30), 0.20, 2, "car")],
        [Detection(BBox(50, 50, 70, 70), 0.95, 2, "car")],
    ]
    for frame_index, detections in enumerate(frames):
        tracks = tracker.update(detections, frame, frame_index)
        print(
            "frame",
            frame_index,
            [
                {
                    "track_id": track.track_id,
                    "label": track.label,
                    "score": round(track.score, 3),
                    "bbox": [track.bbox.x1, track.bbox.y1, track.bbox.x2, track.bbox.y2],
                }
                for track in tracks
            ],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small ByteTrackTracker demo")
    parser.add_argument("--demo", action="store_true", help="Print track ids for synthetic detections")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo:
        _demo()
    else:
        raise SystemExit("Use --demo to run the standalone tracker example")


if __name__ == "__main__":
    main()


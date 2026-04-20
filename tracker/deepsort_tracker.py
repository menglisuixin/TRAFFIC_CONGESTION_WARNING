"""DeepSORT tracker adapter with robust IoU fallback."""

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = sys.path[0]
if SCRIPT_DIR.endswith("tracker"):
    sys.path.pop(0)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import List, Optional, Tuple

import numpy as np

from core.types import BBox, Detection, Track
from tracker.base_tracker import BaseTracker
from tracker.utils import IoUTrackerCore, bbox_iou


class DeepSORTTracker(BaseTracker):
    """Use deep_sort_realtime when available, otherwise fall back to IoU tracking."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missing: int = 30,
        use_real_deepsort: bool = True,
        return_missing: bool = False,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.return_missing = return_missing
        self.core = IoUTrackerCore(iou_threshold=iou_threshold, max_missing=max_missing)
        self.backend_name = "iou_fallback"
        self.backend_error: Optional[str] = None
        self._real_tracker = self._create_real_tracker() if use_real_deepsort else None

    @property
    def using_real_deepsort(self) -> bool:
        return self._real_tracker is not None

    def update(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        frame_index: int,
    ) -> List[Track]:
        if self._real_tracker is not None:
            try:
                return self._update_real_tracker(detections, frame, frame_index)
            except Exception as exc:
                self.backend_error = f"DeepSORT runtime failure: {exc}"
                self._real_tracker = None
                self.backend_name = "iou_fallback"
                self.core.clear()
        return self.core.update(detections, frame_index, return_missing=self.return_missing)

    def clear(self) -> None:
        self.core.clear()
        if self._real_tracker is not None and hasattr(self._real_tracker, "delete_all_tracks"):
            try:
                self._real_tracker.delete_all_tracks()
            except Exception:
                pass

    def _create_real_tracker(self):
        try:
            from deep_sort_realtime.deepsort_tracker import DeepSort
        except ImportError as exc:
            self.backend_error = f"deep_sort_realtime is not installed: {exc}"
            self.backend_name = "iou_fallback"
            return None
        except Exception as exc:
            self.backend_error = f"failed to import deep_sort_realtime: {exc}"
            self.backend_name = "iou_fallback"
            return None

        try:
            tracker = DeepSort(max_age=self.max_missing, n_init=1)
        except Exception as exc:
            self.backend_error = f"failed to initialize DeepSort: {exc}"
            self.backend_name = "iou_fallback"
            return None

        self.backend_name = "deep_sort_realtime"
        self.backend_error = None
        return tracker

    def _update_real_tracker(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        frame_index: int,
    ) -> List[Track]:
        ds_detections = []
        for detection in detections:
            bbox = detection.bbox
            ds_detections.append(
                ([bbox.x1, bbox.y1, bbox.width, bbox.height], detection.conf, detection.label)
            )

        raw_tracks = self._real_tracker.update_tracks(ds_detections, frame=frame)
        frame_height, frame_width = frame.shape[:2]
        tracks: List[Track] = []
        for raw_track in raw_tracks:
            if hasattr(raw_track, "is_confirmed") and not raw_track.is_confirmed():
                continue
            if not self.return_missing and int(getattr(raw_track, "time_since_update", 0)) > 0:
                continue
            bbox = raw_track_to_bbox(raw_track, frame_width=frame_width, frame_height=frame_height, allow_prediction=self.return_missing)
            if bbox is None:
                continue
            cls_id, label, score = best_detection_metadata(bbox, detections)
            tracks.append(
                Track(
                    track_id=int(raw_track.track_id),
                    bbox=bbox,
                    cls_id=cls_id,
                    label=label,
                    score=score,
                    frame_index=frame_index,
                )
            )
        return tracks


# Backward-compatible alias for common capitalization.
DeepSortTracker = DeepSORTTracker


def raw_track_to_bbox(raw_track, frame_width: int, frame_height: int, allow_prediction: bool = False) -> Optional[BBox]:
    if not hasattr(raw_track, "to_ltrb"):
        return None
    try:
        ltrb = raw_track.to_ltrb(orig=True, orig_strict=True)
    except TypeError:
        ltrb = None
    if ltrb is None and allow_prediction:
        ltrb = raw_track.to_ltrb()
    if ltrb is None:
        return None

    left, top, right, bottom = [float(value) for value in ltrb]
    left = max(0.0, min(left, float(frame_width - 1)))
    right = max(0.0, min(right, float(frame_width - 1)))
    top = max(0.0, min(top, float(frame_height - 1)))
    bottom = max(0.0, min(bottom, float(frame_height - 1)))
    if right <= left or bottom <= top:
        return None
    return BBox(left, top, right, bottom)


def best_detection_metadata(bbox: BBox, detections: List[Detection]) -> Tuple[int, str, float]:
    best_detection: Optional[Detection] = None
    best_iou = 0.0
    for detection in detections:
        iou = bbox_iou(bbox, detection.bbox)
        if iou > best_iou:
            best_detection = detection
            best_iou = iou
    if best_detection is None:
        return -1, "", 0.0
    return best_detection.cls_id, best_detection.label, best_detection.conf


def _demo() -> None:
    tracker = DeepSORTTracker(use_real_deepsort=False)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frames = [
        [Detection(BBox(10, 10, 30, 30), 0.90, 2, "car")],
        [Detection(BBox(12, 10, 32, 30), 0.88, 2, "car")],
        [Detection(BBox(60, 60, 80, 80), 0.91, 2, "car")],
    ]
    for frame_index, detections in enumerate(frames):
        tracks = tracker.update(detections, frame, frame_index)
        print(
            "frame",
            frame_index,
            "backend",
            tracker.backend_name,
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
    parser = argparse.ArgumentParser(description="Run a small DeepSORTTracker fallback demo")
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


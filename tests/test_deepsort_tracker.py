import numpy as np

from core.types import BBox, Detection
from tracker.deepsort_tracker import DeepSORTTracker, raw_track_to_bbox


class FakeRawTrack:
    def __init__(self, track_id=1, time_since_update=0, original=None, predicted=None):
        self.track_id = track_id
        self.time_since_update = time_since_update
        self.original = original
        self.predicted = predicted or [100.0, 100.0, 160.0, 160.0]

    def is_confirmed(self):
        return True

    def to_ltrb(self, orig=False, orig_strict=False):
        if orig:
            if self.original is None and orig_strict:
                return None
            if self.original is not None:
                return np.array(self.original, dtype=float)
        return np.array(self.predicted, dtype=float)


class FakeRealTracker:
    def __init__(self, tracks):
        self.tracks = tracks

    def update_tracks(self, detections, frame=None):
        return self.tracks


def test_raw_track_to_bbox_prefers_original_detection_box() -> None:
    raw = FakeRawTrack(original=[10.0, 20.0, 50.0, 70.0], predicted=[100.0, 120.0, 180.0, 220.0])

    bbox = raw_track_to_bbox(raw, frame_width=200, frame_height=200)

    assert bbox == BBox(10.0, 20.0, 50.0, 70.0)


def test_raw_track_to_bbox_skips_prediction_when_not_allowed() -> None:
    raw = FakeRawTrack(original=None, predicted=[100.0, 120.0, 180.0, 220.0])

    assert raw_track_to_bbox(raw, frame_width=200, frame_height=200, allow_prediction=False) is None
    assert raw_track_to_bbox(raw, frame_width=200, frame_height=200, allow_prediction=True) == BBox(100.0, 120.0, 180.0, 199.0)


def test_deepsort_adapter_skips_unmatched_prediction_tracks_by_default() -> None:
    tracker = DeepSORTTracker(use_real_deepsort=False)
    tracker._real_tracker = FakeRealTracker([
        FakeRawTrack(track_id=1, time_since_update=0, original=[10.0, 10.0, 30.0, 30.0]),
        FakeRawTrack(track_id=2, time_since_update=2, original=None, predicted=[80.0, 80.0, 120.0, 120.0]),
    ])
    tracker.backend_name = "deep_sort_realtime"
    frame = np.zeros((160, 160, 3), dtype=np.uint8)
    detections = [Detection(BBox(10, 10, 30, 30), 0.9, 2, "car")]

    tracks = tracker.update(detections, frame, frame_index=0)

    assert len(tracks) == 1
    assert tracks[0].track_id == 1
    assert tracks[0].bbox == BBox(10.0, 10.0, 30.0, 30.0)

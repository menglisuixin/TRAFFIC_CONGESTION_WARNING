from core.pipeline import create_tracker
from tracker.bytetrack_tracker import ByteTrackTracker
from tracker.deepsort_tracker import DeepSORTTracker


def test_create_tracker_uses_bytetrack_when_configured() -> None:
    tracker = create_tracker({"type": "bytetrack", "iou_threshold": 0.4, "max_missing": 12})

    assert isinstance(tracker, ByteTrackTracker)
    assert tracker.core.iou_threshold == 0.4
    assert tracker.core.max_missing == 12


def test_create_tracker_can_force_iou_fallback() -> None:
    tracker = create_tracker({"type": "iou", "iou_threshold": 0.2, "max_missing": 5})

    assert isinstance(tracker, DeepSORTTracker)
    assert tracker.backend_name == "iou_fallback"
    assert not tracker.using_real_deepsort
    assert tracker.core.iou_threshold == 0.2


def test_create_tracker_deepsort_adapter_reports_backend() -> None:
    tracker = create_tracker({"type": "deepsort", "use_real_deepsort": False})

    assert isinstance(tracker, DeepSORTTracker)
    assert tracker.backend_name == "iou_fallback"
    assert not tracker.using_real_deepsort

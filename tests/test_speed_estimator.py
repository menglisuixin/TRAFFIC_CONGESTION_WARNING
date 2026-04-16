import math
from pathlib import Path

from analytics.speed_estimator import estimate_speeds_from_frames
from calibration.homography import compute_homography


def test_speed_estimator_uses_track_ids_and_pixel_speed() -> None:
    frames = [
        {
            "frame_index": 0,
            "detections": [
                {"track_id": 1, "class_id": 0, "label": "Motor Vehicle", "bbox": [0, 0, 10, 10]},
            ],
        },
        {
            "frame_index": 1,
            "detections": [
                {"track_id": 1, "class_id": 0, "label": "Motor Vehicle", "bbox": [3, 4, 13, 14]},
            ],
        },
    ]

    stats = estimate_speeds_from_frames(frames, fps=10.0)

    assert len(stats) == 2
    assert stats[0].vehicles[0].speed_px_per_frame == 0.0
    assert stats[1].vehicles[0].speed_px_per_frame == 5.0
    assert stats[1].vehicles[0].speed_px_per_second == 50.0
    assert stats[1].vehicles[0].speed_mps == 0.0
    assert stats[1].vehicles[0].speed_kmh == 0.0


def test_speed_estimator_converts_calibrated_world_speed_to_kmh() -> None:
    homography = compute_homography(
        [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)],
        [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
    )
    frames = [
        {"frame_index": 0, "detections": [{"track_id": 7, "class_id": 0, "label": "Motor Vehicle", "bbox": [0, 0, 10, 10]}]},
        {"frame_index": 1, "detections": [{"track_id": 7, "class_id": 0, "label": "Motor Vehicle", "bbox": [10, 0, 20, 10]}]},
    ]

    stats = estimate_speeds_from_frames(frames, fps=10.0, homography=homography)
    speed = stats[1].vehicles[0]

    # Bottom-center moves 10 pixels horizontally; calibration is 0.1 m/pixel.
    assert math.isclose(speed.speed_mps, 10.0, rel_tol=1e-6)
    assert math.isclose(speed.speed_kmh, 36.0, rel_tol=1e-6)
    assert math.isfinite(speed.speed_mps)
    assert math.isfinite(speed.speed_kmh)


def test_speed_estimator_fallback_iou_assigns_stable_ids() -> None:
    frames = [
        {"frame_index": 0, "detections": [{"class_id": 0, "label": "Motor Vehicle", "bbox": [0, 0, 20, 20]}]},
        {"frame_index": 1, "detections": [{"class_id": 0, "label": "Motor Vehicle", "bbox": [2, 0, 22, 20]}]},
    ]

    stats = estimate_speeds_from_frames(frames, fps=5.0, iou_threshold=0.1)

    assert stats[0].vehicles[0].track_id == stats[1].vehicles[0].track_id
    assert stats[1].vehicles[0].speed_px_per_frame == 2.0
    assert stats[1].vehicles[0].speed_px_per_second == 10.0

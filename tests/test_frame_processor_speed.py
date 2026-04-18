import math
from typing import List

import numpy as np

from core.frame_processor import FrameProcessor
from core.types import BBox, Detection, Point, Track


class StaticDetector:
    def infer(self, frame):
        return [Detection(BBox(0, 0, 10, 10), 0.9, 0, "Motor Vehicle")]




class JitterTracker:
    def __init__(self) -> None:
        self.positions = [0.0, 0.5, -0.4, 0.8, -0.2]

    def update(self, detections, frame, frame_index):
        x = self.positions[min(frame_index, len(self.positions) - 1)]
        return [Track(track_id=1, bbox=BBox(x, 0, x + 10, 10), cls_id=0, label="Motor Vehicle", score=0.9, frame_index=frame_index)]

    def clear(self) -> None:
        pass

class MovingTracker:
    def __init__(self) -> None:
        self.positions = [0.0, 10.0]

    def update(self, detections, frame, frame_index):
        x = self.positions[min(frame_index, len(self.positions) - 1)]
        return [Track(track_id=1, bbox=BBox(x, 0, x + 10, 10), cls_id=0, label="Motor Vehicle", score=0.9, frame_index=frame_index)]

    def clear(self) -> None:
        pass


def test_frame_processor_estimates_kmh_with_meters_per_pixel() -> None:
    processor = FrameProcessor(
        detector=StaticDetector(),
        tracker=MovingTracker(),
        fps=10.0,
        meters_per_pixel=0.1,
        speed_smoothing_alpha=0.0,
        max_speed_kmh=200.0,
        roi_points=[Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)],
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    first = processor.process(frame, 0).summary
    second = processor.process(frame, 1).summary

    first_det = first["detections"][0]
    second_det = second["detections"][0]
    assert first_det["speed_kmh"] == 0.0
    assert math.isclose(second_det["speed_px_per_frame"], 10.0, rel_tol=1e-6)
    assert math.isclose(second_det["speed_px_per_second"], 100.0, rel_tol=1e-6)
    assert math.isclose(second_det["speed_mps"], 10.0, rel_tol=1e-6)
    assert math.isclose(second_det["speed_kmh"], 36.0, rel_tol=1e-6)
    assert second_det["speed_source"] == "estimated_meters_per_pixel"


def test_frame_processor_suppresses_static_bbox_jitter() -> None:
    processor = FrameProcessor(
        detector=StaticDetector(),
        tracker=JitterTracker(),
        fps=10.0,
        meters_per_pixel=0.1,
        speed_smoothing_alpha=0.0,
        min_motion_px_per_frame=2.0,
        roi_points=[Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)],
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    summaries = [processor.process(frame, index).summary for index in range(5)]
    detections = [summary["detections"][0] for summary in summaries]

    assert all(item["speed_kmh"] == 0.0 for item in detections)
    assert all(item["speed_mps"] == 0.0 for item in detections)
    assert all(item["motion_suppressed"] for item in detections)
    assert detections[-1]["speed_source"] == "stationary_jitter_suppressed"



class OutsideRoiTracker:
    def __init__(self) -> None:
        self.positions = [0.0, 10.0]

    def update(self, detections, frame, frame_index):
        x = self.positions[min(frame_index, len(self.positions) - 1)]
        return [Track(track_id=1, bbox=BBox(x, 80, x + 10, 90), cls_id=0, label="Motor Vehicle", score=0.9, frame_index=frame_index)]

    def clear(self) -> None:
        pass


def test_frame_processor_limits_speed_stats_to_roi() -> None:
    processor = FrameProcessor(
        detector=StaticDetector(),
        tracker=OutsideRoiTracker(),
        fps=10.0,
        meters_per_pixel=0.1,
        speed_smoothing_alpha=0.0,
        min_motion_px_per_frame=0.0,
        roi_points=[Point(0, 0), Point(100, 0), Point(100, 50), Point(0, 50)],
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    first = processor.process(frame, 0).summary
    second = processor.process(frame, 1).summary

    assert first["detections_count"] == 0
    assert first["detections"] == []
    assert second["detections_count"] == 0
    assert second["detections"] == []
    assert second["speed_count"] == 0
    assert second["mean_speed_kmh"] == 0.0


class MultiRegionTracker:
    def update(self, detections, frame, frame_index):
        return [
            Track(track_id=1, bbox=BBox(10, 10, 30, 30), cls_id=2, label="car", score=0.9, frame_index=frame_index),
            Track(track_id=2, bbox=BBox(70, 10, 90, 30), cls_id=7, label="truck", score=0.9, frame_index=frame_index),
        ]

    def clear(self) -> None:
        pass


def test_frame_processor_reports_independent_roi_regions() -> None:
    processor = FrameProcessor(
        detector=StaticDetector(),
        tracker=MultiRegionTracker(),
        fps=10.0,
        meters_per_pixel=0.1,
        speed_smoothing_alpha=0.0,
        roi_regions=[
            {"name": "A_to_B", "points": [Point(0, 0), Point(50, 0), Point(50, 60), Point(0, 60)]},
            {"name": "B_to_A", "points": [Point(50, 0), Point(100, 0), Point(100, 60), Point(50, 60)]},
        ],
    )
    frame = np.zeros((80, 100, 3), dtype=np.uint8)

    summary = processor.process(frame, 0).summary

    assert summary["vehicle_count"] == 2
    assert len(summary["regions"]) == 2
    by_name = {region["name"]: region for region in summary["regions"]}
    assert by_name["A_to_B"]["vehicle_count"] == 1
    assert by_name["B_to_A"]["vehicle_count"] == 1
    assert summary["detections"][0]["region_name"] == "A_to_B"
    assert summary["detections"][1]["region_name"] == "B_to_A"


class MixedInsideOutsideTracker:
    def update(self, detections, frame, frame_index):
        return [
            Track(track_id=1, bbox=BBox(10, 10, 30, 30), cls_id=2, label="car", score=0.9, frame_index=frame_index),
            Track(track_id=2, bbox=BBox(70, 70, 90, 90), cls_id=7, label="truck", score=0.9, frame_index=frame_index),
        ]

    def clear(self) -> None:
        pass


def test_frame_processor_outputs_only_tracks_inside_roi() -> None:
    processor = FrameProcessor(
        detector=StaticDetector(),
        tracker=MixedInsideOutsideTracker(),
        fps=10.0,
        meters_per_pixel=0.1,
        speed_smoothing_alpha=0.0,
        roi_points=[Point(0, 0), Point(50, 0), Point(50, 50), Point(0, 50)],
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    summary = processor.process(frame, 0).summary

    assert summary["detections_count"] == 1
    assert len(summary["detections"]) == 1
    assert summary["detections"][0]["track_id"] == 1
    assert summary["vehicle_count"] == 1



class SequenceTracker:
    def __init__(self, positions: List[float]) -> None:
        self.positions = positions

    def update(self, detections, frame, frame_index):
        x = self.positions[min(frame_index, len(self.positions) - 1)]
        return [Track(track_id=1, bbox=BBox(x, 0, x + 10, 10), cls_id=2, label="car", score=0.9, frame_index=frame_index)]

    def clear(self) -> None:
        pass


def test_frame_processor_warms_up_speed_before_stats() -> None:
    processor = FrameProcessor(
        detector=StaticDetector(),
        tracker=SequenceTracker([0.0, 10.0, 20.0, 30.0]),
        fps=10.0,
        meters_per_pixel=0.1,
        speed_smoothing_alpha=0.0,
        speed_warmup_frames=3,
        speed_history_size=3,
        roi_points=[Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)],
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    summaries = [processor.process(frame, index).summary for index in range(4)]

    assert summaries[1]["detections"][0]["instant_speed_kmh"] == 36.0
    assert summaries[1]["detections"][0]["speed_kmh"] == 0.0
    assert not summaries[1]["detections"][0]["speed_valid"]
    assert summaries[1]["moving_speed_count"] == 0
    assert summaries[3]["detections"][0]["speed_kmh"] == 36.0
    assert summaries[3]["detections"][0]["speed_valid"]
    assert summaries[3]["moving_speed_count"] == 1


def test_frame_processor_suppresses_transient_low_speed_drop() -> None:
    processor = FrameProcessor(
        detector=StaticDetector(),
        tracker=SequenceTracker([0.0, 10.0, 20.0, 30.0, 31.0, 50.0]),
        fps=10.0,
        meters_per_pixel=0.1,
        speed_smoothing_alpha=0.0,
        speed_warmup_frames=1,
        speed_history_size=3,
        speed_max_drop_ratio=0.45,
        min_motion_px_per_frame=0.0,
        roi_points=[Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)],
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    summaries = [processor.process(frame, index).summary for index in range(6)]
    low_drop = summaries[4]["detections"][0]

    assert low_drop["instant_speed_kmh"] == 3.6
    assert low_drop["speed_kmh"] == 36.0
    assert low_drop["speed_valid"]
    assert "transient_low_suppressed" in low_drop["speed_source"]
    assert summaries[4]["moving_mean_speed_kmh"] == 36.0



def test_frame_processor_holds_previous_stable_speed_during_jitter() -> None:
    processor = FrameProcessor(
        detector=StaticDetector(),
        tracker=SequenceTracker([0.0, 10.0, 20.0, 30.0, 30.4, 30.6]),
        fps=10.0,
        meters_per_pixel=0.1,
        speed_smoothing_alpha=0.0,
        speed_warmup_frames=1,
        speed_history_size=3,
        speed_hold_frames=2,
        min_motion_px_per_frame=2.0,
        roi_points=[Point(0, 0), Point(100, 0), Point(100, 100), Point(0, 100)],
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    summaries = [processor.process(frame, index).summary for index in range(6)]
    held = summaries[4]["detections"][0]
    held_again = summaries[5]["detections"][0]

    assert held["instant_speed_kmh"] == 0.0
    assert held["speed_kmh"] == 36.0
    assert held["speed_valid"]
    assert held["speed_source"] == "stationary_jitter_suppressed_hold"
    assert held_again["speed_valid"]
    assert summaries[4]["moving_mean_speed_kmh"] == 36.0

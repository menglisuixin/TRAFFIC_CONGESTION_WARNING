"""Single-frame processing for traffic analysis pipelines."""

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from analytics.congestion_detector import CongestionDetector
from analytics.density_estimator import ROIShape, estimate_roi_density
from analytics.flow_counter import FlowCounter, LineSegment
from analytics.speed_estimator import distance as point_distance
from calibration.homography import Homography
from core.types import BBox, Detection, Point, Track
from detector.yolov5_detector import YOLOv5Detector
from tracker.deepsort_tracker import DeepSORTTracker


CLASS_WEIGHTS = {
    "motor_vehicle": 1.0,
    "non_motor": 0.5,
    "pedestrian": 0.2,
}

SpeedInfo = Dict[str, float]


@dataclass(frozen=True)
class FrameProcessResult:
    """Processed frame and serializable per-frame summary."""

    frame: np.ndarray
    summary: Dict[str, object]


class FrameProcessor:
    """Runs detection, tracking, analytics, and drawing for one frame at a time."""

    def __init__(
        self,
        detector: YOLOv5Detector,
        tracker: Optional[DeepSORTTracker] = None,
        roi_points: Optional[Sequence[Point]] = None,
        fps: float = 30.0,
        every_n: int = 1,
        homography: Optional[Homography] = None,
    ) -> None:
        if every_n <= 0:
            raise ValueError("every_n must be greater than zero")
        if fps <= 0.0:
            raise ValueError("fps must be greater than zero")

        self.detector = detector
        self.tracker = tracker or DeepSORTTracker()
        self.roi_points = list(roi_points) if roi_points else []
        self.fps = fps
        self.every_n = every_n
        self.homography = homography
        self.roi_shape: Optional[ROIShape] = None
        self.flow_counter: Optional[FlowCounter] = None
        self.congestion_detector = CongestionDetector()
        self.previous_points: Dict[int, Point] = {}
        self.previous_world_points: Dict[int, Tuple[float, float]] = {}
        self.last_detections: List[Detection] = []

        if self.roi_points:
            self._set_roi(self.roi_points)

    def set_default_roi(self, frame_width: int, frame_height: int) -> None:
        if self.roi_points:
            return
        self._set_roi(
            [
                Point(frame_width * 0.15, frame_height * 0.45),
                Point(frame_width * 0.85, frame_height * 0.45),
                Point(frame_width * 0.95, frame_height * 0.95),
                Point(frame_width * 0.05, frame_height * 0.95),
            ]
        )

    def process(self, frame: np.ndarray, frame_index: int) -> FrameProcessResult:
        """Process one BGR frame and return an annotated frame plus summary."""

        height, width = frame.shape[:2]
        self.set_default_roi(width, height)
        detected_this_frame = frame_index % self.every_n == 0
        if detected_this_frame:
            self.last_detections = self.detector.infer(frame)

        tracks = self.tracker.update(self.last_detections, frame, frame_index)
        speed_by_track = self._estimate_speeds(tracks)
        self._update_flow(tracks)
        density_stats = self._compute_density_stats(tracks)
        speed_stats = self._speed_stats(speed_by_track)
        low_speed_ratio = self._low_speed_ratio(speed_by_track)
        congestion_level, warning_active = self.congestion_detector.update(
            roi_vehicle_count=density_stats["vehicle_count"] + density_stats["non_motor_count"],
            low_speed_ratio=low_speed_ratio,
            occupancy_ratio=density_stats["occupancy_ratio"],
        )
        congestion_status = self._congestion_status(
            weighted_density_per_100k=density_stats["weighted_density"] * 100000.0,
            warning_active=warning_active,
            congestion_level=congestion_level,
        )

        annotated = self._draw_frame(
            frame.copy(),
            tracks=tracks,
            speed_by_track=speed_by_track,
            density_stats=density_stats,
            speed_stats=speed_stats,
            congestion_status=congestion_status,
        )
        summary = self._build_summary(
            frame_index=frame_index,
            detected_this_frame=detected_this_frame,
            tracks=tracks,
            speed_by_track=speed_by_track,
            speed_stats=speed_stats,
            density_stats=density_stats,
            congestion_level=congestion_level,
            congestion_status=congestion_status,
            warning_active=warning_active,
        )
        return FrameProcessResult(frame=annotated, summary=summary)

    def _set_roi(self, roi_points: Sequence[Point]) -> None:
        self.roi_points = list(roi_points)
        self.roi_shape = ROIShape([(point.x, point.y) for point in self.roi_points])
        self.flow_counter = FlowCounter(LineSegment(self.roi_points[0], self.roi_points[1]), direction="any")

    def _estimate_speeds(self, tracks: Sequence[Track]) -> Dict[int, SpeedInfo]:
        speeds: Dict[int, SpeedInfo] = {}
        for track in tracks:
            current = track.bbox.bottom_center
            previous = self.previous_points.get(track.track_id)
            current_tuple = (current.x, current.y)
            previous_tuple = None if previous is None else (previous.x, previous.y)
            speed_px_per_frame = 0.0 if previous_tuple is None else point_distance(previous_tuple, current_tuple)
            speed_px_per_second = speed_px_per_frame * self.fps

            speed_mps = 0.0
            speed_kmh = 0.0
            if self.homography is not None:
                current_world = self.homography.pixel_to_world(current.x, current.y)
                previous_world = self.previous_world_points.get(track.track_id)
                if previous_world is not None:
                    speed_mps = point_distance(previous_world, current_world) * self.fps
                    speed_kmh = speed_mps * 3.6
                self.previous_world_points[track.track_id] = current_world

            speeds[track.track_id] = {
                "speed_px_per_frame": safe_float(speed_px_per_frame),
                "speed_px_per_second": safe_float(speed_px_per_second),
                "speed_mps": safe_float(speed_mps),
                "speed_kmh": safe_float(speed_kmh),
            }
        return speeds

    def _update_flow(self, tracks: Sequence[Track]) -> None:
        active_ids = set()
        for track in tracks:
            active_ids.add(track.track_id)
            current = track.bbox.bottom_center
            previous = self.previous_points.get(track.track_id)
            if previous is not None and self.flow_counter is not None:
                self.flow_counter.update(track.track_id, previous, current)
            self.previous_points[track.track_id] = current

        for track_id in list(self.previous_points):
            if track_id not in active_ids:
                del self.previous_points[track_id]
                self.previous_world_points.pop(track_id, None)

    def _compute_density_stats(self, tracks: Sequence[Track]) -> Dict[str, float]:
        if self.roi_shape is None:
            raise RuntimeError("ROI is not initialized")

        vehicle_count = 0
        non_motor_count = 0
        pedestrian_count = 0
        weighted_count = 0.0
        occupied_area = 0.0

        for track in tracks:
            bbox_tuple = bbox_to_tuple(track.bbox)
            if not self.roi_shape.contains_bbox(bbox_tuple):
                continue
            category = classify_track(track)
            if category == "motor_vehicle":
                vehicle_count += 1
            elif category == "non_motor":
                non_motor_count += 1
            elif category == "pedestrian":
                pedestrian_count += 1
            else:
                continue
            weighted_count += CLASS_WEIGHTS.get(category, 1.0)
            occupied_area += track.bbox.area

        total_count = vehicle_count + non_motor_count + pedestrian_count
        density = estimate_roi_density(total_count, self.roi_shape.area)
        weighted_density = estimate_roi_density(weighted_count, self.roi_shape.area)
        occupancy_ratio = min(1.0, occupied_area / self.roi_shape.area) if self.roi_shape.area > 0 else 0.0
        return {
            "vehicle_count": vehicle_count,
            "non_motor_count": non_motor_count,
            "pedestrian_count": pedestrian_count,
            "density": safe_float(density),
            "weighted_density": safe_float(weighted_density),
            "density_per_100k": safe_float(weighted_density * 100000.0),
            "occupancy_ratio": safe_float(occupancy_ratio),
            "roi_area": safe_float(self.roi_shape.area),
        }

    def _speed_stats(self, speed_by_track: Mapping[int, SpeedInfo]) -> Dict[str, float]:
        speeds_kmh = [safe_float(info.get("speed_kmh")) for info in speed_by_track.values()]
        speeds_mps = [safe_float(info.get("speed_mps")) for info in speed_by_track.values()]
        speeds_px = [safe_float(info.get("speed_px_per_second")) for info in speed_by_track.values()]
        return {
            "mean_speed_kmh": sum(speeds_kmh) / len(speeds_kmh) if speeds_kmh else 0.0,
            "max_speed_kmh": max(speeds_kmh) if speeds_kmh else 0.0,
            "mean_speed_mps": sum(speeds_mps) / len(speeds_mps) if speeds_mps else 0.0,
            "max_speed_mps": max(speeds_mps) if speeds_mps else 0.0,
            "mean_speed_px_per_second": sum(speeds_px) / len(speeds_px) if speeds_px else 0.0,
        }

    def _low_speed_ratio(self, speed_by_track: Mapping[int, SpeedInfo], threshold_px_per_second: float = 60.0) -> float:
        if not speed_by_track:
            return 0.0
        low_count = sum(1 for info in speed_by_track.values() if safe_float(info.get("speed_px_per_second")) < threshold_px_per_second)
        return low_count / len(speed_by_track)

    def _congestion_status(self, weighted_density_per_100k: float, warning_active: bool, congestion_level: str) -> str:
        if weighted_density_per_100k >= 0.8 or congestion_level in {"congested", "severe"}:
            return "CONGESTED"
        if weighted_density_per_100k >= 0.3 or warning_active or congestion_level == "slow":
            return "WARNING"
        return "CLEAR"

    def _draw_frame(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        speed_by_track: Mapping[int, SpeedInfo],
        density_stats: Mapping[str, float],
        speed_stats: Mapping[str, float],
        congestion_status: str,
    ) -> np.ndarray:
        for track in tracks:
            self._draw_track(frame, track, speed_by_track.get(track.track_id, {}))
        self._draw_roi(frame)
        self._draw_status_panel(frame, tracks, density_stats, speed_stats, congestion_status)
        return frame

    def _draw_track(self, frame: np.ndarray, track: Track, speed_info: Mapping[str, float]) -> None:
        color = color_for_class(track.cls_id)
        bbox = track.bbox
        p1 = int(round(bbox.x1)), int(round(bbox.y1))
        p2 = int(round(bbox.x2)), int(round(bbox.y2))
        cv2.rectangle(frame, p1, p2, color, 2, cv2.LINE_AA)
        speed_kmh = safe_float(speed_info.get("speed_kmh"))
        label = f"ID {track.track_id} {track.label} {track.score:.2f} {speed_kmh:.1f} km/h"
        draw_label(frame, label, p1, color)

    def _draw_roi(self, frame: np.ndarray) -> None:
        if not self.roi_points:
            return
        points = np.array([(int(p.x), int(p.y)) for p in self.roi_points], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], isClosed=True, color=(255, 180, 0), thickness=2)

    def _draw_status_panel(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        density_stats: Mapping[str, float],
        speed_stats: Mapping[str, float],
        congestion_status: str,
    ) -> None:
        status_color = {
            "CLEAR": (0, 220, 0),
            "WARNING": (0, 220, 255),
            "CONGESTED": (0, 0, 255),
        }.get(congestion_status, (255, 255, 255))
        flow_count = self.flow_counter.total_count if self.flow_counter is not None else 0
        lines = [
            f"Tracks: {len(tracks):d}",
            f"Motor: {int(density_stats['vehicle_count'])}  NonMotor: {int(density_stats['non_motor_count'])}  Ped: {int(density_stats['pedestrian_count'])}",
            f"Flow Count: {flow_count:d}",
            f"Density: {density_stats['density']:.4f}",
            f"Weighted Density: {density_stats['weighted_density']:.4f}",
            f"Occupancy: {density_stats['occupancy_ratio']:.4f}",
            f"Avg Speed: {safe_float(speed_stats.get('mean_speed_kmh')):.2f} km/h",
            f"Max Speed: {safe_float(speed_stats.get('max_speed_kmh')):.2f} km/h",
            f"Status: {congestion_status}",
        ]
        panel_x1, panel_y1 = 8, 8
        row_height = 24
        panel_width = 590
        panel_height = 22 + row_height * len(lines)
        panel_x2 = min(frame.shape[1] - 1, panel_x1 + panel_width)
        panel_y2 = min(frame.shape[0] - 1, panel_y1 + panel_height)

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), status_color, 1)

        for index, line in enumerate(lines):
            color = status_color if index == len(lines) - 1 else (255, 255, 255)
            cv2.putText(
                frame,
                line,
                (panel_x1 + 12, panel_y1 + 26 + index * row_height),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
                cv2.LINE_AA,
            )

    def _build_summary(
        self,
        frame_index: int,
        detected_this_frame: bool,
        tracks: Sequence[Track],
        speed_by_track: Mapping[int, SpeedInfo],
        speed_stats: Mapping[str, float],
        density_stats: Mapping[str, float],
        congestion_level: str,
        congestion_status: str,
        warning_active: bool,
    ) -> Dict[str, object]:
        return {
            "frame_index": frame_index,
            "detected_this_frame": detected_this_frame,
            "detections_count": len(tracks),
            "detections": [track_to_dict(track, speed_by_track.get(track.track_id, {})) for track in tracks],
            "flow_count": self.flow_counter.total_count if self.flow_counter is not None else 0,
            "vehicle_count": int(density_stats["vehicle_count"]),
            "non_motor_count": int(density_stats["non_motor_count"]),
            "pedestrian_count": int(density_stats["pedestrian_count"]),
            "density": round(float(density_stats["density"]), 10),
            "weighted_density": round(float(density_stats["weighted_density"]), 10),
            "density_per_100k": round(float(density_stats["density_per_100k"]), 6),
            "occupancy_ratio": round(float(density_stats["occupancy_ratio"]), 6),
            "roi_area": round(float(density_stats["roi_area"]), 3),
            "mean_speed_kmh": round(safe_float(speed_stats.get("mean_speed_kmh")), 6),
            "max_speed_kmh": round(safe_float(speed_stats.get("max_speed_kmh")), 6),
            "mean_speed_mps": round(safe_float(speed_stats.get("mean_speed_mps")), 6),
            "max_speed_mps": round(safe_float(speed_stats.get("max_speed_mps")), 6),
            "mean_speed_px_per_second": round(safe_float(speed_stats.get("mean_speed_px_per_second")), 6),
            "congestion_level": congestion_level,
            "congestion_status": congestion_status,
            "warning_active": warning_active,
        }


def classify_track(track: Track) -> str:
    label = track.label.strip().lower()
    if label in {"motor vehicle", "vehicle", "car", "bus", "truck", "van"}:
        return "motor_vehicle"
    if label in {"non_motorized vehicle", "non-motorized vehicle", "bicycle", "motorcycle", "bike"}:
        return "non_motor"
    if label in {"pedestrian", "person"}:
        return "pedestrian"
    if track.cls_id in {0, 2, 5, 7}:
        return "motor_vehicle"
    if track.cls_id == 1:
        return "non_motor"
    return "other"


def track_to_dict(track: Track, speed_info: Mapping[str, float]) -> Dict[str, object]:
    bbox = track.bbox
    return {
        "track_id": track.track_id,
        "class_id": track.cls_id,
        "label": track.label,
        "conf": round(float(track.score), 6),
        "bbox": [round(float(bbox.x1), 3), round(float(bbox.y1), 3), round(float(bbox.x2), 3), round(float(bbox.y2), 3)],
        "speed_px_per_frame": round(safe_float(speed_info.get("speed_px_per_frame")), 6),
        "speed_px_per_second": round(safe_float(speed_info.get("speed_px_per_second")), 6),
        "speed_mps": round(safe_float(speed_info.get("speed_mps")), 6),
        "speed_kmh": round(safe_float(speed_info.get("speed_kmh")), 6),
    }


def bbox_to_tuple(bbox: BBox) -> Tuple[float, float, float, float]:
    return bbox.x1, bbox.y1, bbox.x2, bbox.y2


def draw_label(frame: np.ndarray, label: str, top_left: Tuple[int, int], color: Tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    text_size, baseline = cv2.getTextSize(label, font, scale, thickness)
    text_w, text_h = text_size
    x, y = top_left
    outside = y - text_h - baseline - 4 >= 0
    y1 = y - text_h - baseline - 4 if outside else y
    y2 = y if outside else y + text_h + baseline + 4
    cv2.rectangle(frame, (x, y1), (x + text_w + 4, y2), color, -1)
    text_y = y - baseline - 2 if outside else y + text_h + 2
    cv2.putText(frame, label, (x + 2, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def color_for_class(class_id: int) -> Tuple[int, int, int]:
    palette = [
        (56, 56, 255),
        (151, 157, 255),
        (31, 112, 255),
        (29, 178, 255),
        (49, 210, 207),
        (10, 249, 72),
        (23, 204, 146),
        (134, 219, 61),
    ]
    return palette[class_id % len(palette)]


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default

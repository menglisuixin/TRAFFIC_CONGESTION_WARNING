"""Single-frame processing for traffic analysis pipelines."""

import math
import statistics
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

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

SpeedInfo = Dict[str, object]


@dataclass
class TrafficRegion:
    """Independent traffic statistics area."""

    name: str
    points: List[Point]
    shape: ROIShape
    congestion_detector: CongestionDetector
    flow_counter: Optional[FlowCounter]


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
        roi_regions: Optional[Sequence[Mapping[str, object]]] = None,
        fps: float = 30.0,
        every_n: int = 1,
        homography: Optional[Homography] = None,
        meters_per_pixel: Optional[float] = None,
        speed_smoothing_alpha: float = 0.35,
        max_speed_kmh: float = 160.0,
        min_motion_px_per_frame: float = 2.0,
        speed_warmup_frames: int = 1,
        speed_history_size: int = 5,
        speed_max_drop_ratio: float = 0.45,
        speed_hold_frames: int = 8,
    ) -> None:
        if every_n <= 0:
            raise ValueError("every_n must be greater than zero")
        if fps <= 0.0:
            raise ValueError("fps must be greater than zero")

        self.detector = detector
        self.tracker = tracker or DeepSORTTracker()
        self.roi_points = list(roi_points) if roi_points else []
        self.regions: List[TrafficRegion] = []
        self.fps = fps
        self.every_n = every_n
        self.homography = homography
        self.meters_per_pixel = meters_per_pixel if meters_per_pixel and meters_per_pixel > 0.0 else None
        self.speed_smoothing_alpha = min(1.0, max(0.0, speed_smoothing_alpha))
        self.max_speed_kmh = max_speed_kmh if max_speed_kmh > 0.0 else 160.0
        self.min_motion_px_per_frame = max(0.0, float(min_motion_px_per_frame))
        self.speed_warmup_frames = max(1, int(speed_warmup_frames))
        self.speed_history_size = max(1, int(speed_history_size))
        self.speed_max_drop_ratio = min(1.0, max(0.0, float(speed_max_drop_ratio)))
        self.speed_hold_frames = max(0, int(speed_hold_frames))
        self.roi_shape: Optional[ROIShape] = None
        self.flow_counter: Optional[FlowCounter] = None
        self.congestion_detector = CongestionDetector()
        self.previous_points: Dict[int, Point] = {}
        self.previous_frame_indices: Dict[int, int] = {}
        self.previous_speed_points: Dict[int, Point] = {}
        self.previous_speed_frame_indices: Dict[int, int] = {}
        self.previous_world_points: Dict[int, Tuple[float, float]] = {}
        self.previous_speed_info: Dict[int, SpeedInfo] = {}
        self.speed_valid_counts: Dict[int, int] = {}
        self.stable_speed_histories: Dict[int, List[float]] = {}
        self.speed_hold_counts: Dict[int, int] = {}
        self.last_detections: List[Detection] = []

        if roi_regions:
            self._set_regions(roi_regions)
        elif self.roi_points:
            self._set_roi(self.roi_points)

    def set_default_roi(self, frame_width: int, frame_height: int) -> None:
        if self.regions:
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

        all_tracks = self.tracker.update(self.last_detections, frame, frame_index)
        speed_by_track = self._estimate_speeds(all_tracks, frame_index)
        visible_tracks = [track for track in all_tracks if self._track_in_roi(track)]
        visible_speed_by_track = {track.track_id: speed_by_track.get(track.track_id, {}) for track in visible_tracks}
        self._update_flow(visible_tracks, active_track_ids={track.track_id for track in all_tracks})
        region_summaries = self._compute_region_summaries(visible_tracks, visible_speed_by_track)
        density_stats = aggregate_region_summaries(region_summaries)
        speed_stats = density_stats
        congestion_level = str(density_stats.get("congestion_level", "normal"))
        congestion_status = str(density_stats.get("congestion_status", "CLEAR"))
        warning_active = bool(density_stats.get("warning_active", False))

        annotated = self._draw_frame(
            frame.copy(),
            tracks=visible_tracks,
            speed_by_track=visible_speed_by_track,
            density_stats=density_stats,
            speed_stats=speed_stats,
            congestion_status=congestion_status,
            region_summaries=region_summaries,
        )
        summary = self._build_summary(
            frame_index=frame_index,
            detected_this_frame=detected_this_frame,
            tracks=visible_tracks,
            speed_by_track=visible_speed_by_track,
            speed_stats=speed_stats,
            density_stats=density_stats,
            congestion_level=congestion_level,
            congestion_status=congestion_status,
            warning_active=warning_active,
            region_summaries=region_summaries,
        )
        return FrameProcessResult(frame=annotated, summary=summary)

    def _set_roi(self, roi_points: Sequence[Point]) -> None:
        self._set_regions([{"name": "ROI", "points": list(roi_points)}])

    def _set_regions(self, roi_regions: Sequence[Mapping[str, object]]) -> None:
        regions: List[TrafficRegion] = []
        for index, raw in enumerate(roi_regions):
            name = str(raw.get("name") or raw.get("id") or f"ROI_{index + 1}")
            points = parse_points(raw.get("points") or raw.get("polygon") or raw.get("roi"))
            shape = ROIShape([(point.x, point.y) for point in points])
            direction = str(raw.get("direction") or "any")
            flow_counter = FlowCounter(LineSegment(points[0], points[1]), direction=direction) if len(points) >= 2 else None
            regions.append(TrafficRegion(name=name, points=points, shape=shape, congestion_detector=CongestionDetector(), flow_counter=flow_counter))
        if not regions:
            raise ValueError("at least one ROI region is required")
        self.regions = regions
        self.roi_points = regions[0].points
        self.roi_shape = regions[0].shape
        self.flow_counter = regions[0].flow_counter

    def _estimate_speeds(self, tracks: Sequence[Track], frame_index: int) -> Dict[int, SpeedInfo]:
        speeds: Dict[int, SpeedInfo] = {}
        for track in tracks:
            region_names = self._track_region_names(track)
            in_roi = bool(region_names)
            if not in_roi:
                self._clear_speed_state(track.track_id)
                speeds[track.track_id] = {
                    "speed_px_per_frame": 0.0,
                    "speed_px_per_second": 0.0,
                    "speed_mps": 0.0,
                    "speed_kmh": 0.0,
                    "instant_speed_mps": 0.0,
                    "instant_speed_kmh": 0.0,
                    "stable_speed_kmh": 0.0,
                    "speed_valid": 0.0,
                    "speed_sample_count": 0.0,
                    "speed_source": "outside_roi",
                    "delta_frames": 1.0,
                    "motion_suppressed": 1.0,
                    "raw_motion_px": 0.0,
                    "in_roi": 0.0,
                    "region_name": "",
                    "regions": [],
                }
                continue

            current = track.bbox.bottom_center
            previous = self.previous_speed_points.get(track.track_id)
            previous_frame_index = self.previous_speed_frame_indices.get(track.track_id)
            current_tuple = (current.x, current.y)
            previous_tuple = None if previous is None else (previous.x, previous.y)
            delta_frames = max(1, frame_index - previous_frame_index) if previous_frame_index is not None else 1
            delta_time = delta_frames / self.fps
            raw_distance_px = 0.0 if previous_tuple is None else point_distance(previous_tuple, current_tuple)
            raw_speed_px_per_frame = raw_distance_px / delta_frames
            motion_suppressed = previous_tuple is None or raw_speed_px_per_frame < self.min_motion_px_per_frame

            speed_px_per_frame = 0.0
            speed_px_per_second = 0.0
            speed_mps = 0.0
            speed_kmh = 0.0
            speed_source = "pixel_only"

            if previous_tuple is None:
                self.previous_speed_points[track.track_id] = current
                self.previous_speed_frame_indices[track.track_id] = frame_index
                if self.homography is not None:
                    self.previous_world_points[track.track_id] = self.homography.pixel_to_world(current.x, current.y)
            elif motion_suppressed:
                # Treat sub-threshold bbox jitter as stationary and keep the anchor point.
                speed_source = "stationary_jitter_suppressed"
            elif delta_time > 0.0:
                speed_px_per_frame = raw_speed_px_per_frame
                speed_px_per_second = speed_px_per_frame * self.fps
                if self.homography is not None:
                    current_world = self.homography.pixel_to_world(current.x, current.y)
                    previous_world = self.previous_world_points.get(track.track_id)
                    if previous_world is not None:
                        speed_mps = point_distance(previous_world, current_world) / delta_time
                        speed_kmh = speed_mps * 3.6
                        speed_source = "homography"
                    self.previous_world_points[track.track_id] = current_world
                elif self.meters_per_pixel is not None:
                    speed_mps = speed_px_per_second * self.meters_per_pixel
                    speed_kmh = speed_mps * 3.6
                    speed_source = "estimated_meters_per_pixel"

                self.previous_speed_points[track.track_id] = current
                self.previous_speed_frame_indices[track.track_id] = frame_index

            speed_kmh = min(safe_float(speed_kmh), self.max_speed_kmh)
            speed_mps = speed_kmh / 3.6 if speed_kmh > 0.0 else safe_float(speed_mps)
            previous_speed = self.previous_speed_info.get(track.track_id)
            if not motion_suppressed and previous_speed is not None and self.speed_smoothing_alpha > 0.0:
                alpha = self.speed_smoothing_alpha
                speed_px_per_frame = alpha * speed_px_per_frame + (1.0 - alpha) * safe_float(previous_speed.get("speed_px_per_frame"))
                speed_px_per_second = alpha * speed_px_per_second + (1.0 - alpha) * safe_float(previous_speed.get("speed_px_per_second"))
                speed_mps = alpha * speed_mps + (1.0 - alpha) * safe_float(previous_speed.get("instant_speed_mps", previous_speed.get("speed_mps")))
                speed_kmh = alpha * speed_kmh + (1.0 - alpha) * safe_float(previous_speed.get("instant_speed_kmh", previous_speed.get("speed_kmh")))

            instant_speed_kmh = safe_float(speed_kmh)
            instant_speed_mps = instant_speed_kmh / 3.6 if instant_speed_kmh > 0.0 else safe_float(speed_mps)
            stable_speed_kmh, stable_speed_mps, speed_valid, speed_sample_count, speed_source = self._stabilize_speed(
                track_id=track.track_id,
                instant_speed_kmh=instant_speed_kmh,
                motion_suppressed=motion_suppressed,
                speed_source=speed_source,
            )

            info = {
                "speed_px_per_frame": safe_float(speed_px_per_frame),
                "speed_px_per_second": safe_float(speed_px_per_second),
                "speed_mps": safe_float(stable_speed_mps),
                "speed_kmh": safe_float(stable_speed_kmh),
                "instant_speed_mps": safe_float(instant_speed_mps),
                "instant_speed_kmh": safe_float(instant_speed_kmh),
                "stable_speed_kmh": safe_float(stable_speed_kmh),
                "speed_valid": 1.0 if speed_valid else 0.0,
                "speed_sample_count": float(speed_sample_count),
                "speed_source": speed_source,
                "delta_frames": float(delta_frames),
                "motion_suppressed": 1.0 if motion_suppressed else 0.0,
                "raw_motion_px": safe_float(raw_distance_px),
                "in_roi": 1.0,
                "region_name": region_names[0],
                "regions": list(region_names),
            }
            speeds[track.track_id] = info
            self.previous_speed_info[track.track_id] = info
        return speeds

    def _track_region_names(self, track: Track) -> List[str]:
        if not self.regions:
            return []
        bbox_tuple = bbox_to_tuple(track.bbox)
        return [region.name for region in self.regions if region.shape.contains_bbox(bbox_tuple)]

    def _track_in_roi(self, track: Track) -> bool:
        return bool(self._track_region_names(track))

    def _stabilize_speed(
        self,
        track_id: int,
        instant_speed_kmh: float,
        motion_suppressed: bool,
        speed_source: str,
    ) -> Tuple[float, float, bool, int, str]:
        previous_info = self.previous_speed_info.get(track_id)
        previous_stable = safe_float(previous_info.get("stable_speed_kmh") if previous_info else 0.0)
        previous_count = self.speed_valid_counts.get(track_id, 0)

        if motion_suppressed or instant_speed_kmh <= 0.1:
            hold_count = self.speed_hold_counts.get(track_id, 0)
            if previous_stable > 0.1 and hold_count < self.speed_hold_frames:
                self.speed_hold_counts[track_id] = hold_count + 1
                return previous_stable, previous_stable / 3.6, True, previous_count, f"{speed_source}_hold"
            return 0.0, 0.0, False, previous_count, speed_source

        self.speed_hold_counts[track_id] = 0

        if (
            previous_stable > 0.1
            and self.speed_max_drop_ratio > 0.0
            and instant_speed_kmh < previous_stable * self.speed_max_drop_ratio
        ):
            return previous_stable, previous_stable / 3.6, True, previous_count, f"{speed_source}_transient_low_suppressed"

        history = self.stable_speed_histories.setdefault(track_id, [])
        history.append(instant_speed_kmh)
        if len(history) > self.speed_history_size:
            del history[: len(history) - self.speed_history_size]

        sample_count = previous_count + 1
        self.speed_valid_counts[track_id] = sample_count
        if sample_count < self.speed_warmup_frames:
            return 0.0, 0.0, False, sample_count, f"{speed_source}_warming_up"

        stable_speed_kmh = safe_float(statistics.median(history))
        return stable_speed_kmh, stable_speed_kmh / 3.6, stable_speed_kmh > 0.1, sample_count, speed_source

    def _clear_speed_state(self, track_id: int) -> None:
        self.previous_speed_points.pop(track_id, None)
        self.previous_speed_frame_indices.pop(track_id, None)
        self.previous_world_points.pop(track_id, None)
        self.previous_speed_info.pop(track_id, None)
        self.speed_valid_counts.pop(track_id, None)
        self.stable_speed_histories.pop(track_id, None)
        self.speed_hold_counts.pop(track_id, None)

    def _update_flow(self, tracks: Sequence[Track], active_track_ids: Optional[set] = None) -> None:
        active_ids = set(active_track_ids) if active_track_ids is not None else {track.track_id for track in tracks}
        for track in tracks:
            current = track.bbox.bottom_center
            previous = self.previous_points.get(track.track_id)
            if previous is not None:
                bbox_tuple = bbox_to_tuple(track.bbox)
                for region in self.regions:
                    if region.flow_counter is not None and region.shape.contains_bbox(bbox_tuple):
                        region.flow_counter.update(track.track_id, previous, current)
            self.previous_points[track.track_id] = current
            self.previous_frame_indices[track.track_id] = track.frame_index

        for track_id in list(self.previous_points):
            if track_id not in active_ids:
                del self.previous_points[track_id]
                self.previous_frame_indices.pop(track_id, None)
                self._clear_speed_state(track_id)

    def _compute_region_summaries(
        self,
        tracks: Sequence[Track],
        speed_by_track: Mapping[int, SpeedInfo],
    ) -> List[Dict[str, object]]:
        summaries: List[Dict[str, object]] = []
        for region in self.regions:
            vehicle_count = 0
            non_motor_count = 0
            pedestrian_count = 0
            weighted_count = 0.0
            occupied_area = 0.0
            track_ids: List[int] = []
            for track in tracks:
                if not region.shape.contains_bbox(bbox_to_tuple(track.bbox)):
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
                track_ids.append(track.track_id)
            total_count = vehicle_count + non_motor_count + pedestrian_count
            density = estimate_roi_density(total_count, region.shape.area)
            weighted_density = estimate_roi_density(weighted_count, region.shape.area)
            occupancy_ratio = min(1.0, occupied_area / region.shape.area) if region.shape.area > 0 else 0.0
            speed_stats = self._speed_stats(speed_by_track, region.name)
            low_speed_ratio = self._low_speed_ratio(speed_by_track, region.name)
            congestion_level, warning_active = region.congestion_detector.update(
                roi_vehicle_count=total_count,
                low_speed_ratio=low_speed_ratio,
                occupancy_ratio=occupancy_ratio,
            )
            status = self._congestion_status(weighted_density * 100000.0, warning_active, congestion_level)
            summaries.append({
                "name": region.name,
                "vehicle_count": vehicle_count,
                "non_motor_count": non_motor_count,
                "pedestrian_count": pedestrian_count,
                "total_count": total_count,
                "density": safe_float(density),
                "weighted_density": safe_float(weighted_density),
                "density_per_100k": safe_float(weighted_density * 100000.0),
                "occupancy_ratio": safe_float(occupancy_ratio),
                "roi_area": safe_float(region.shape.area),
                "flow_count": region.flow_counter.total_count if region.flow_counter is not None else 0,
                "low_speed_ratio": safe_float(low_speed_ratio),
                "congestion_level": congestion_level,
                "congestion_status": status,
                "warning_active": bool(warning_active),
                "track_ids": track_ids,
                **speed_stats,
            })
        return summaries

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

    def _speed_stats(self, speed_by_track: Mapping[int, SpeedInfo], region_name: Optional[str] = None) -> Dict[str, float]:
        roi_speed_infos = []
        for info in speed_by_track.values():
            if safe_float(info.get("in_roi")) <= 0.0:
                continue
            if region_name is not None and region_name not in list(info.get("regions") or []):
                continue
            roi_speed_infos.append(info)
        valid_infos = [info for info in roi_speed_infos if safe_float(info.get("speed_valid")) > 0.0]
        speeds_kmh = [safe_float(info.get("stable_speed_kmh", info.get("speed_kmh"))) for info in valid_infos]
        speeds_mps = [safe_float(info.get("speed_mps")) for info in valid_infos]
        speeds_px = [safe_float(info.get("speed_px_per_second")) for info in valid_infos]
        return {
            "mean_speed_kmh": sum(speeds_kmh) / len(speeds_kmh) if speeds_kmh else 0.0,
            "moving_mean_speed_kmh": sum(speeds_kmh) / len(speeds_kmh) if speeds_kmh else 0.0,
            "max_speed_kmh": max(speeds_kmh) if speeds_kmh else 0.0,
            "mean_speed_mps": sum(speeds_mps) / len(speeds_mps) if speeds_mps else 0.0,
            "max_speed_mps": max(speeds_mps) if speeds_mps else 0.0,
            "mean_speed_px_per_second": sum(speeds_px) / len(speeds_px) if speeds_px else 0.0,
            "speed_count": float(len(roi_speed_infos)),
            "moving_speed_count": float(len(speeds_kmh)),
        }

    def _low_speed_ratio(self, speed_by_track: Mapping[int, SpeedInfo], region_name: Optional[str] = None, threshold_px_per_second: float = 60.0) -> float:
        roi_speed_infos = []
        for info in speed_by_track.values():
            if safe_float(info.get("in_roi")) <= 0.0:
                continue
            if region_name is not None and region_name not in list(info.get("regions") or []):
                continue
            roi_speed_infos.append(info)
        valid_infos = [info for info in roi_speed_infos if safe_float(info.get("speed_valid")) > 0.0]
        if not valid_infos:
            return 0.0
        low_count = sum(1 for info in valid_infos if safe_float(info.get("speed_px_per_second")) < threshold_px_per_second)
        return low_count / len(valid_infos)

    def _congestion_status(self, weighted_density_per_100k: float, warning_active: bool, congestion_level: str) -> str:
        # Congestion state is rule-based. Density is still reported as a metric,
        # but a single vehicle in a small ROI should not become CONGESTED only
        # because density_per_100k crosses a display threshold.
        if congestion_level in {"congested", "severe"}:
            return "CONGESTED"
        if warning_active or congestion_level == "slow":
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
        region_summaries: Sequence[Mapping[str, object]],
    ) -> np.ndarray:
        for track in tracks:
            self._draw_track(frame, track, speed_by_track.get(track.track_id, {}))
        self._draw_regions(frame)
        self._draw_status_panel(frame, tracks, density_stats, speed_stats, congestion_status, region_summaries)
        return frame

    def _draw_track(self, frame: np.ndarray, track: Track, speed_info: Mapping[str, float]) -> None:
        color = color_for_class(track.cls_id)
        bbox = track.bbox
        p1 = int(round(bbox.x1)), int(round(bbox.y1))
        p2 = int(round(bbox.x2)), int(round(bbox.y2))
        cv2.rectangle(frame, p1, p2, color, 2, cv2.LINE_AA)
        in_roi = safe_float(speed_info.get("in_roi")) > 0.0
        speed_valid = safe_float(speed_info.get("speed_valid")) > 0.0
        speed_kmh = safe_float(speed_info.get("stable_speed_kmh", speed_info.get("speed_kmh")))
        label = f"ID {track.track_id} {track.label} {track.score:.2f}"
        if in_roi and speed_valid:
            label = f"{label} {speed_kmh:.1f} km/h"
        draw_label(frame, label, p1, color)

    def _draw_regions(self, frame: np.ndarray) -> None:
        if not self.regions:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        for index, region in enumerate(self.regions):
            color = region_color(index)
            points = np.array([(int(p.x), int(p.y)) for p in region.points], dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)
            if region.points:
                p0 = region.points[0]
                cv2.putText(frame, region.name, (int(p0.x) + 4, max(20, int(p0.y) - 8)), font, 0.6, color, 2, cv2.LINE_AA)

    def _draw_status_panel(
        self,
        frame: np.ndarray,
        tracks: Sequence[Track],
        density_stats: Mapping[str, float],
        speed_stats: Mapping[str, float],
        congestion_status: str,
        region_summaries: Sequence[Mapping[str, object]],
    ) -> None:
        status_color = {
            "CLEAR": (0, 220, 0),
            "WARNING": (0, 220, 255),
            "CONGESTED": (0, 0, 255),
        }.get(congestion_status, (255, 255, 255))
        status_cn = {"CLEAR": "畅通", "WARNING": "预警", "CONGESTED": "拥堵"}.get(congestion_status, "未知")
        speed_count = int(safe_float(speed_stats.get("speed_count")))
        moving_speed_count = int(safe_float(speed_stats.get("moving_speed_count")))
        moving_avg_speed_text = "-- km/h" if moving_speed_count <= 0 else f"{safe_float(speed_stats.get('moving_mean_speed_kmh')):.2f} km/h"
        lines = [
            f"目标总数：{len(tracks):d}",
            f"ROI内车辆(car/truck)：{int(safe_float(density_stats.get('vehicle_count')))}",
            f"总流量计数：{int(safe_float(density_stats.get('flow_count')))}",
            f"道路密度：{safe_float(density_stats.get('density')):.4f}",
            f"加权密度：{safe_float(density_stats.get('weighted_density')):.4f}",
            f"占用率：{safe_float(density_stats.get('occupancy_ratio')):.4f}",
            f"ROI内测速目标：{speed_count:d}",
            f"平均车速：{moving_avg_speed_text}",
            f"拥堵状态：{congestion_status} / {status_cn}",
        ]
        if len(region_summaries) > 1:
            for region in region_summaries[:4]:
                status = str(region.get("congestion_status", "CLEAR"))
                lines.append(
                    f"{region.get('name', 'ROI')}：{ {'CLEAR':'畅通','WARNING':'预警','CONGESTED':'拥堵'}.get(status, status)} "
                    f"车辆{int(safe_float(region.get('vehicle_count')))} "
                    f"均速{safe_float(region.get('moving_mean_speed_kmh')):.1f}km/h"
                )

        panel_x1, panel_y1 = 8, 8
        row_height = 27
        padding_x = 12
        padding_y = 10
        font = load_chinese_font(18)
        title_font = load_chinese_font(20)
        max_text_width = max(text_size(font, line)[0] for line in lines)
        panel_width = min(frame.shape[1] - panel_x1 - 2, max(680, max_text_width + padding_x * 2))
        panel_height = padding_y * 2 + row_height * len(lines)
        panel_x2 = min(frame.shape[1] - 1, panel_x1 + panel_width)
        panel_y2 = min(frame.shape[0] - 1, panel_y1 + panel_height)

        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), status_color, 1)

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        drawer = ImageDraw.Draw(pil_image)
        status_rgb = bgr_to_rgb(status_color)
        for index, line in enumerate(lines):
            is_status_line = "拥堵状态" in line
            fill = status_rgb if is_status_line else (255, 255, 255)
            drawer.text((panel_x1 + padding_x, panel_y1 + padding_y + index * row_height), line, font=title_font if is_status_line else font, fill=fill)
        frame[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)

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
        region_summaries: Sequence[Mapping[str, object]],
    ) -> Dict[str, object]:
        return {
            "frame_index": frame_index,
            "detected_this_frame": detected_this_frame,
            "detections_count": len(tracks),
            "detections": [track_to_dict(track, speed_by_track.get(track.track_id, {})) for track in tracks],
            "regions": [serialize_region(region) for region in region_summaries],
            "flow_count": int(safe_float(density_stats.get("flow_count"))),
            "vehicle_count": int(density_stats["vehicle_count"]),
            "non_motor_count": int(density_stats["non_motor_count"]),
            "pedestrian_count": int(density_stats["pedestrian_count"]),
            "density": round(float(density_stats["density"]), 10),
            "weighted_density": round(float(density_stats["weighted_density"]), 10),
            "density_per_100k": round(float(density_stats["density_per_100k"]), 6),
            "occupancy_ratio": round(float(density_stats["occupancy_ratio"]), 6),
            "roi_area": round(float(density_stats["roi_area"]), 3),
            "mean_speed_kmh": round(safe_float(speed_stats.get("mean_speed_kmh")), 6),
            "moving_mean_speed_kmh": round(safe_float(speed_stats.get("moving_mean_speed_kmh")), 6),
            "max_speed_kmh": round(safe_float(speed_stats.get("max_speed_kmh")), 6),
            "mean_speed_mps": round(safe_float(speed_stats.get("mean_speed_mps")), 6),
            "max_speed_mps": round(safe_float(speed_stats.get("max_speed_mps")), 6),
            "mean_speed_px_per_second": round(safe_float(speed_stats.get("mean_speed_px_per_second")), 6),
            "speed_count": int(safe_float(speed_stats.get("speed_count"))),
            "moving_speed_count": int(safe_float(speed_stats.get("moving_speed_count"))),
            "congestion_level": congestion_level,
            "congestion_status": congestion_status,
            "warning_active": warning_active,
        }



def parse_points(raw_points: object) -> List[Point]:
    if not isinstance(raw_points, Sequence) or isinstance(raw_points, (str, bytes)):
        raise ValueError("ROI region points must be a sequence of [x, y] pairs")
    points: List[Point] = []
    for raw in raw_points:
        if isinstance(raw, Point):
            points.append(raw)
        elif isinstance(raw, Mapping):
            points.append(Point(float(raw["x"]), float(raw["y"])))
        elif isinstance(raw, Sequence) and len(raw) == 2 and not isinstance(raw, (str, bytes)):
            points.append(Point(float(raw[0]), float(raw[1])))
        else:
            raise ValueError("ROI points must be Point, {'x','y'}, or [x, y]")
    if len(points) < 3:
        raise ValueError("ROI region requires at least three polygon points")
    return points


def aggregate_region_summaries(region_summaries: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    if not region_summaries:
        return {"vehicle_count": 0, "non_motor_count": 0, "pedestrian_count": 0, "density": 0.0, "weighted_density": 0.0, "density_per_100k": 0.0, "occupancy_ratio": 0.0, "roi_area": 0.0, "flow_count": 0, "mean_speed_kmh": 0.0, "moving_mean_speed_kmh": 0.0, "max_speed_kmh": 0.0, "mean_speed_mps": 0.0, "max_speed_mps": 0.0, "mean_speed_px_per_second": 0.0, "speed_count": 0.0, "moving_speed_count": 0.0, "congestion_level": "normal", "congestion_status": "CLEAR", "warning_active": False}
    vehicle_count = sum(int(safe_float(item.get("vehicle_count"))) for item in region_summaries)
    non_motor_count = sum(int(safe_float(item.get("non_motor_count"))) for item in region_summaries)
    pedestrian_count = sum(int(safe_float(item.get("pedestrian_count"))) for item in region_summaries)
    roi_area = sum(safe_float(item.get("roi_area")) for item in region_summaries)
    weighted_count = sum(safe_float(item.get("weighted_density")) * safe_float(item.get("roi_area")) for item in region_summaries)
    occupied_area = sum(safe_float(item.get("occupancy_ratio")) * safe_float(item.get("roi_area")) for item in region_summaries)
    flow_count = sum(int(safe_float(item.get("flow_count"))) for item in region_summaries)
    speed_count = sum(safe_float(item.get("speed_count")) for item in region_summaries)
    moving_speed_count = sum(safe_float(item.get("moving_speed_count")) for item in region_summaries)
    statuses = [str(item.get("congestion_status", "CLEAR")) for item in region_summaries]
    status = max(statuses, key=lambda value: {"CLEAR": 0, "WARNING": 1, "CONGESTED": 2}.get(value, 0))
    weighted_density = estimate_roi_density(weighted_count, roi_area)
    total_count = vehicle_count + non_motor_count + pedestrian_count
    return {
        "vehicle_count": vehicle_count,
        "non_motor_count": non_motor_count,
        "pedestrian_count": pedestrian_count,
        "density": estimate_roi_density(total_count, roi_area),
        "weighted_density": weighted_density,
        "density_per_100k": weighted_density * 100000.0,
        "occupancy_ratio": min(1.0, occupied_area / roi_area) if roi_area > 0.0 else 0.0,
        "roi_area": roi_area,
        "flow_count": flow_count,
        "mean_speed_kmh": weighted_average(region_summaries, "mean_speed_kmh", "moving_speed_count"),
        "moving_mean_speed_kmh": weighted_average(region_summaries, "moving_mean_speed_kmh", "moving_speed_count"),
        "max_speed_kmh": max(safe_float(item.get("max_speed_kmh")) for item in region_summaries),
        "mean_speed_mps": weighted_average(region_summaries, "mean_speed_mps", "moving_speed_count"),
        "max_speed_mps": max(safe_float(item.get("max_speed_mps")) for item in region_summaries),
        "mean_speed_px_per_second": weighted_average(region_summaries, "mean_speed_px_per_second", "moving_speed_count"),
        "speed_count": speed_count,
        "moving_speed_count": moving_speed_count,
        "congestion_level": "congested" if status == "CONGESTED" else "slow" if status == "WARNING" else "normal",
        "congestion_status": status,
        "warning_active": any(bool(item.get("warning_active")) for item in region_summaries),
    }


def weighted_average(items: Sequence[Mapping[str, object]], value_key: str, weight_key: str) -> float:
    total_weight = sum(safe_float(item.get(weight_key)) for item in items)
    if total_weight <= 0.0:
        return 0.0
    return sum(safe_float(item.get(value_key)) * safe_float(item.get(weight_key)) for item in items) / total_weight


def serialize_region(region: Mapping[str, object]) -> Dict[str, object]:
    return {
        "name": str(region.get("name", "ROI")),
        "vehicle_count": int(safe_float(region.get("vehicle_count"))),
        "non_motor_count": int(safe_float(region.get("non_motor_count"))),
        "pedestrian_count": int(safe_float(region.get("pedestrian_count"))),
        "total_count": int(safe_float(region.get("total_count"))),
        "flow_count": int(safe_float(region.get("flow_count"))),
        "density": round(safe_float(region.get("density")), 10),
        "weighted_density": round(safe_float(region.get("weighted_density")), 10),
        "density_per_100k": round(safe_float(region.get("density_per_100k")), 6),
        "occupancy_ratio": round(safe_float(region.get("occupancy_ratio")), 6),
        "roi_area": round(safe_float(region.get("roi_area")), 3),
        "mean_speed_kmh": round(safe_float(region.get("mean_speed_kmh")), 6),
        "moving_mean_speed_kmh": round(safe_float(region.get("moving_mean_speed_kmh")), 6),
        "max_speed_kmh": round(safe_float(region.get("max_speed_kmh")), 6),
        "speed_count": int(safe_float(region.get("speed_count"))),
        "moving_speed_count": int(safe_float(region.get("moving_speed_count"))),
        "low_speed_ratio": round(safe_float(region.get("low_speed_ratio")), 6),
        "congestion_level": str(region.get("congestion_level", "normal")),
        "congestion_status": str(region.get("congestion_status", "CLEAR")),
        "warning_active": bool(region.get("warning_active")),
        "track_ids": [int(item) for item in list(region.get("track_ids") or [])],
    }


def classify_track(track: Track) -> str:
    label = track.label.strip().lower()
    if label in {"motor vehicle", "vehicle", "car", "bus", "truck", "van"}:
        return "motor_vehicle"
    if label in {"non_motorized vehicle", "non-motorized vehicle", "bicycle", "motorcycle", "bike"}:
        return "non_motor"
    if label in {"pedestrian", "person"}:
        return "pedestrian"
    if track.cls_id in {2, 5, 7}:
        return "motor_vehicle"
    if track.cls_id in {1, 3}:
        return "non_motor"
    if track.cls_id == 0:
        return "pedestrian"
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
        "instant_speed_mps": round(safe_float(speed_info.get("instant_speed_mps", speed_info.get("speed_mps"))), 6),
        "instant_speed_kmh": round(safe_float(speed_info.get("instant_speed_kmh", speed_info.get("speed_kmh"))), 6),
        "stable_speed_kmh": round(safe_float(speed_info.get("stable_speed_kmh", speed_info.get("speed_kmh"))), 6),
        "speed_valid": bool(safe_float(speed_info.get("speed_valid"))),
        "speed_sample_count": int(safe_float(speed_info.get("speed_sample_count"))),
        "speed_source": str(speed_info.get("speed_source", "pixel_only")),
        "delta_frames": round(safe_float(speed_info.get("delta_frames"), 1.0), 3),
        "motion_suppressed": bool(safe_float(speed_info.get("motion_suppressed"))),
        "raw_motion_px": round(safe_float(speed_info.get("raw_motion_px")), 6),
        "in_roi": bool(safe_float(speed_info.get("in_roi"))),
        "region_name": str(speed_info.get("region_name", "")),
        "regions": [str(item) for item in list(speed_info.get("regions") or [])],
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




def region_color(index: int) -> Tuple[int, int, int]:
    palette = [(255, 180, 0), (0, 200, 120), (255, 120, 0), (220, 120, 255)]
    return palette[index % len(palette)]


def load_chinese_font(size: int) -> ImageFont.ImageFont:
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/arialuni.ttf",
    ]
    for font_path in font_paths:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def text_size(font: ImageFont.ImageFont, text: str) -> Tuple[int, int]:
    if hasattr(font, "getbbox"):
        left, top, right, bottom = font.getbbox(text)
        return int(right - left), int(bottom - top)
    width, height = font.getsize(text)
    return int(width), int(height)


def bgr_to_rgb(color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    blue, green, red = color
    return int(red), int(green), int(blue)

def safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default
















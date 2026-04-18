"""Speed estimation from tracked detection summaries.

The estimator prefers detection-level track IDs. If they are missing, it uses a
small IoU matcher so existing summary.json files can still be analyzed. When a
pixel-to-world homography is provided, speeds are also reported in km/h.
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration.homography import Homography, compute_homography, load_point_pairs

BBox = Tuple[float, float, float, float]
Point = Tuple[float, float]


@dataclass(frozen=True)
class SpeedRecord:
    frame_index: int
    track_id: int
    class_id: int
    label: str
    speed_px_per_frame: float
    speed_px_per_second: float
    speed_mps: float
    speed_kmh: float
    bbox: BBox

    def to_dict(self) -> Dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "track_id": self.track_id,
            "class_id": self.class_id,
            "label": self.label,
            "speed_px_per_frame": round(self.speed_px_per_frame, 6),
            "speed_px_per_second": round(self.speed_px_per_second, 6),
            "speed_mps": round(self.speed_mps, 6),
            "speed_kmh": round(self.speed_kmh, 6),
            "bbox": [round(value, 3) for value in self.bbox],
        }


@dataclass(frozen=True)
class SpeedFrameStats:
    frame_index: int
    vehicles: List[SpeedRecord]
    mean_speed_px_per_second: float
    mean_speed_kmh: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "frame_index": self.frame_index,
            "vehicles": [item.to_dict() for item in self.vehicles],
            "mean_speed_px_per_second": round(self.mean_speed_px_per_second, 6),
            "mean_speed_kmh": round(self.mean_speed_kmh, 6),
        }


class IoUTrackAssigner:
    """Assigns temporary track IDs when summary detections do not include them."""

    def __init__(self, iou_threshold: float = 0.3, max_missing: int = 10) -> None:
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.next_track_id = 1
        self.tracks: Dict[int, Dict[str, object]] = {}

    def update(self, detections: Sequence[Mapping[str, object]]) -> Dict[int, int]:
        assigned: Dict[int, int] = {}
        matched_track_ids = set()
        for index, detection in enumerate(detections):
            bbox = parse_bbox(detection.get("bbox"))
            if bbox is None:
                continue
            track_id = self.match(bbox, matched_track_ids)
            if track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
            self.tracks[track_id] = {"bbox": bbox, "missing": 0}
            matched_track_ids.add(track_id)
            assigned[index] = track_id

        stale_ids = []
        for track_id, state in self.tracks.items():
            if track_id in matched_track_ids:
                continue
            state["missing"] = safe_int(state.get("missing")) + 1
            if safe_int(state.get("missing")) > self.max_missing:
                stale_ids.append(track_id)
        for track_id in stale_ids:
            del self.tracks[track_id]
        return assigned

    def match(self, bbox: BBox, excluded_track_ids: set) -> Optional[int]:
        best_id = None
        best_iou = self.iou_threshold
        for track_id, state in self.tracks.items():
            if track_id in excluded_track_ids:
                continue
            previous_bbox = state.get("bbox")
            if not isinstance(previous_bbox, tuple):
                continue
            iou = bbox_iou(bbox, previous_bbox)
            if iou >= best_iou:
                best_iou = iou
                best_id = track_id
        return best_id


def estimate_speeds_from_frames(
    frames: Sequence[Mapping[str, object]],
    fps: float,
    iou_threshold: float = 0.3,
    homography: Optional[Homography] = None,
    min_motion_px_per_frame: float = 2.0,
) -> List[SpeedFrameStats]:
    """Estimate speed for each tracked object in every frame.

    If homography is provided, bottom-center bbox points are mapped to world
    coordinates in meters and speed_kmh is calculated. Otherwise speed_kmh is 0.
    """

    fps = safe_float(fps, 30.0)
    if fps <= 0.0:
        fps = 30.0
    min_motion_px_per_frame = max(0.0, safe_float(min_motion_px_per_frame, 2.0))

    assigner = IoUTrackAssigner(iou_threshold=iou_threshold)
    previous_pixel_points: Dict[int, Point] = {}
    previous_frame_indices: Dict[int, int] = {}
    previous_world_points: Dict[int, Point] = {}
    frame_stats: List[SpeedFrameStats] = []

    for frame in frames:
        detections = frame_detections(frame)
        fallback_ids = assigner.update(detections) if not detections_have_track_ids(detections) else {}
        records: List[SpeedRecord] = []

        for index, detection in enumerate(detections):
            bbox = parse_bbox(detection.get("bbox"))
            if bbox is None:
                continue
            track_id = extract_track_id(detection)
            if track_id is None:
                track_id = fallback_ids.get(index)
            if track_id is None:
                continue

            frame_index = safe_int(frame.get("frame_index"))
            current_pixel = bbox_bottom_center(bbox)
            previous_pixel = previous_pixel_points.get(track_id)
            previous_frame_index = previous_frame_indices.get(track_id)
            delta_frames = max(1, frame_index - previous_frame_index) if previous_frame_index is not None else 1
            delta_time = delta_frames / fps
            raw_distance_px = 0.0 if previous_pixel is None else distance(previous_pixel, current_pixel)
            raw_speed_px_per_frame = raw_distance_px / delta_frames
            motion_suppressed = previous_pixel is None or raw_speed_px_per_frame < min_motion_px_per_frame

            speed_px_per_frame = 0.0
            speed_mps = 0.0
            speed_kmh = 0.0
            if previous_pixel is None:
                previous_pixel_points[track_id] = current_pixel
                previous_frame_indices[track_id] = frame_index
                if homography is not None:
                    previous_world_points[track_id] = homography.pixel_to_world(current_pixel[0], current_pixel[1])
            elif not motion_suppressed:
                speed_px_per_frame = raw_speed_px_per_frame
                if homography is not None:
                    current_world = homography.pixel_to_world(current_pixel[0], current_pixel[1])
                    previous_world = previous_world_points.get(track_id)
                    if previous_world is not None and delta_time > 0.0:
                        speed_mps = distance(previous_world, current_world) / delta_time
                        speed_kmh = speed_mps * 3.6
                    previous_world_points[track_id] = current_world
                previous_pixel_points[track_id] = current_pixel
                previous_frame_indices[track_id] = frame_index

            records.append(
                SpeedRecord(
                    frame_index=frame_index,
                    track_id=track_id,
                    class_id=safe_int(detection.get("class_id"), -1),
                    label=str(detection.get("label", "")),
                    speed_px_per_frame=speed_px_per_frame,
                    speed_px_per_second=speed_px_per_frame * fps,
                    speed_mps=speed_mps,
                    speed_kmh=speed_kmh,
                    bbox=bbox,
                )
            )

        active_ids = {record.track_id for record in records}
        for track_id in list(previous_pixel_points):
            if track_id not in active_ids and detections:
                previous_pixel_points.pop(track_id, None)
                previous_frame_indices.pop(track_id, None)
                previous_world_points.pop(track_id, None)

        mean_px_speed = sum(record.speed_px_per_second for record in records) / len(records) if records else 0.0
        mean_kmh = sum(record.speed_kmh for record in records) / len(records) if records else 0.0
        frame_stats.append(
            SpeedFrameStats(
                frame_index=safe_int(frame.get("frame_index")),
                vehicles=records,
                mean_speed_px_per_second=mean_px_speed,
                mean_speed_kmh=mean_kmh,
            )
        )
    return frame_stats


def load_summary(path: Path) -> Tuple[List[Mapping[str, object]], Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data, {}
    if not isinstance(data, dict):
        raise ValueError("summary JSON must be an object or a list")
    frames = data.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError("summary JSON field 'frames' must be a list")
    return frames, data


def frame_detections(frame: Mapping[str, object]) -> List[Mapping[str, object]]:
    detections = frame.get("detections", [])
    if not isinstance(detections, list):
        return []
    return [item for item in detections if isinstance(item, dict)]


def detections_have_track_ids(detections: Sequence[Mapping[str, object]]) -> bool:
    return any(extract_track_id(detection) is not None for detection in detections)


def extract_track_id(detection: Mapping[str, object]) -> Optional[int]:
    for key in ("track_id", "id", "track"):
        if key in detection:
            value = safe_int(detection.get(key), -1)
            if value >= 0:
                return value
    return None


def parse_bbox(value: object) -> Optional[BBox]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    x1, y1, x2, y2 = [safe_float(item) for item in value]
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_bottom_center(bbox: BBox) -> Point:
    return (bbox[0] + bbox[2]) / 2.0, bbox[3]


def bbox_iou(first: BBox, second: BBox) -> float:
    inter_x1 = max(first[0], second[0])
    inter_y1 = max(first[1], second[1])
    inter_x2 = min(first[2], second[2])
    inter_y2 = min(first[3], second[3])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union_area = first_area + second_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def build_homography_from_args(args: argparse.Namespace) -> Optional[Homography]:
    if args.calibration:
        pixel_points, world_points = load_point_pairs(args.calibration)
        return compute_homography(pixel_points, world_points)
    if args.pixel_points and args.world_points:
        pixel_points = parse_points_json(args.pixel_points, "pixel_points")
        world_points = parse_points_json(args.world_points, "world_points")
        return compute_homography(pixel_points, world_points)
    return None


def parse_points_json(value: str, name: str) -> List[Point]:
    data = json.loads(value)
    if not isinstance(data, list):
        raise ValueError(f"{name} must be a JSON list")
    points = []
    for point in data:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"{name} points must be [x, y]")
        points.append((safe_float(point[0]), safe_float(point[1])))
    return points


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def safe_int(value: object, default: int = 0) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(float(result)) else default


def write_json(path: Path, stats: Sequence[SpeedFrameStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"frames": [item.to_dict() for item in stats]}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, stats: Sequence[SpeedFrameStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame_index",
                "track_id",
                "class_id",
                "label",
                "speed_px_per_frame",
                "speed_px_per_second",
                "speed_mps",
                "speed_kmh",
                "bbox",
            ],
        )
        writer.writeheader()
        for frame in stats:
            for record in frame.vehicles:
                row = record.to_dict()
                row["bbox"] = json.dumps(row["bbox"])
                writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate object speeds from detection summary JSON.")
    parser.add_argument("--summary", required=True, help="Input summary.json path")
    parser.add_argument("--output", default="outputs/realtime/speed.json", help="Output JSON or CSV path")
    parser.add_argument("--format", choices=("json", "csv"), default="json", help="Output format")
    parser.add_argument("--fps", type=float, default=None, help="Video FPS. Defaults to summary fps or 30")
    parser.add_argument("--iou-threshold", type=float, default=0.3, help="IoU threshold for fallback tracking")
    parser.add_argument("--min-motion-px-per-frame", type=float, default=2.0, help="Suppress bbox jitter below this pixel/frame speed")
    parser.add_argument("--calibration", default=None, help="JSON file with pixel_points and world_points")
    parser.add_argument("--pixel-points", default=None, help='JSON pixel points, e.g. "[[100,500],[500,500],[500,700],[100,700]]"')
    parser.add_argument("--world-points", default=None, help='JSON world meter points, e.g. "[[0,0],[10,0],[10,20],[0,20]]"')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames, metadata = load_summary(Path(args.summary))
    fps = safe_float(args.fps, 0.0) if args.fps is not None else safe_float(metadata.get("fps"), 30.0)
    if fps <= 0.0:
        fps = 30.0
    homography = build_homography_from_args(args)
    stats = estimate_speeds_from_frames(
        frames,
        fps=fps,
        iou_threshold=args.iou_threshold,
        homography=homography,
        min_motion_px_per_frame=args.min_motion_px_per_frame,
    )
    output = Path(args.output)
    if args.format == "json":
        write_json(output, stats)
    else:
        write_csv(output, stats)
    print(f"output: {output.resolve()}")
    print(f"frames: {len(stats)}")
    print(f"fps: {fps}")
    print(f"homography: {'enabled' if homography is not None else 'disabled'}")


if __name__ == "__main__":
    main()


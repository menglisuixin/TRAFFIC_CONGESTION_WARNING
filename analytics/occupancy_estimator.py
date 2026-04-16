"""ROI occupancy estimation from detection summaries.

This module can be imported by realtime pipelines or run as a CLI tool:

    python analytics/occupancy_estimator.py --summary outputs/realtime/summary.json \
        --output outputs/realtime/occupancy.json --format json
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]

MOTOR_LABELS = {"motor vehicle", "vehicle", "car", "bus", "truck", "van"}
NON_MOTOR_LABELS = {"non_motorized vehicle", "non-motorized vehicle", "bicycle", "motorcycle", "bike"}
PEDESTRIAN_LABELS = {"pedestrian", "person"}


@dataclass(frozen=True)
class ROI:
    """Polygon ROI with area and containment helpers."""

    points: List[Point]

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError("ROI requires at least three points")
        if self.area <= 0.0:
            raise ValueError("ROI area must be greater than zero")

    @classmethod
    def from_rect(cls, rect: Sequence[float]) -> "ROI":
        if len(rect) != 4:
            raise ValueError("rect ROI must be [x1, y1, x2, y2]")
        x1, y1, x2, y2 = [safe_float(value) for value in rect]
        return cls([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    @property
    def area(self) -> float:
        return polygon_area(self.points)

    def contains_point(self, point: Point) -> bool:
        return point_in_polygon(point, self.points)

    def contains_bbox(self, bbox: BBox) -> bool:
        return self.contains_point(bbox_bottom_center(bbox))


@dataclass(frozen=True)
class OccupancyFrameStats:
    frame_index: int
    vehicle_count: int
    non_motor_count: int
    pedestrian_count: int
    occupied_area: float
    roi_area: float
    occupancy_ratio: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "frame_index": self.frame_index,
            "vehicle_count": self.vehicle_count,
            "non_motor_count": self.non_motor_count,
            "pedestrian_count": self.pedestrian_count,
            "occupied_area": round(self.occupied_area, 3),
            "roi_area": round(self.roi_area, 3),
            "occupancy_ratio": round(self.occupancy_ratio, 6),
        }


def estimate_occupancy_ratio(detections: Sequence[Mapping[str, object]], roi: ROI, include_pedestrians: bool = False) -> float:
    """Return occupied bbox area divided by ROI area.

    A bbox is counted when its bottom-center point is inside the ROI.
    """

    stats = compute_frame_occupancy({"frame_index": 0, "detections": list(detections)}, roi, include_pedestrians)
    return stats.occupancy_ratio


def compute_frame_occupancy(
    frame: Mapping[str, object],
    roi: ROI,
    include_pedestrians: bool = False,
) -> OccupancyFrameStats:
    vehicle_count = 0
    non_motor_count = 0
    pedestrian_count = 0
    occupied_area = 0.0

    for detection in frame_detections(frame):
        bbox = parse_bbox(detection.get("bbox"))
        if bbox is None or not roi.contains_bbox(bbox):
            continue
        category = classify_detection(detection)
        if category == "motor":
            vehicle_count += 1
        elif category == "non_motor":
            non_motor_count += 1
        elif category == "pedestrian":
            pedestrian_count += 1
            if not include_pedestrians:
                continue
        else:
            continue
        occupied_area += bbox_area(bbox)

    occupancy_ratio = 0.0 if roi.area <= 0.0 else min(1.0, occupied_area / roi.area)
    return OccupancyFrameStats(
        frame_index=safe_int(frame.get("frame_index")),
        vehicle_count=vehicle_count,
        non_motor_count=non_motor_count,
        pedestrian_count=pedestrian_count,
        occupied_area=occupied_area,
        roi_area=roi.area,
        occupancy_ratio=occupancy_ratio,
    )


def compute_occupancy_summary(
    frames: Sequence[Mapping[str, object]],
    roi: ROI,
    include_pedestrians: bool = False,
) -> List[OccupancyFrameStats]:
    return [compute_frame_occupancy(frame, roi, include_pedestrians) for frame in frames]


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


def build_roi(roi_arg: Optional[str], frames: Sequence[Mapping[str, object]], metadata: Mapping[str, object]) -> ROI:
    if roi_arg:
        raw = load_roi_arg(roi_arg)
        if isinstance(raw, dict) and "rect" in raw:
            return ROI.from_rect(normalize_rect(raw["rect"], metadata))
        if isinstance(raw, dict) and "polygon" in raw:
            return ROI(normalize_points(raw["polygon"], metadata))
        if isinstance(raw, list):
            return ROI(normalize_points(raw, metadata))
        raise ValueError("ROI must be {'rect': [...]} or {'polygon': [[...], ...]}")

    meta_roi = metadata.get("roi")
    if isinstance(meta_roi, dict) and isinstance(meta_roi.get("points"), list):
        return ROI(normalize_points(meta_roi["points"], metadata))

    width = safe_float(metadata.get("width"), 0.0)
    height = safe_float(metadata.get("height"), 0.0)
    if width > 0.0 and height > 0.0:
        return ROI.from_rect([0.0, 0.0, width, height])

    width, height = infer_frame_extent(frames)
    return ROI.from_rect([0.0, 0.0, width, height])


def load_roi_arg(value: str):
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def normalize_rect(rect: object, metadata: Mapping[str, object]) -> List[float]:
    if not isinstance(rect, list) and not isinstance(rect, tuple):
        raise ValueError("rect ROI must be [x1, y1, x2, y2]")
    if len(rect) != 4:
        raise ValueError("rect ROI must be [x1, y1, x2, y2]")
    values = [safe_float(value) for value in rect]
    width = safe_float(metadata.get("width"), 0.0)
    height = safe_float(metadata.get("height"), 0.0)
    if width > 0.0 and height > 0.0 and all(0.0 <= value <= 1.0 for value in values):
        return [values[0] * width, values[1] * height, values[2] * width, values[3] * height]
    return values


def normalize_points(points: object, metadata: Mapping[str, object]) -> List[Point]:
    if not isinstance(points, list):
        raise ValueError("polygon ROI must be a list of [x, y] points")
    parsed = []
    flat_values = []
    for point in points:
        if not isinstance(point, list) and not isinstance(point, tuple):
            raise ValueError("polygon point must be [x, y]")
        if len(point) != 2:
            raise ValueError("polygon point must be [x, y]")
        x = safe_float(point[0])
        y = safe_float(point[1])
        parsed.append((x, y))
        flat_values.extend([x, y])
    width = safe_float(metadata.get("width"), 0.0)
    height = safe_float(metadata.get("height"), 0.0)
    if width > 0.0 and height > 0.0 and flat_values and all(0.0 <= value <= 1.0 for value in flat_values):
        return [(x * width, y * height) for x, y in parsed]
    return parsed


def frame_detections(frame: Mapping[str, object]) -> List[Mapping[str, object]]:
    detections = frame.get("detections", [])
    if not isinstance(detections, list):
        return []
    return [item for item in detections if isinstance(item, dict)]


def parse_bbox(value: object) -> Optional[BBox]:
    if not isinstance(value, list) and not isinstance(value, tuple):
        return None
    if len(value) != 4:
        return None
    x1, y1, x2, y2 = [safe_float(item) for item in value]
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def classify_detection(detection: Mapping[str, object]) -> str:
    label = str(detection.get("label", "")).strip().lower()
    class_id = detection.get("class_id")
    if label in MOTOR_LABELS:
        return "motor"
    if label in NON_MOTOR_LABELS:
        return "non_motor"
    if label in PEDESTRIAN_LABELS:
        return "pedestrian"
    if class_id in {0, 2, 5, 7}:
        return "motor"
    if class_id == 1:
        return "non_motor"
    return "other"


def bbox_bottom_center(bbox: BBox) -> Point:
    return (bbox[0] + bbox[2]) / 2.0, bbox[3]


def bbox_area(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def infer_frame_extent(frames: Sequence[Mapping[str, object]]) -> Point:
    max_x = 1.0
    max_y = 1.0
    for frame in frames:
        for detection in frame_detections(frame):
            bbox = parse_bbox(detection.get("bbox"))
            if bbox is None:
                continue
            max_x = max(max_x, bbox[2])
            max_y = max(max_y, bbox[3])
    return max_x, max_y


def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, current in enumerate(polygon):
        previous = polygon[j]
        if point_on_segment(point, previous, current):
            return True
        xi, yi = current
        xj, yj = previous
        if (yi > y) != (yj > y):
            x_intersection = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersection:
                inside = not inside
        j = i
    return inside


def point_on_segment(point: Point, start: Point, end: Point) -> bool:
    x, y = point
    x1, y1 = start
    x2, y2 = end
    cross = (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)
    if abs(cross) > 1e-9:
        return False
    return min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9


def polygon_area(points: Sequence[Point]) -> float:
    area = 0.0
    previous = points[-1]
    for current in points:
        area += previous[0] * current[1] - current[0] * previous[1]
        previous = current
    return abs(area) / 2.0


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


def write_json(path: Path, stats: Sequence[OccupancyFrameStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"frames": [item.to_dict() for item in stats]}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, stats: Sequence[OccupancyFrameStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(OccupancyFrameStats(0, 0, 0, 0, 0, 1, 0).to_dict().keys()))
        writer.writeheader()
        for item in stats:
            writer.writerow(item.to_dict())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate ROI occupancy from detection summary JSON.")
    parser.add_argument("--summary", required=True, help="Input summary.json path")
    parser.add_argument("--output", default="outputs/realtime/occupancy.json", help="Output JSON or CSV path")
    parser.add_argument("--format", choices=("json", "csv"), default="json", help="Output format")
    parser.add_argument("--roi", default=None, help='Optional ROI JSON string/file, e.g. {"rect":[0,0,1280,720]}')
    parser.add_argument("--include-pedestrians", action="store_true", help="Include pedestrian bbox area in occupancy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames, metadata = load_summary(Path(args.summary))
    roi = build_roi(args.roi, frames, metadata)
    stats = compute_occupancy_summary(frames, roi, args.include_pedestrians)
    output = Path(args.output)
    if args.format == "json":
        write_json(output, stats)
    else:
        write_csv(output, stats)
    print(f"output: {output.resolve()}")
    print(f"frames: {len(stats)}")
    print(f"roi_area: {roi.area:.3f}")


if __name__ == "__main__":
    main()


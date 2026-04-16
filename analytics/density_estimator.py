"""Density estimation utilities and CLI for detection summaries.

The module can be imported by realtime pipelines or run directly against a
summary.json produced by diagnostic/realtime scripts.
"""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

PointTuple = Tuple[float, float]
BBoxTuple = Tuple[float, float, float, float]

DEFAULT_CLASS_WEIGHTS = {
    "motor_vehicle": 1.0,
    "non_motor": 0.5,
    "pedestrian": 0.2,
}


@dataclass(frozen=True)
class DensityFrameStats:
    """Density statistics for one frame."""

    frame_index: int
    vehicle_count: int
    non_motor_count: int
    pedestrian_count: int
    density: float
    weighted_density: float
    roi_area: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "frame_index": self.frame_index,
            "vehicle_count": self.vehicle_count,
            "non_motor_count": self.non_motor_count,
            "pedestrian_count": self.pedestrian_count,
            "density": self.density,
            "weighted_density": self.weighted_density,
            "roi_area": self.roi_area,
        }


class ROIShape:
    """Polygon or rectangle ROI used for containment and area calculations."""

    def __init__(self, points: Sequence[PointTuple]) -> None:
        if len(points) < 3:
            raise ValueError("ROI requires at least three points")
        self.points = [(float(x), float(y)) for x, y in points]
        area = polygon_area(self.points)
        if area <= 0.0:
            raise ValueError("ROI area must be greater than zero")
        self.area = area

    @classmethod
    def from_rect(cls, rect: Sequence[float]) -> "ROIShape":
        if len(rect) != 4:
            raise ValueError("Rectangle ROI must be [x1, y1, x2, y2]")
        x1, y1, x2, y2 = [float(value) for value in rect]
        return cls([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    def contains_point(self, point: PointTuple) -> bool:
        return point_in_polygon(point, self.points)

    def contains_bbox(self, bbox: BBoxTuple) -> bool:
        return self.contains_point(bbox_bottom_center(bbox))


def estimate_density(vehicle_count: int, frame_width: float, frame_height: float) -> float:
    """Estimate density as vehicles per 100000 pixels of frame area.

    This small function is intentionally stable for callers such as
    scripts/realtime_traffic_analysis.py.
    """

    area_units = max(1.0, (float(frame_width) * float(frame_height)) / 100000.0)
    return float(vehicle_count) / area_units


def estimate_roi_density(vehicle_count: int, roi_area: float) -> float:
    """Estimate density as object count divided by ROI area."""

    if roi_area <= 0.0:
        return 0.0
    return float(vehicle_count) / float(roi_area)


def compute_frame_density(
    frame: Mapping[str, object],
    roi: ROIShape,
    class_weights: Optional[Mapping[str, float]] = None,
) -> DensityFrameStats:
    """Compute per-class counts and density for one summary frame."""

    weights = dict(DEFAULT_CLASS_WEIGHTS)
    if class_weights:
        weights.update({key: float(value) for key, value in class_weights.items()})

    vehicle_count = 0
    non_motor_count = 0
    pedestrian_count = 0
    weighted_count = 0.0

    for detection in get_frame_detections(frame):
        bbox = parse_bbox(detection.get("bbox"))
        if bbox is None or not roi.contains_bbox(bbox):
            continue

        category = classify_detection(detection)
        if category == "motor_vehicle":
            vehicle_count += 1
        elif category == "non_motor":
            non_motor_count += 1
        elif category == "pedestrian":
            pedestrian_count += 1
        else:
            continue
        weighted_count += weights.get(category, 1.0)

    total_count = vehicle_count + non_motor_count + pedestrian_count
    return DensityFrameStats(
        frame_index=int(frame.get("frame_index", 0)),
        vehicle_count=vehicle_count,
        non_motor_count=non_motor_count,
        pedestrian_count=pedestrian_count,
        density=round(estimate_roi_density(total_count, roi.area), 10),
        weighted_density=round(estimate_roi_density(weighted_count, roi.area), 10),
        roi_area=round(roi.area, 3),
    )


def compute_density_summary(
    frames: Sequence[Mapping[str, object]],
    roi: ROIShape,
    class_weights: Optional[Mapping[str, float]] = None,
) -> List[DensityFrameStats]:
    return [compute_frame_density(frame, roi, class_weights) for frame in frames]


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


def build_roi(
    roi_arg: Optional[str],
    frames: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object],
) -> ROIShape:
    if roi_arg:
        roi_data = load_roi_arg(roi_arg)
        if "polygon" in roi_data:
            return ROIShape(to_points(roi_data["polygon"]))
        if "rect" in roi_data:
            return ROIShape.from_rect(roi_data["rect"])
        if isinstance(roi_data, list):
            return ROIShape(to_points(roi_data))
        raise ValueError("ROI must contain 'polygon' or 'rect'")

    width = metadata.get("width")
    height = metadata.get("height")
    if width and height:
        return ROIShape.from_rect([0.0, 0.0, float(width), float(height)])

    inferred_width, inferred_height = infer_extent_from_detections(frames)
    return ROIShape.from_rect([0.0, 0.0, inferred_width, inferred_height])


def load_roi_arg(value: str):
    path = Path(value)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(value)


def to_points(values: object) -> List[PointTuple]:
    if not isinstance(values, list):
        raise ValueError("polygon ROI must be a list of [x, y] points")
    points: List[PointTuple] = []
    for item in values:
        if not isinstance(item, list) and not isinstance(item, tuple):
            raise ValueError("polygon point must be [x, y]")
        if len(item) != 2:
            raise ValueError("polygon point must have two values")
        points.append((float(item[0]), float(item[1])))
    return points


def infer_extent_from_detections(frames: Sequence[Mapping[str, object]]) -> Tuple[float, float]:
    max_x = 1.0
    max_y = 1.0
    for frame in frames:
        for detection in get_frame_detections(frame):
            bbox = parse_bbox(detection.get("bbox"))
            if bbox is None:
                continue
            max_x = max(max_x, bbox[2])
            max_y = max(max_y, bbox[3])
    return max_x, max_y


def get_frame_detections(frame: Mapping[str, object]) -> List[Mapping[str, object]]:
    detections = frame.get("detections", [])
    if isinstance(detections, list):
        return [item for item in detections if isinstance(item, dict)]
    return []


def parse_bbox(value: object) -> Optional[BBoxTuple]:
    if not isinstance(value, list) and not isinstance(value, tuple):
        return None
    if len(value) != 4:
        return None
    x1, y1, x2, y2 = [float(item) for item in value]
    return x1, y1, x2, y2


def classify_detection(detection: Mapping[str, object]) -> str:
    label = str(detection.get("label", "")).strip().lower()
    class_id = detection.get("class_id")

    if class_id == 0 or label in {"motor vehicle", "car", "bus", "truck", "vehicle"}:
        return "motor_vehicle"
    if class_id == 1 or label in {"non_motorized vehicle", "non-motorized vehicle", "bicycle", "motorcycle"}:
        return "non_motor"
    if class_id == 2 or label in {"pedestrian", "person"}:
        return "pedestrian"
    return "other"


def bbox_bottom_center(bbox: BBoxTuple) -> PointTuple:
    x1, _y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, y2


def point_in_polygon(point: PointTuple, polygon: Sequence[PointTuple]) -> bool:
    x, y = point
    inside = False
    previous_index = len(polygon) - 1
    for index, current in enumerate(polygon):
        previous = polygon[previous_index]
        if point_on_segment(point, previous, current):
            return True
        xi, yi = current
        xj, yj = previous
        intersects = (yi > y) != (yj > y)
        if intersects:
            x_intersection = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersection:
                inside = not inside
        previous_index = index
    return inside


def point_on_segment(point: PointTuple, start: PointTuple, end: PointTuple) -> bool:
    x, y = point
    x1, y1 = start
    x2, y2 = end
    cross = (y - y1) * (x2 - x1) - (x - x1) * (y2 - y1)
    if abs(cross) > 1e-9:
        return False
    return min(x1, x2) - 1e-9 <= x <= max(x1, x2) + 1e-9 and min(y1, y2) - 1e-9 <= y <= max(y1, y2) + 1e-9


def polygon_area(points: Sequence[PointTuple]) -> float:
    area = 0.0
    previous = points[-1]
    for current in points:
        area += previous[0] * current[1] - current[0] * previous[1]
        previous = current
    return abs(area) / 2.0


def write_json(path: Path, stats: Sequence[DensityFrameStats]) -> None:
    payload = {"frames": [item.to_dict() for item in stats]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, stats: Sequence[DensityFrameStats]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "frame_index",
                "vehicle_count",
                "non_motor_count",
                "pedestrian_count",
                "density",
                "weighted_density",
                "roi_area",
            ],
        )
        writer.writeheader()
        for item in stats:
            writer.writerow(item.to_dict())


def parse_class_weights(value: Optional[str]) -> Optional[Dict[str, float]]:
    if not value:
        return None
    raw = json.loads(value)
    if not isinstance(raw, dict):
        raise ValueError("class weights must be a JSON object")
    return {str(key): float(val) for key, val in raw.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate traffic density from detection summary JSON.")
    parser.add_argument("--summary", required=True, help="Path to summary.json containing per-frame detections")
    parser.add_argument("--output", default="outputs/realtime/density_summary.json", help="Output JSON or CSV path")
    parser.add_argument("--format", choices=("json", "csv"), default="json", help="Output format")
    parser.add_argument(
        "--roi",
        default=None,
        help=(
            "Optional ROI JSON string or JSON file. Examples: "
            "'{\"rect\":[0,0,1280,720]}' or '{\"polygon\":[[0,0],[1,0],[1,1],[0,1]]}'"
        ),
    )
    parser.add_argument(
        "--class-weights",
        default=None,
        help='Optional JSON object, e.g. "{\"motor_vehicle\":1.0,\"non_motor\":0.5,\"pedestrian\":0.2}"',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary)
    output_path = Path(args.output)

    frames, metadata = load_summary(summary_path)
    roi = build_roi(args.roi, frames, metadata)
    class_weights = parse_class_weights(args.class_weights)
    stats = compute_density_summary(frames, roi, class_weights)

    if args.format == "json":
        write_json(output_path, stats)
    else:
        write_csv(output_path, stats)

    print(f"output: {output_path.resolve()}")
    print(f"frames: {len(stats)}")
    print(f"roi_area: {roi.area:.3f}")


if __name__ == "__main__":
    main()

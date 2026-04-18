"""Realtime traffic analysis with YOLOv5 detection, density estimation, and MP4 output.

This script is designed for Windows PowerShell and the conda traffic_warn
environment. It does not modify YOLOv5 detect.py.
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analytics.congestion_detector import CongestionDetector
from analytics.density_estimator import ROIShape, classify_detection, estimate_roi_density
from analytics.flow_counter import FlowCounter, LineSegment
from core.types import BBox, Detection, Point
from detector.yolov5_detector import YOLOv5Detector


CLASS_WEIGHTS = {
    "motor_vehicle": 1.0,
    "non_motor": 0.5,
    "pedestrian": 0.2,
}


@dataclass
class TrackedDetection:
    track_id: int
    detection: Detection
    missing: int = 0


@dataclass(frozen=True)
class DensityStats:
    vehicle_count: int
    non_motor_count: int
    pedestrian_count: int
    density: float
    weighted_density: float
    roi_area: float
    occupancy_ratio: float

    @property
    def weighted_density_per_100k(self) -> float:
        return self.weighted_density * 100000.0


class SimpleIoUTracker:
    """Small IoU tracker used only to support flow counting in this script."""

    def __init__(self, iou_threshold: float = 0.3, max_missing: int = 15) -> None:
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self._next_track_id = 1
        self._tracks: Dict[int, TrackedDetection] = {}

    def update(self, detections: Sequence[Detection]) -> List[TrackedDetection]:
        matched_ids = set()
        current_tracks: List[TrackedDetection] = []

        for detection in detections:
            track_id = self._match(detection, matched_ids)
            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1

            tracked = TrackedDetection(track_id=track_id, detection=detection, missing=0)
            self._tracks[track_id] = tracked
            matched_ids.add(track_id)
            current_tracks.append(tracked)

        stale_ids: List[int] = []
        for track_id, tracked in self._tracks.items():
            if track_id in matched_ids:
                continue
            tracked.missing += 1
            if tracked.missing > self.max_missing:
                stale_ids.append(track_id)
        for track_id in stale_ids:
            del self._tracks[track_id]

        return current_tracks

    def _match(self, detection: Detection, excluded_ids: Iterable[int]) -> Optional[int]:
        excluded = set(excluded_ids)
        best_id: Optional[int] = None
        best_iou = self.iou_threshold
        for track_id, tracked in self._tracks.items():
            if track_id in excluded:
                continue
            iou = bbox_iou(detection.bbox, tracked.detection.bbox)
            if iou >= best_iou:
                best_id = track_id
                best_iou = iou
        return best_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime traffic analysis with YOLOv5.")
    parser.add_argument("--weights", required=True, help="YOLOv5 weights path, e.g. yolov5s.pt or weights/best.pt")
    parser.add_argument("--source", required=True, help="Input video path or camera index such as 0")
    parser.add_argument("--img-size", type=int, default=640, help="YOLOv5 input image size")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS")
    parser.add_argument("--output-dir", default="outputs/realtime", help="Output directory for video and summary.json")
    parser.add_argument("--device", default="cuda", help="Inference device, e.g. 0, cuda, or cpu")
    parser.add_argument("--every-n", type=int, default=1, help="Run detection every N frames")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="YOLOv5 NMS IoU threshold")
    parser.add_argument("--classes", default="2,7", help="COCO class ids to detect. Default: 2,7 for car,truck")
    parser.add_argument("--show", action="store_true", help="Show realtime preview window")
    parser.add_argument(
        "--roi",
        default=None,
        help=(
            "Optional ROI JSON string or JSON file. Examples: "
            "'{\"rect\":[100,100,1100,700]}' or "
            "'{\"polygon\":[[100,200],[1100,200],[1200,700],[50,700]]}'. "
            "Coordinates in [0,1] are treated as normalized."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.every_n <= 0:
        raise ValueError("--every-n must be greater than zero")
    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "traffic_analysis.mp4"
    summary_json = output_dir / "summary.json"

    capture = cv2.VideoCapture(parse_source(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    ok, first_frame = capture.read()
    if not ok or first_frame is None:
        capture.release()
        raise RuntimeError(f"Could not read first frame from source: {args.source}")

    height, width = first_frame.shape[:2]
    writer = create_video_writer(output_video, args.fps, width, height)

    detector = YOLOv5Detector(
        weights=args.weights,
        device=args.device,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        classes=parse_class_ids(args.classes),
    )
    tracker = SimpleIoUTracker(iou_threshold=0.3, max_missing=max(5, args.every_n * 3))
    roi_shape = build_roi(args.roi, width, height)
    roi_points = [Point(x, y) for x, y in roi_shape.points]
    flow_line = LineSegment(roi_points[0], roi_points[1])
    flow_counter = FlowCounter(flow_line, direction="any")
    congestion_detector = CongestionDetector()

    previous_points: Dict[int, Point] = {}
    last_detections: List[Detection] = []
    summaries: List[Dict[str, object]] = []
    frame_index = 0

    try:
        frame = first_frame
        while True:
            detected_this_frame = frame_index % args.every_n == 0
            if detected_this_frame:
                last_detections = detector.infer(frame)

            tracked = tracker.update(last_detections)
            low_speed_ratio = estimate_low_speed_ratio(tracked, previous_points)
            update_flow_counter(flow_counter, tracked, previous_points)

            density_stats = compute_density_stats([item.detection for item in tracked], roi_shape)
            congestion_level, warning_active = congestion_detector.update(
                roi_vehicle_count=density_stats.vehicle_count + density_stats.non_motor_count,
                low_speed_ratio=low_speed_ratio,
                occupancy_ratio=density_stats.occupancy_ratio,
            )
            congestion_status, congestion_text = density_congestion_status(
                density_stats.weighted_density_per_100k,
                warning_active,
                congestion_level,
            )

            annotated = draw_frame(
                frame.copy(),
                tracked,
                roi_points,
                flow_counter.total_count,
                density_stats,
                congestion_status,
                congestion_text,
            )
            writer.write(annotated)

            summaries.append(
                build_frame_summary(
                    frame_index=frame_index,
                    detections=last_detections,
                    detected_this_frame=detected_this_frame,
                    flow_count=flow_counter.total_count,
                    density_stats=density_stats,
                    congestion_level=congestion_level,
                    congestion_status=congestion_status,
                    congestion_text=congestion_text,
                    warning_active=warning_active,
                )
            )

            if args.show:
                cv2.imshow("realtime_traffic_analysis", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
            ok, frame = capture.read()
            if not ok or frame is None:
                break
    finally:
        capture.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    result = {
        "source": args.source,
        "weights": args.weights,
        "output_video": str(output_video),
        "fps": args.fps,
        "width": width,
        "height": height,
        "img_size": args.img_size,
        "conf_thres": args.conf_thres,
        "iou_thres": args.iou_thres,
        "every_n": args.every_n,
        "roi": {
            "points": [[round(x, 3), round(y, 3)] for x, y in roi_shape.points],
            "area": round(roi_shape.area, 3),
        },
        "density_thresholds_per_100k": {
            "warning": 0.3,
            "congested": 0.8,
        },
        "frames_written": len(summaries),
        "frames": summaries,
    }
    summary_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"output_video: {output_video.resolve()}")
    print(f"summary_json: {summary_json.resolve()}")
    print(f"frames_written: {len(summaries)}")
    print(f"size: {width}x{height}")
    print(f"fps: {args.fps}")




def parse_class_ids(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "all"}:
        return None
    return [int(item.strip()) for item in text.split(",") if item.strip()]

def parse_source(source: str):
    return int(source) if source.isdigit() else source


def create_video_writer(output_video: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output_video}")
    return writer


def build_roi(roi_arg: Optional[str], width: int, height: int) -> ROIShape:
    if not roi_arg:
        return ROIShape([(width * 0.15, height * 0.45), (width * 0.85, height * 0.45), (width * 0.95, height * 0.95), (width * 0.05, height * 0.95)])

    path = Path(roi_arg)
    raw = json.loads(path.read_text(encoding="utf-8") if path.exists() else roi_arg)
    if isinstance(raw, Mapping) and "rect" in raw:
        rect = normalize_rect(raw["rect"], width, height)
        return ROIShape.from_rect(rect)
    if isinstance(raw, Mapping) and "polygon" in raw:
        return ROIShape(normalize_points(raw["polygon"], width, height))
    if isinstance(raw, list):
        return ROIShape(normalize_points(raw, width, height))
    raise ValueError("ROI must be a JSON object with 'rect' or 'polygon', or a point list")


def normalize_rect(rect: object, width: int, height: int) -> List[float]:
    if not isinstance(rect, list) and not isinstance(rect, tuple):
        raise ValueError("ROI rect must be [x1, y1, x2, y2]")
    if len(rect) != 4:
        raise ValueError("ROI rect must be [x1, y1, x2, y2]")
    values = [float(value) for value in rect]
    if all(0.0 <= value <= 1.0 for value in values):
        return [values[0] * width, values[1] * height, values[2] * width, values[3] * height]
    return values


def normalize_points(points: object, width: int, height: int) -> List[Tuple[float, float]]:
    if not isinstance(points, list):
        raise ValueError("ROI polygon must be a list of [x, y] points")
    normalized = []
    flat_values = []
    for point in points:
        if not isinstance(point, list) and not isinstance(point, tuple):
            raise ValueError("ROI polygon point must be [x, y]")
        if len(point) != 2:
            raise ValueError("ROI polygon point must be [x, y]")
        x, y = float(point[0]), float(point[1])
        normalized.append((x, y))
        flat_values.extend([x, y])
    if flat_values and all(0.0 <= value <= 1.0 for value in flat_values):
        return [(x * width, y * height) for x, y in normalized]
    return normalized


def update_flow_counter(
    flow_counter: FlowCounter,
    tracked: Sequence[TrackedDetection],
    previous_points: Dict[int, Point],
) -> None:
    active_ids = set()
    for item in tracked:
        active_ids.add(item.track_id)
        current = item.detection.bbox.bottom_center
        previous = previous_points.get(item.track_id)
        if previous is not None:
            flow_counter.update(item.track_id, previous, current)
        previous_points[item.track_id] = current

    for track_id in list(previous_points):
        if track_id not in active_ids:
            del previous_points[track_id]


def estimate_low_speed_ratio(
    tracked: Sequence[TrackedDetection],
    previous_points: Dict[int, Point],
    low_speed_threshold: float = 2.0,
) -> float:
    if not tracked:
        return 0.0
    low_speed_count = 0
    for item in tracked:
        previous = previous_points.get(item.track_id)
        current = item.detection.bbox.bottom_center
        if previous is None or distance(previous, current) < low_speed_threshold:
            low_speed_count += 1
    return low_speed_count / len(tracked)


def compute_density_stats(detections: Sequence[Detection], roi_shape: ROIShape) -> DensityStats:
    vehicle_count = 0
    non_motor_count = 0
    pedestrian_count = 0
    weighted_count = 0.0
    occupied_area = 0.0

    for detection in detections:
        bbox_tuple = detection_bbox_tuple(detection)
        if not roi_shape.contains_bbox(bbox_tuple):
            continue
        category = classify_detection(detection_to_dict(detection))
        if category == "motor_vehicle":
            vehicle_count += 1
        elif category == "non_motor":
            non_motor_count += 1
        elif category == "pedestrian":
            pedestrian_count += 1
        else:
            continue
        weighted_count += CLASS_WEIGHTS.get(category, 1.0)
        occupied_area += detection.bbox.area

    total_count = vehicle_count + non_motor_count + pedestrian_count
    density = estimate_roi_density(total_count, roi_shape.area)
    weighted_density = estimate_roi_density(weighted_count, roi_shape.area)
    occupancy_ratio = min(1.0, max(0.0, occupied_area / roi_shape.area)) if roi_shape.area > 0 else 0.0
    return DensityStats(
        vehicle_count=vehicle_count,
        non_motor_count=non_motor_count,
        pedestrian_count=pedestrian_count,
        density=density,
        weighted_density=weighted_density,
        roi_area=roi_shape.area,
        occupancy_ratio=occupancy_ratio,
    )


def density_congestion_status(
    weighted_density_per_100k: float,
    warning_active: bool,
    congestion_level: str,
) -> Tuple[str, str]:
    safe_density = safe_float(weighted_density_per_100k)
    if safe_density >= 0.8 or congestion_level in {"congested", "severe"}:
        return "congested", "CONGESTED"
    if safe_density >= 0.3 or warning_active or congestion_level == "slow":
        return "warning", "WARNING"
    return "clear", "CLEAR"


def draw_frame(
    frame,
    tracked: Sequence[TrackedDetection],
    roi_points: Sequence[Point],
    flow_count: int,
    density_stats: DensityStats,
    congestion_status: str,
    congestion_text: str,
):
    for item in tracked:
        draw_detection(frame, item)
    draw_roi(frame, roi_points)
    draw_status_panel(
        frame,
        vehicle_count=len(tracked),
        flow_count=flow_count,
        density_stats=density_stats,
        congestion_status=congestion_status,
        congestion_text=congestion_text,
    )
    return frame


def draw_detection(frame, item: TrackedDetection) -> None:
    det = item.detection
    bbox = det.bbox
    x1, y1 = int(round(bbox.x1)), int(round(bbox.y1))
    x2, y2 = int(round(bbox.x2)), int(round(bbox.y2))
    color = color_for_class(det.cls_id)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    label = f"ID {item.track_id} {det.label} {det.conf:.2f}"
    draw_label(frame, label, (x1, y1), color)


def draw_label(frame, label: str, top_left: Tuple[int, int], color: Tuple[int, int, int]) -> None:
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


def draw_roi(frame, roi_points: Sequence[Point]) -> None:
    points = np.array([(int(p.x), int(p.y)) for p in roi_points], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [points], isClosed=True, color=(255, 180, 0), thickness=2)


def draw_status_panel(
    frame,
    vehicle_count: int,
    flow_count: int,
    density_stats: DensityStats,
    congestion_status: str,
    congestion_text: str,
) -> None:
    status = sanitize_status(congestion_status)
    status_code = sanitize_status_text(congestion_text, status)
    status_cn = {
        "clear": "畅通",
        "warning": "警告",
        "congested": "拥堵",
    }.get(status, "畅通")
    status_color_bgr = {
        "clear": (0, 220, 0),
        "warning": (0, 220, 255),
        "congested": (0, 0, 255),
    }.get(status, (255, 255, 255))
    status_color_rgb = bgr_to_rgb(status_color_bgr)

    motor_count = safe_int(density_stats.vehicle_count)
    non_motor_count = safe_int(density_stats.non_motor_count)
    pedestrian_count = safe_int(density_stats.pedestrian_count)
    total_count = safe_int(vehicle_count)
    flow_total = safe_int(flow_count)
    density = safe_float(density_stats.density)
    weighted_density = safe_float(density_stats.weighted_density)
    density_per_100k = safe_float(density_stats.weighted_density_per_100k)
    occupancy_ratio = safe_float(density_stats.occupancy_ratio)
    roi_area = safe_float(density_stats.roi_area)

    lines = [
        f"Vehicles 总目标数量: {total_count:d}",
        f"ROI车辆(car/truck): {motor_count:d}",
        f"Flow Count 检测线通过车辆数: {flow_total:d}",
        f"Density 道路密度(车辆数/ROI面积): {density:.4f}",
        f"Weighted Density 类别加权密度: {weighted_density:.4f}",
        f"Density/100k 每10万像素密度: {density_per_100k:.2f}",
        f"Occupancy 道路占用率: {occupancy_ratio:.4f}",
        f"ROI Area ROI区域像素面积: {roi_area:.2f}",
        f"Status 拥堵状态: {status_code}={status_cn}",
    ]

    font = load_chinese_font(size=17)
    title_font = load_chinese_font(size=22)
    padding_x = 12
    padding_y = 10
    row_height = 25
    title_height = 34
    panel_x1 = 8
    panel_y1 = 8

    max_text_width = 0
    for line in lines:
        max_text_width = max(max_text_width, pil_text_size(font, line)[0])
    max_text_width = max(max_text_width, pil_text_size(title_font, f"{status_code} {status_cn}")[0])

    panel_width = max(620, max_text_width + padding_x * 2)
    panel_height = padding_y * 2 + row_height * len(lines) + title_height
    panel_x2 = min(frame.shape[1] - 1, panel_x1 + panel_width)
    panel_y2 = min(frame.shape[0] - 1, panel_y1 + panel_height)

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.74, frame, 0.26, 0, frame)
    cv2.rectangle(frame, (panel_x1, panel_y1), (panel_x2, panel_y2), status_color_bgr, 1)

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    text_x = panel_x1 + padding_x
    text_y = panel_y1 + padding_y

    for index, line in enumerate(lines):
        fill = status_color_rgb if index == len(lines) - 1 else (255, 255, 255)
        draw.text((text_x, text_y + index * row_height), line, font=font, fill=fill)

    title_y = text_y + row_height * len(lines) + 4
    draw.text((text_x, title_y), f"{status_code} {status_cn}", font=title_font, fill=status_color_rgb)
    frame[:] = cv2.cvtColor(np.asarray(pil_image), cv2.COLOR_RGB2BGR)


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


def pil_text_size(font: ImageFont.ImageFont, text: str) -> Tuple[int, int]:
    if hasattr(font, "getbbox"):
        left, top, right, bottom = font.getbbox(text)
        return right - left, bottom - top
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


def safe_int(value: object, default: int = 0) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(float(result)) else default


def sanitize_status(status: object) -> str:
    text = str(status).strip().lower()
    if text in {"clear", "warning", "congested"}:
        return text
    return "clear"


def sanitize_status_text(text: object, status: str) -> str:
    fallback = {
        "clear": "CLEAR",
        "warning": "WARNING",
        "congested": "CONGESTED",
    }.get(status, "CLEAR")
    raw = str(text).strip().upper()
    if raw in {"CLEAR", "WARNING", "CONGESTED"}:
        return raw
    return fallback


def build_frame_summary(
    frame_index: int,
    detections: Sequence[Detection],
    detected_this_frame: bool,
    flow_count: int,
    density_stats: DensityStats,
    congestion_level: str,
    congestion_status: str,
    congestion_text: str,
    warning_active: bool,
) -> Dict[str, object]:
    return {
        "frame_index": frame_index,
        "detected_this_frame": detected_this_frame,
        "detections_count": len(detections),
        "detections": [detection_to_dict(det) for det in detections],
        "flow_count": flow_count,
        "vehicle_count": density_stats.vehicle_count,
        "non_motor_count": density_stats.non_motor_count,
        "pedestrian_count": density_stats.pedestrian_count,
        "density": round(float(density_stats.density), 10),
        "weighted_density": round(float(density_stats.weighted_density), 10),
        "density_per_100k": round(float(density_stats.weighted_density_per_100k), 6),
        "roi_area": round(float(density_stats.roi_area), 3),
        "occupancy_ratio": round(float(density_stats.occupancy_ratio), 6),
        "congestion_level": congestion_level,
        "congestion_status": congestion_status,
        "congestion_text": congestion_text,
        "warning_active": warning_active,
    }


def detection_to_dict(detection: Detection) -> Dict[str, object]:
    bbox = detection.bbox
    return {
        "class_id": detection.cls_id,
        "label": detection.label,
        "conf": round(float(detection.conf), 6),
        "bbox": [
            round(float(bbox.x1), 3),
            round(float(bbox.y1), 3),
            round(float(bbox.x2), 3),
            round(float(bbox.y2), 3),
        ],
    }


def detection_bbox_tuple(detection: Detection) -> Tuple[float, float, float, float]:
    bbox = detection.bbox
    return bbox.x1, bbox.y1, bbox.x2, bbox.y2


def bbox_iou(first: BBox, second: BBox) -> float:
    inter_x1 = max(first.x1, second.x1)
    inter_y1 = max(first.y1, second.y1)
    inter_x2 = min(first.x2, second.x2)
    inter_y2 = min(first.y2, second.y2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    union_area = first.area + second.area - inter_area
    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def distance(first: Point, second: Point) -> float:
    return ((first.x - second.x) ** 2 + (first.y - second.y) ** 2) ** 0.5


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


if __name__ == "__main__":
    main()










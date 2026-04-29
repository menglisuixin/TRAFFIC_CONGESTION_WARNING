"""Command-line entry point for the traffic congestion warning MVP."""

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from analytics.congestion_detector import CongestionDetector
from analytics.flow_counter import FlowCounter, LineSegment
from analytics.roi import PolygonROI
from analytics.trajectory import TrajectoryStore
from core.types import Point, Track
from detector.yolov5_detector import YOLOv5Detector
from tracker.deepsort_tracker import DeepSORTTracker
from visualization.drawer import TrafficDrawer
from core.pipeline import load_weight_class_names

SourceType = Union[int, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traffic congestion warning MVP")
    parser.add_argument("--source", default="0", help="Video path or camera index")
    parser.add_argument("--weights", required=True, help="YOLOv5 weights path")
    parser.add_argument("--device", default="cuda", help="Inference device, e.g. cuda or cpu")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="Detection confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="Detection NMS IoU threshold")
    parser.add_argument("--img-size", type=int, default=640, help="YOLOv5 inference image size")
    parser.add_argument("--output", default="", help="Optional output video path")
    parser.add_argument("--show", action="store_true", help="Show live visualization window")
    parser.add_argument(
        "--classes",
        default="auto",
        help="Class ids to detect (comma-separated), 'all', or 'auto'. Default: auto.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = _parse_source(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    ok, frame = capture.read()
    if not ok or frame is None:
        capture.release()
        raise RuntimeError(f"Could not read first frame from source: {args.source}")

    frame_height, frame_width = frame.shape[:2]
    roi_points = _default_roi(frame_width, frame_height)
    roi = PolygonROI(roi_points)
    flow_line = LineSegment(roi_points[0], roi_points[1])

    detector = YOLOv5Detector(
        weights=args.weights,
        device=args.device,
        img_size=args.img_size,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        classes=resolve_classes_arg(args.weights, args.classes),
    )
    tracker = DeepSORTTracker()
    trajectories = TrajectoryStore(max_length=30, use_bottom_center=True)
    flow_counter = FlowCounter(flow_line, direction="any")
    congestion_detector = CongestionDetector()
    drawer = TrafficDrawer()

    writer = _create_writer(args.output, capture, frame_width, frame_height)

    frame_index = 0
    try:
        while True:
            if frame_index > 0:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

            detections = detector.infer(frame)
            tracks = tracker.update(detections, frame, frame_index)
            previous_points = {
                track.track_id: trajectories.get_latest_point(track.track_id)
                for track in tracks
            }

            for track in tracks:
                trajectories.update(track)
                previous = previous_points.get(track.track_id)
                current = trajectories.get_latest_point(track.track_id)
                if previous is not None and current is not None:
                    flow_counter.update(track.track_id, previous, current)

            active_track_ids = {track.track_id for track in tracks}
            trajectories.prune_missing(active_track_ids)

            stats = _build_stats(
                frame_index=frame_index,
                frame_shape=frame.shape,
                tracks=tracks,
                roi=roi,
                trajectories=trajectories,
                flow_count=flow_counter.total_count,
            )
            level, warning_active = congestion_detector.update(
                roi_vehicle_count=int(stats["roi_vehicle_count"]),
                low_speed_ratio=float(stats["low_speed_ratio"]),
                occupancy_ratio=float(stats["occupancy_ratio"]),
            )
            stats["congestion_level"] = level
            stats["warning_active"] = warning_active

            output_frame = drawer.draw_tracks(frame.copy(), tracks)
            output_frame = drawer.draw_roi(output_frame, roi_points)
            output_frame = drawer.draw_stats(output_frame, stats)

            if writer is not None:
                writer.write(output_frame)
            if args.show:
                cv2.imshow("traffic_congestion_warning", output_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()




def parse_class_ids(value: str):
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "all"}:
        return None
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def resolve_classes_arg(weights: str, classes_arg: str):
    text = str(classes_arg).strip().lower()
    if text in {"", "all", "none", "null"}:
        return None
    if text != "auto":
        return parse_class_ids(classes_arg)

    names = load_weight_class_names(weights)
    normalized = {str(name).strip().lower() for name in names.values()}
    if {"car", "truck"}.issubset(normalized):
        return [2, 7]
    if {"motor vehicle", "motor_vehicle", "vehicle"} & normalized:
        # Custom traffic weights (single-class or multi-class) - keep Motor Vehicle only.
        motor_ids = [int(i) for i, name in names.items() if str(name).strip().lower() in {"motor vehicle", "motor_vehicle", "vehicle"}]
        return motor_ids or None
    return None

def _parse_source(source: str) -> SourceType:
    return int(source) if source.isdigit() else source


def _default_roi(frame_width: int, frame_height: int) -> List[Point]:
    top_y = frame_height * 0.45
    bottom_y = frame_height * 0.95
    return [
        Point(frame_width * 0.15, top_y),
        Point(frame_width * 0.85, top_y),
        Point(frame_width * 0.95, bottom_y),
        Point(frame_width * 0.05, bottom_y),
    ]


def _create_writer(
    output_path: str,
    capture: cv2.VideoCapture,
    frame_width: int,
    frame_height: int,
) -> Optional[cv2.VideoWriter]:
    if not output_path:
        return None

    path = Path(output_path)
    if path.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv"}:
        path.mkdir(parents=True, exist_ok=True)
        path = path / "result.mp4"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output video writer: {output_path}")
    print(f"output_video: {path.resolve()}")
    return writer


def _build_stats(
    frame_index: int,
    frame_shape: Tuple[int, int, int],
    tracks: List[Track],
    roi: PolygonROI,
    trajectories: TrajectoryStore,
    flow_count: int,
) -> Dict[str, object]:
    roi_tracks = [track for track in tracks if roi.contains_bbox_bottom_center(track.bbox)]
    speeds = [trajectories.get_displacement(track.track_id, window=2) for track in tracks]
    mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
    low_speed_count = sum(1 for speed in speeds if speed < 2.0)
    low_speed_ratio = low_speed_count / len(speeds) if speeds else 0.0

    frame_height, frame_width = frame_shape[:2]
    frame_area = float(frame_width * frame_height)
    roi_vehicle_area = sum(track.bbox.area for track in roi_tracks)
    occupancy_ratio = min(1.0, roi_vehicle_area / frame_area) if frame_area > 0 else 0.0
    density = len(roi_tracks) / max(1.0, frame_area / 100000.0)

    return {
        "frame_index": frame_index,
        "vehicle_count": len(tracks),
        "roi_vehicle_count": len(roi_tracks),
        "mean_speed": mean_speed,
        "low_speed_ratio": low_speed_ratio,
        "occupancy_ratio": occupancy_ratio,
        "density": density,
        "congestion_level": "normal",
        "warning_active": False,
        "flow_count": flow_count,
    }


if __name__ == "__main__":
    main()



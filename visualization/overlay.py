"""Reusable OpenCV overlays for traffic analysis visualization."""

import argparse
import sys
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.types import BBox, Point, Track

Color = Tuple[int, int, int]

STATUS_COLORS: Dict[str, Color] = {
    "CLEAR": (0, 220, 0),
    "WARNING": (0, 220, 255),
    "CONGESTED": (0, 0, 255),
}


def draw_transparent_rect(frame: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int], color: Color = (0, 0, 0), alpha: float = 0.65) -> np.ndarray:
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    return frame


def draw_bbox(frame: np.ndarray, bbox: BBox, label: str = "", color: Color = (0, 255, 0), thickness: int = 2) -> np.ndarray:
    p1 = int(round(bbox.x1)), int(round(bbox.y1))
    p2 = int(round(bbox.x2)), int(round(bbox.y2))
    cv2.rectangle(frame, p1, p2, color, thickness, cv2.LINE_AA)
    if label:
        draw_text_label(frame, label, p1, color)
    return frame


def draw_tracks(frame: np.ndarray, tracks: Sequence[Track], speeds: Optional[Mapping[int, Mapping[str, float]]] = None) -> np.ndarray:
    for track in tracks:
        speed = 0.0
        if speeds and track.track_id in speeds:
            speed = safe_float(speeds[track.track_id].get("speed_kmh"))
        label = f"ID {track.track_id} {track.label} {track.score:.2f} {speed:.1f} km/h"
        draw_bbox(frame, track.bbox, label=label, color=color_for_id(track.track_id))
    return frame


def draw_trajectory(frame: np.ndarray, points: Sequence[Point], color: Color = (255, 255, 0), thickness: int = 2) -> np.ndarray:
    if len(points) < 2:
        return frame
    coords = [(int(round(point.x)), int(round(point.y))) for point in points]
    for first, second in zip(coords, coords[1:]):
        cv2.line(frame, first, second, color, thickness, cv2.LINE_AA)
    return frame


def draw_roi(frame: np.ndarray, points: Sequence[Point], color: Color = (255, 180, 0), thickness: int = 2, fill_alpha: float = 0.0) -> np.ndarray:
    if len(points) < 3:
        return frame
    polygon = np.array([(int(p.x), int(p.y)) for p in points], dtype=np.int32).reshape((-1, 1, 2))
    if fill_alpha > 0.0:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [polygon], color)
        cv2.addWeighted(overlay, fill_alpha, frame, 1.0 - fill_alpha, 0, frame)
    cv2.polylines(frame, [polygon], True, color, thickness, cv2.LINE_AA)
    return frame


def draw_status_panel(frame: np.ndarray, stats: Mapping[str, object], origin: Tuple[int, int] = (8, 8)) -> np.ndarray:
    status = str(stats.get("congestion_status", stats.get("status", "CLEAR"))).upper()
    status_color = STATUS_COLORS.get(status, (255, 255, 255))
    lines = [
        f"Vehicles: {safe_int(stats.get('detections_count', stats.get('vehicle_count', 0)))}",
        f"Motor: {safe_int(stats.get('vehicle_count'))} NonMotor: {safe_int(stats.get('non_motor_count'))} Ped: {safe_int(stats.get('pedestrian_count'))}",
        f"Flow Count: {safe_int(stats.get('flow_count'))}",
        f"Density: {safe_float(stats.get('density')):.4f}",
        f"Weighted Density: {safe_float(stats.get('weighted_density')):.4f}",
        f"Occupancy: {safe_float(stats.get('occupancy_ratio')):.4f}",
        f"Avg Speed: {safe_float(stats.get('mean_speed_kmh')):.2f} km/h",
        f"Max Speed: {safe_float(stats.get('max_speed_kmh')):.2f} km/h",
        f"Status: {status}",
    ]
    x, y = origin
    row_h = 24
    width = min(frame.shape[1] - x - 2, 600)
    height = 18 + row_h * len(lines)
    draw_transparent_rect(frame, (x, y), (x + width, min(frame.shape[0] - 1, y + height)), (0, 0, 0), 0.72)
    cv2.rectangle(frame, (x, y), (x + width, min(frame.shape[0] - 1, y + height)), status_color, 1)
    for index, line in enumerate(lines):
        color = status_color if line.startswith("Status:") else (255, 255, 255)
        cv2.putText(frame, line, (x + 12, y + 26 + index * row_h), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
    return frame


def draw_text_label(frame: np.ndarray, text: str, top_left: Tuple[int, int], color: Color) -> np.ndarray:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    text_w, text_h = text_size
    x, y = top_left
    y1 = max(0, y - text_h - baseline - 5)
    y2 = y if y1 < y else y + text_h + baseline + 5
    cv2.rectangle(frame, (x, y1), (x + text_w + 5, y2), color, -1)
    cv2.putText(frame, text, (x + 2, y2 - baseline - 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame


def color_for_id(identifier: int) -> Color:
    palette = [(56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72)]
    return palette[int(identifier) % len(palette)]


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a status overlay on one image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--stats", required=True, help="JSON string or JSON file with status fields")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"could not read image: {args.image}")
    stats_path = Path(args.stats)
    stats = json.loads(stats_path.read_text(encoding="utf-8") if stats_path.exists() else args.stats)
    draw_status_panel(image, stats)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), image)
    print(f"output: {output.resolve()}")


if __name__ == "__main__":
    main()

"""Calibration point collection tool for homography setup."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from calibration.homography import compute_homography

Point = Tuple[float, float]


def parse_points(value: str, name: str) -> List[Point]:
    raw = json.loads(Path(value).read_text(encoding="utf-8") if Path(value).exists() else value)
    if not isinstance(raw, list):
        raise ValueError(f"{name} must be a JSON list")
    points = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{name} points must be [x, y]")
        points.append((float(item[0]), float(item[1])))
    return points


def collect_pixel_points(image_path: str, count: int = 4, window_name: str = "calibrate_points") -> List[Point]:
    image = cv2.imread(image_path)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    points: List[Point] = []
    preview = image.copy()

    def redraw() -> None:
        preview[:] = image
        for index, (x, y) in enumerate(points):
            cv2.circle(preview, (int(x), int(y)), 5, (0, 255, 255), -1)
            cv2.putText(preview, str(index + 1), (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow(window_name, preview)

    def on_mouse(event, x, y, flags, userdata) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < count:
            points.append((float(x), float(y)))
            redraw()
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
            redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    redraw()
    print("Left click to add points, right click to undo, Enter to finish, Esc to cancel.")
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(points) >= count:
            break
        if key == 27:
            points = []
            break
    cv2.destroyWindow(window_name)
    if len(points) < count:
        raise RuntimeError(f"Need at least {count} pixel points, got {len(points)}")
    return points[:count]


def save_calibration(output: str, pixel_points: Sequence[Point], world_points: Sequence[Point]) -> Path:
    homography = compute_homography(pixel_points, world_points)
    payload = {
        "pixel_points": [[float(x), float(y)] for x, y in pixel_points],
        "world_points": [[float(x), float(y)] for x, y in world_points],
        "homography": homography.to_list(),
        "description": "pixel_points map to world_points in meters",
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect or save calibration points for pixel-to-world homography.")
    parser.add_argument("--image", help="Image path for interactive pixel point collection")
    parser.add_argument("--pixel-points", help='Pixel points JSON/file, e.g. "[[100,500],[500,500],[500,700],[100,700]]"')
    parser.add_argument("--world-points", required=True, help='World meter points JSON/file, e.g. "[[0,0],[10,0],[10,20],[0,20]]"')
    parser.add_argument("--output", default="configs/calibration.json", help="Output calibration JSON")
    parser.add_argument("--count", type=int, default=4, help="Number of point pairs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    world_points = parse_points(args.world_points, "world_points")
    if args.pixel_points:
        pixel_points = parse_points(args.pixel_points, "pixel_points")
    elif args.image:
        pixel_points = collect_pixel_points(args.image, count=args.count)
    else:
        raise SystemExit("Provide --pixel-points or --image for interactive point collection")

    if len(pixel_points) != len(world_points):
        raise SystemExit("pixel_points and world_points must have the same length")
    if len(pixel_points) < 4:
        raise SystemExit("at least 4 point pairs are required")

    output = save_calibration(args.output, pixel_points, world_points)
    print(f"output: {output.resolve()}")
    print(f"points: {len(pixel_points)}")


if __name__ == "__main__":
    main()

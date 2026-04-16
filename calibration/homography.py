"""Homography calibration helpers for pixel-to-world mapping."""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

Point2D = Tuple[float, float]


@dataclass(frozen=True)
class Homography:
    """Pixel-to-world homography transform.

    matrix maps image pixel coordinates (x, y) to world coordinates (X, Y) in meters.
    """

    matrix: np.ndarray

    def pixel_to_world(self, x: float, y: float) -> Point2D:
        return apply_homography(self.matrix, x, y)

    def distance_pixels(self, first: Point2D, second: Point2D) -> float:
        first_world = self.pixel_to_world(first[0], first[1])
        second_world = self.pixel_to_world(second[0], second[1])
        return world_distance(first_world, second_world)

    def to_list(self) -> List[List[float]]:
        return self.matrix.astype(float).tolist()


# Backward-compatible module-level mapper. Set by set_default_homography().
_DEFAULT_HOMOGRAPHY: Optional[Homography] = None


def compute_homography(pixel_points: Sequence[Point2D], world_points: Sequence[Point2D]) -> Homography:
    """Compute a pixel-to-world homography from corresponding points.

    Args:
        pixel_points: image coordinates [(x, y), ...]. At least four points.
        world_points: real-world meter coordinates [(X, Y), ...]. Same length.
    """

    pixels = validate_points(pixel_points, "pixel_points")
    worlds = validate_points(world_points, "world_points")
    if len(pixels) != len(worlds):
        raise ValueError("pixel_points and world_points must have the same length")
    if len(pixels) < 4:
        raise ValueError("at least four point pairs are required")

    src = np.asarray(pixels, dtype=np.float32)
    dst = np.asarray(worlds, dtype=np.float32)
    matrix, status = cv2.findHomography(src, dst, method=0)
    if matrix is None or status is None:
        raise ValueError("failed to compute homography from provided points")
    return Homography(matrix=matrix.astype(np.float64))


def apply_homography(matrix: np.ndarray, x: float, y: float) -> Point2D:
    """Map one pixel coordinate to world coordinates using a homography matrix."""

    if matrix is None:
        raise ValueError("homography matrix is required")
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (3, 3):
        raise ValueError("homography matrix must have shape 3x3")

    vector = mat @ np.asarray([safe_float(x), safe_float(y), 1.0], dtype=np.float64)
    if abs(vector[2]) < 1e-12:
        raise ValueError("invalid homography projection with near-zero scale")
    return float(vector[0] / vector[2]), float(vector[1] / vector[2])


def set_default_homography(pixel_points: Sequence[Point2D], world_points: Sequence[Point2D]) -> Homography:
    """Compute and store a default module-level homography."""

    global _DEFAULT_HOMOGRAPHY
    _DEFAULT_HOMOGRAPHY = compute_homography(pixel_points, world_points)
    return _DEFAULT_HOMOGRAPHY


def pixel_to_world(x: float, y: float) -> Point2D:
    """Map pixel to world coordinates using the default homography."""

    if _DEFAULT_HOMOGRAPHY is None:
        raise RuntimeError("default homography is not set")
    return _DEFAULT_HOMOGRAPHY.pixel_to_world(x, y)


def world_distance(first: Point2D, second: Point2D) -> float:
    """Euclidean distance in meters between two world points."""

    x1, y1 = first
    x2, y2 = second
    return math.hypot(safe_float(x2) - safe_float(x1), safe_float(y2) - safe_float(y1))


def pixel_world_distance(first_pixel: Point2D, second_pixel: Point2D, homography: Optional[Homography] = None) -> float:
    """Distance in meters between two pixel points after homography mapping."""

    mapper = homography or _DEFAULT_HOMOGRAPHY
    if mapper is None:
        raise RuntimeError("homography is not set")
    return mapper.distance_pixels(first_pixel, second_pixel)


def load_point_pairs(path: Union[str, Path]) -> Tuple[List[Point2D], List[Point2D]]:
    """Load calibration pairs from JSON.

    Supported formats:
      {"pixel_points": [[x, y], ...], "world_points": [[X, Y], ...]}
      {"pixels": [[x, y], ...], "worlds": [[X, Y], ...]}
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    pixel_points = data.get("pixel_points", data.get("pixels"))
    world_points = data.get("world_points", data.get("worlds"))
    if pixel_points is None or world_points is None:
        raise ValueError("calibration JSON must contain pixel_points and world_points")
    return validate_points(pixel_points, "pixel_points"), validate_points(world_points, "world_points")


def validate_points(points: Sequence[Point2D], name: str) -> List[Point2D]:
    if not isinstance(points, Iterable):
        raise ValueError(f"{name} must be a sequence of [x, y] points")
    parsed: List[Point2D] = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"{name} points must be [x, y]")
        x = safe_float(point[0])
        y = safe_float(point[1])
        parsed.append((x, y))
    return parsed


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


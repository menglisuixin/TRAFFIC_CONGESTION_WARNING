"""Geometry helpers for calibration and traffic analytics."""

import math
from typing import Sequence, Tuple

Point2D = Tuple[float, float]


def point_distance(first: Point2D, second: Point2D) -> float:
    """Euclidean distance between two 2D points."""

    return math.hypot(safe_float(second[0]) - safe_float(first[0]), safe_float(second[1]) - safe_float(first[1]))


def segment_length(start: Point2D, end: Point2D) -> float:
    """Length of a line segment."""

    return point_distance(start, end)


def segment_angle_degrees(start: Point2D, end: Point2D) -> float:
    """Angle of a segment in degrees, measured counter-clockwise from x-axis."""

    return math.degrees(math.atan2(safe_float(end[1]) - safe_float(start[1]), safe_float(end[0]) - safe_float(start[0])))


def polygon_area(points: Sequence[Point2D]) -> float:
    """Area of a polygon using the shoelace formula."""

    if len(points) < 3:
        return 0.0
    area = 0.0
    previous = points[-1]
    for current in points:
        area += safe_float(previous[0]) * safe_float(current[1]) - safe_float(current[0]) * safe_float(previous[1])
        previous = current
    return abs(area) / 2.0


def rectangle_area(x1: float, y1: float, x2: float, y2: float) -> float:
    """Area of an axis-aligned rectangle."""

    return max(0.0, safe_float(x2) - safe_float(x1)) * max(0.0, safe_float(y2) - safe_float(y1))


def roi_area(points: Sequence[Point2D]) -> float:
    """Alias for polygon ROI area."""

    return polygon_area(points)


def lane_width(left_boundary_point: Point2D, right_boundary_point: Point2D) -> float:
    """Lane width as distance between left and right boundary points."""

    return point_distance(left_boundary_point, right_boundary_point)


def midpoint(first: Point2D, second: Point2D) -> Point2D:
    """Midpoint between two points."""

    return (safe_float(first[0]) + safe_float(second[0])) / 2.0, (safe_float(first[1]) + safe_float(second[1])) / 2.0


def bbox_center(bbox: Sequence[float]) -> Point2D:
    """Center point of bbox [x1, y1, x2, y2]."""

    if len(bbox) != 4:
        raise ValueError("bbox must be [x1, y1, x2, y2]")
    return (safe_float(bbox[0]) + safe_float(bbox[2])) / 2.0, (safe_float(bbox[1]) + safe_float(bbox[3])) / 2.0


def bbox_bottom_center(bbox: Sequence[float]) -> Point2D:
    """Bottom-center point of bbox [x1, y1, x2, y2]."""

    if len(bbox) != 4:
        raise ValueError("bbox must be [x1, y1, x2, y2]")
    return (safe_float(bbox[0]) + safe_float(bbox[2])) / 2.0, safe_float(bbox[3])


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default

"""Region of interest helpers."""

from typing import Iterable, List, Tuple, Union

from core.types import BBox, Point

PointLike = Union[Point, Tuple[float, float]]


class PolygonROI:
    """Polygon region of interest using ray casting for point inclusion."""

    def __init__(self, vertices: Iterable[PointLike]) -> None:
        self.vertices: List[Point] = [self._to_point(vertex) for vertex in vertices]
        if len(self.vertices) < 3:
            raise ValueError("PolygonROI requires at least three vertices")

    def contains_point(self, point: PointLike) -> bool:
        """Return True when the point is inside the polygon or on its boundary."""

        target = self._to_point(point)
        if self._is_on_boundary(target):
            return True

        inside = False
        j = len(self.vertices) - 1
        for i, current in enumerate(self.vertices):
            previous = self.vertices[j]
            crosses_ray = (current.y > target.y) != (previous.y > target.y)
            if crosses_ray:
                x_intersection = (
                    (previous.x - current.x)
                    * (target.y - current.y)
                    / (previous.y - current.y)
                    + current.x
                )
                if target.x < x_intersection:
                    inside = not inside
            j = i

        return inside

    def contains_bbox_center(self, bbox: BBox) -> bool:
        """Return True when the bounding box center is inside the ROI."""

        return self.contains_point(bbox.center)

    def contains_bbox_bottom_center(self, bbox: BBox) -> bool:
        """Return True when the bounding box bottom-center is inside the ROI."""

        return self.contains_point(bbox.bottom_center)

    @staticmethod
    def _to_point(value: PointLike) -> Point:
        if isinstance(value, Point):
            return value
        x, y = value
        return Point(float(x), float(y))

    def _is_on_boundary(self, point: Point) -> bool:
        for start, end in self._edges():
            if self._point_on_segment(point, start, end):
                return True
        return False

    def _edges(self):
        previous = self.vertices[-1]
        for current in self.vertices:
            yield previous, current
            previous = current

    @staticmethod
    def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
        cross = (point.y - start.y) * (end.x - start.x) - (point.x - start.x) * (
            end.y - start.y
        )
        if abs(cross) > 1e-9:
            return False

        min_x, max_x = sorted((start.x, end.x))
        min_y, max_y = sorted((start.y, end.y))
        return min_x - 1e-9 <= point.x <= max_x + 1e-9 and min_y - 1e-9 <= point.y <= max_y + 1e-9

"""Traffic flow counting by line crossing."""

from dataclasses import dataclass
from typing import Set, Tuple, Union

from core.types import Point

PointLike = Union[Point, Tuple[float, float]]


@dataclass(frozen=True)
class LineSegment:
    """A directed detection line from start to end."""

    start: Point
    end: Point


class FlowCounter:
    """Counts unique tracks that cross a detection line."""

    VALID_DIRECTIONS = {"any", "forward", "backward"}

    def __init__(self, line: LineSegment, direction: str = "any") -> None:
        if direction not in self.VALID_DIRECTIONS:
            raise ValueError(f"direction must be one of {sorted(self.VALID_DIRECTIONS)}")
        self.line = line
        self.direction = direction
        self._counted_ids: Set[int] = set()
        self._total_count = 0

    def update(self, track_id: int, prev_point: PointLike, curr_point: PointLike) -> bool:
        """Return True only when this update creates a new crossing count."""

        if track_id in self._counted_ids:
            return False

        previous = self._to_point(prev_point)
        current = self._to_point(curr_point)
        if not self._crosses_line(previous, current):
            return False

        movement_direction = self._movement_direction(previous, current)
        if self.direction != "any" and movement_direction != self.direction:
            return False

        self._counted_ids.add(track_id)
        self._total_count += 1
        return True

    @property
    def total_count(self) -> int:
        return self._total_count

    @property
    def counted_ids(self) -> Set[int]:
        return set(self._counted_ids)

    def reset(self) -> None:
        """Clear all counts and counted track ids."""

        self._counted_ids.clear()
        self._total_count = 0

    def _crosses_line(self, previous: Point, current: Point) -> bool:
        start_side = self._signed_side(previous)
        end_side = self._signed_side(current)

        if start_side == 0.0 and end_side == 0.0:
            return False
        if start_side == 0.0 or end_side == 0.0:
            return self._segment_intersects_line(previous, current)
        return start_side * end_side < 0.0 and self._segment_intersects_line(previous, current)

    def _movement_direction(self, previous: Point, current: Point) -> str:
        start_side = self._signed_side(previous)
        end_side = self._signed_side(current)
        return "forward" if start_side < end_side else "backward"

    def _signed_side(self, point: Point) -> float:
        value = self._cross(self.line.start, self.line.end, point)
        if abs(value) <= 1e-9:
            return 0.0
        return value

    def _segment_intersects_line(self, previous: Point, current: Point) -> bool:
        a = self.line.start
        b = self.line.end
        c = previous
        d = current

        o1 = self._cross(a, b, c)
        o2 = self._cross(a, b, d)
        o3 = self._cross(c, d, a)
        o4 = self._cross(c, d, b)

        if self._opposite_or_zero(o1, o2) and self._opposite_or_zero(o3, o4):
            return True
        return False

    @staticmethod
    def _opposite_or_zero(a: float, b: float) -> bool:
        return a == 0.0 or b == 0.0 or a * b < 0.0

    @staticmethod
    def _cross(a: Point, b: Point, c: Point) -> float:
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

    @staticmethod
    def _to_point(value: PointLike) -> Point:
        if isinstance(value, Point):
            return value
        x, y = value
        return Point(float(x), float(y))

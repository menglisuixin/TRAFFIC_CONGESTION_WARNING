"""Trajectory storage and distance helpers."""

from collections import defaultdict, deque
from math import hypot
from typing import Deque, Dict, List, Optional, Set

from core.types import Point, Track


class TrajectoryStore:
    """Keeps recent trajectory points for tracked objects."""

    def __init__(self, max_length: int = 30, use_bottom_center: bool = True) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be greater than zero")
        self.max_length = max_length
        self.use_bottom_center = use_bottom_center
        self._points: Dict[int, Deque[Point]] = defaultdict(
            lambda: deque(maxlen=self.max_length)
        )

    def update(self, track: Track) -> None:
        """Append the latest point for a track."""

        point = track.bbox.bottom_center if self.use_bottom_center else track.bbox.center
        self._points[track.track_id].append(point)

    def get_points(self, track_id: int) -> List[Point]:
        """Return stored points for a track from oldest to newest."""

        return list(self._points.get(track_id, ()))

    def get_latest_point(self, track_id: int) -> Optional[Point]:
        """Return the latest point for a track, or None when it has no trajectory."""

        points = self._points.get(track_id)
        if not points:
            return None
        return points[-1]

    def get_displacement(self, track_id: int, window: int = 5) -> float:
        """Return straight-line displacement over the latest window of points."""

        if window <= 1:
            return 0.0

        points = self.get_points(track_id)
        if len(points) < 2:
            return 0.0

        selected = points[-window:]
        if len(selected) < 2:
            return 0.0

        return self._distance(selected[0], selected[-1])

    def get_total_displacement(self, track_id: int) -> float:
        """Return accumulated path length across all stored points."""

        points = self.get_points(track_id)
        if len(points) < 2:
            return 0.0

        total = 0.0
        previous = points[0]
        for current in points[1:]:
            total += self._distance(previous, current)
            previous = current
        return total

    def prune_missing(self, active_track_ids: Set[int]) -> None:
        """Drop trajectories whose track ids are no longer active."""

        inactive_ids = set(self._points) - active_track_ids
        for track_id in inactive_ids:
            del self._points[track_id]

    def clear(self) -> None:
        """Remove all stored trajectories."""

        self._points.clear()

    @staticmethod
    def _distance(start: Point, end: Point) -> float:
        return hypot(end.x - start.x, end.y - start.y)

"""Drawing utilities for traffic warning visualization."""

from typing import Iterable, Mapping, Sequence, Tuple, Union

import cv2
import numpy as np

from core.types import Point, Track

PointLike = Union[Point, Tuple[float, float]]


class TrafficDrawer:
    """Draws tracking and traffic status overlays on video frames."""

    def __init__(self) -> None:
        self.track_color = (0, 255, 0)
        self.roi_color = (255, 180, 0)
        self.text_color = (255, 255, 255)
        self.warning_color = (0, 0, 255)
        self.panel_color = (0, 0, 0)

    def draw_tracks(self, frame: np.ndarray, tracks: Sequence[Track]) -> np.ndarray:
        for track in tracks:
            bbox = track.bbox
            p1 = (int(round(bbox.x1)), int(round(bbox.y1)))
            p2 = (int(round(bbox.x2)), int(round(bbox.y2)))
            cv2.rectangle(frame, p1, p2, self.track_color, 2)
            label = f"ID {track.track_id} {track.label} {track.score:.2f}"
            cv2.putText(
                frame,
                label,
                (p1[0], max(0, p1[1] - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                self.track_color,
                2,
                cv2.LINE_AA,
            )
        return frame

    def draw_roi(self, frame: np.ndarray, polygon_points: Iterable[PointLike]) -> np.ndarray:
        points = [self._to_xy(point) for point in polygon_points]
        if len(points) < 2:
            return frame
        array = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [array], isClosed=True, color=self.roi_color, thickness=2)
        return frame

    def draw_stats(self, frame: np.ndarray, stats_dict: Mapping[str, object]) -> np.ndarray:
        lines = [
            f"Vehicles: {stats_dict.get('vehicle_count', 0)}",
            f"ROI Vehicles: {stats_dict.get('roi_vehicle_count', 0)}",
            f"Flow Count: {stats_dict.get('flow_count', 0)}",
            f"Mean Speed: {float(stats_dict.get('mean_speed', 0.0)):.2f}",
            f"Congestion: {stats_dict.get('congestion_level', 'normal')}",
            f"Warning: {stats_dict.get('warning_active', False)}",
        ]

        warning_active = bool(stats_dict.get("warning_active", False))
        height = 24 * len(lines) + 12
        cv2.rectangle(frame, (8, 8), (300, height), self.panel_color, -1)

        for index, line in enumerate(lines):
            color = self.warning_color if warning_active and index >= 4 else self.text_color
            cv2.putText(
                frame,
                line,
                (16, 32 + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2,
                cv2.LINE_AA,
            )
        return frame

    @staticmethod
    def _to_xy(point: PointLike) -> Tuple[int, int]:
        if isinstance(point, Point):
            return int(round(point.x)), int(round(point.y))
        x, y = point
        return int(round(x)), int(round(y))

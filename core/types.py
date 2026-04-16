"""Shared data structures for the traffic congestion warning pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Point:
    """A 2D point in image or projected road coordinates."""

    x: float
    y: float


@dataclass(frozen=True)
class BBox:
    """Bounding box in xyxy format."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        """Box width in pixels or coordinate units."""

        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        """Box height in pixels or coordinate units."""

        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        """Box area."""

        return self.width * self.height

    @property
    def center(self) -> Point:
        """Center point of the box."""

        return Point(
            x=(self.x1 + self.x2) / 2.0,
            y=(self.y1 + self.y2) / 2.0,
        )

    @property
    def bottom_center(self) -> Point:
        """Bottom-center point, usually closer to the road contact point."""

        return Point(
            x=(self.x1 + self.x2) / 2.0,
            y=self.y2,
        )


@dataclass(frozen=True)
class Detection:
    """Single detector output before tracking."""

    bbox: BBox
    conf: float
    cls_id: int
    label: str


@dataclass(frozen=True)
class Track:
    """Tracked object state for one frame."""

    track_id: int
    bbox: BBox
    cls_id: int
    label: str
    score: float
    frame_index: int


@dataclass(frozen=True)
class FrameTrafficStats:
    """Traffic statistics and warning state for one video frame."""

    frame_index: int
    vehicle_count: int
    roi_vehicle_count: int
    mean_speed: float
    low_speed_ratio: float
    occupancy_ratio: float
    density: float
    congestion_level: str
    warning_active: bool

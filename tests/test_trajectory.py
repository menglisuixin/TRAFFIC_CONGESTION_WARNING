import pytest

from analytics.trajectory import TrajectoryStore
from core.types import BBox, Point, Track


def make_track(track_id: int, x1: float, y1: float, x2: float, y2: float, frame_index: int = 0) -> Track:
    return Track(
        track_id=track_id,
        bbox=BBox(x1, y1, x2, y2),
        cls_id=2,
        label="car",
        score=0.9,
        frame_index=frame_index,
    )


def test_update_uses_bottom_center_by_default() -> None:
    store = TrajectoryStore()

    store.update(make_track(1, 0, 0, 10, 20))

    assert store.get_points(1) == [Point(5, 20)]
    assert store.get_latest_point(1) == Point(5, 20)


def test_update_can_use_bbox_center() -> None:
    store = TrajectoryStore(use_bottom_center=False)

    store.update(make_track(1, 0, 0, 10, 20))

    assert store.get_latest_point(1) == Point(5, 10)


def test_max_length_truncates_old_points() -> None:
    store = TrajectoryStore(max_length=3)

    for index in range(5):
        store.update(make_track(1, index, 0, index + 2, 2, index))

    assert store.get_points(1) == [Point(3, 2), Point(4, 2), Point(5, 2)]


def test_empty_trajectory_returns_defaults() -> None:
    store = TrajectoryStore()

    assert store.get_points(99) == []
    assert store.get_latest_point(99) is None
    assert store.get_displacement(99) == 0.0
    assert store.get_total_displacement(99) == 0.0


def test_displacement_uses_latest_window() -> None:
    store = TrajectoryStore(max_length=10)
    for x, y in [(0, 0), (3, 4), (6, 8), (9, 12)]:
        store.update(make_track(1, x, y, x, y))

    assert store.get_displacement(1, window=2) == 5.0
    assert store.get_displacement(1, window=3) == 10.0
    assert store.get_displacement(1, window=1) == 0.0


def test_total_displacement_sums_path_length() -> None:
    store = TrajectoryStore(max_length=10)
    for x, y in [(0, 0), (3, 4), (6, 4)]:
        store.update(make_track(1, x, y, x, y))

    assert store.get_total_displacement(1) == 8.0


def test_prune_missing_and_clear() -> None:
    store = TrajectoryStore()
    store.update(make_track(1, 0, 0, 0, 0))
    store.update(make_track(2, 1, 1, 1, 1))

    store.prune_missing({2})

    assert store.get_points(1) == []
    assert store.get_points(2) == [Point(1, 1)]

    store.clear()

    assert store.get_points(2) == []


def test_max_length_must_be_positive() -> None:
    with pytest.raises(ValueError):
        TrajectoryStore(max_length=0)

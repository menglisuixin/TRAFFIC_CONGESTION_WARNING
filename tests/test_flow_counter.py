import pytest

from analytics.flow_counter import FlowCounter, LineSegment
from core.types import Point


def test_flow_counter_counts_crossing_once_per_track() -> None:
    counter = FlowCounter(LineSegment(Point(0, 0), Point(10, 0)))

    assert counter.update(1, Point(5, -1), Point(5, 1))
    assert not counter.update(1, Point(5, 1), Point(5, -1))

    assert counter.total_count == 1
    assert counter.counted_ids == {1}


def test_flow_counter_ignores_non_crossing_motion() -> None:
    counter = FlowCounter(LineSegment(Point(0, 0), Point(10, 0)))

    assert not counter.update(1, Point(2, 1), Point(8, 1))
    assert counter.total_count == 0
    assert counter.counted_ids == set()


def test_flow_counter_direction_forward() -> None:
    counter = FlowCounter(LineSegment(Point(0, 0), Point(10, 0)), direction="forward")

    assert counter.update(1, Point(5, -1), Point(5, 1))
    assert not counter.update(2, Point(5, 1), Point(5, -1))

    assert counter.total_count == 1
    assert counter.counted_ids == {1}


def test_flow_counter_direction_backward() -> None:
    counter = FlowCounter(LineSegment(Point(0, 0), Point(10, 0)), direction="backward")

    assert not counter.update(1, Point(5, -1), Point(5, 1))
    assert counter.update(2, Point(5, 1), Point(5, -1))

    assert counter.total_count == 1
    assert counter.counted_ids == {2}


def test_flow_counter_reset() -> None:
    counter = FlowCounter(LineSegment(Point(0, 0), Point(10, 0)))
    counter.update(1, Point(5, -1), Point(5, 1))

    counter.reset()

    assert counter.total_count == 0
    assert counter.counted_ids == set()


def test_flow_counter_rejects_invalid_direction() -> None:
    with pytest.raises(ValueError):
        FlowCounter(LineSegment(Point(0, 0), Point(10, 0)), direction="up")

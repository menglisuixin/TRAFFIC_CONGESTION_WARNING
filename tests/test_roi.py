import pytest

from analytics.roi import PolygonROI
from core.types import BBox, Point


def test_contains_point_accepts_tuple_and_point() -> None:
    roi = PolygonROI([(0, 0), (10, 0), Point(10, 10), Point(0, 10)])

    assert roi.contains_point((5, 5))
    assert roi.contains_point(Point(1, 1))
    assert not roi.contains_point((11, 5))


def test_contains_point_treats_boundary_as_inside() -> None:
    roi = PolygonROI([(0, 0), (10, 0), (10, 10), (0, 10)])

    assert roi.contains_point((0, 5))
    assert roi.contains_point((10, 10))


def test_contains_bbox_center() -> None:
    roi = PolygonROI([(0, 0), (10, 0), (10, 10), (0, 10)])

    assert roi.contains_bbox_center(BBox(2, 2, 4, 4))
    assert not roi.contains_bbox_center(BBox(12, 2, 14, 4))


def test_contains_bbox_bottom_center() -> None:
    roi = PolygonROI([(0, 0), (10, 0), (10, 10), (0, 10)])

    assert roi.contains_bbox_bottom_center(BBox(2, 2, 4, 8))
    assert not roi.contains_bbox_bottom_center(BBox(2, 8, 4, 12))


def test_polygon_requires_at_least_three_vertices() -> None:
    with pytest.raises(ValueError):
        PolygonROI([(0, 0), (1, 1)])

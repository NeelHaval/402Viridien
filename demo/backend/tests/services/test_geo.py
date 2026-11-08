import math

import pytest

from app.services.geo import haversine_km, point_in_polygon

SQUARE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
CONCAVE = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (5.0, 3.0), (0.0, 10.0)]     # a notch cut into the top edge


def test_inside_and_outside() -> None:
    assert point_in_polygon(5, 5, SQUARE) and not point_in_polygon(15, 5, SQUARE)
    assert not point_in_polygon(-0.1, 5, SQUARE) and not point_in_polygon(5, 10.1, SQUARE)


@pytest.mark.parametrize('point', [(5, 0), (10, 5), (5, 10), (0, 5)])
def test_point_on_an_edge_is_inside(point) -> None:
    assert point_in_polygon(*point, SQUARE)


@pytest.mark.parametrize('vertex', SQUARE)
def test_vertex_is_inside(vertex) -> None:
    assert point_in_polygon(*vertex, SQUARE)


def test_concave_notch_is_outside_but_arms_are_inside() -> None:
    assert not point_in_polygon(5, 8, CONCAVE)      # inside the notch
    assert point_in_polygon(1, 8, CONCAVE) and point_in_polygon(9, 8, CONCAVE)
    assert point_in_polygon(5, 1, CONCAVE)


def test_triangle_and_ray_through_vertex() -> None:
    triangle = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)]
    assert point_in_polygon(5, 5, triangle) and not point_in_polygon(1, 8, triangle)
    assert not point_in_polygon(-3, 0, triangle)     # ray along the bottom edge's line, outside


def test_winding_direction_does_not_matter() -> None:
    assert point_in_polygon(5, 5, SQUARE[::-1])


def test_haversine() -> None:
    assert haversine_km(27, -90, 27, -90) == 0
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.195, rel=1e-3)
    assert haversine_km(0, 179.5, 0, -179.5) == pytest.approx(111.195, rel=1e-3)
    assert haversine_km(10, 20, 30, 40) == pytest.approx(haversine_km(30, 40, 10, 20))
    assert not math.isnan(haversine_km(90, 0, -90, 0))

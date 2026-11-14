"""Analytical geometry checks, including contact and high-speed sweeps."""

import math
from dataclasses import FrozenInstanceError

import pytest

from neuroevolution.course import Circle, Course, Rectangle, slalom_course


def arena(obstacles=(), checkpoints=()):
    return Course(100, 100, (10, 50), Circle((90, 50), 3), obstacles, checkpoints)


def test_slalom_is_fixed_and_regions_are_clear():
    first, second = slalom_course(), slalom_course()
    assert first == second
    assert len(first.obstacles) == len(first.checkpoints) == 3
    assert not first.collides_circle(first.spawn, 8)
    assert not first.collides_circle(first.goal.center, 8)
    with pytest.raises(FrozenInstanceError):
        first.width = 100


def test_course_copies_mutable_input_containers():
    spawn = [10, 50]
    walls = [Rectangle(40, 0, 1, 70)]
    course = Course(100, 100, spawn, Circle([90, 50], 3), walls)
    walls.clear()
    spawn[0] = 80
    assert course.spawn == (10, 50)
    assert len(course.obstacles) == 1


@pytest.mark.parametrize(
    ("position", "radius", "collides"),
    [
        ((3, 5), 1, True),
        ((2.9, 5), 1, False),
        ((3.2, 3.2), 1, False),
        ((4, 5), 0, True),
        ((5, 5), 0.1, True),
    ],
)
def test_circle_rectangle_collision_uses_actual_corners(position, radius, collides):
    assert Rectangle(4, 4, 2, 2).intersects_circle(position, radius) is collides


def test_sweep_cannot_jump_through_a_thin_wall():
    course = arena((Rectangle(40, 0, 0.1, 90),))
    assert course.first_collision((20, 50), (90, 50), 1) == pytest.approx(19 / 70)


def test_rounded_corner_sweep_is_exact_instead_of_an_expanded_box():
    wall = Rectangle(4, 4, 2, 2)
    assert wall.collision_fraction((2, 2), (3.2, 3.2), 1) is None
    expected = (2 - 1 / math.sqrt(2)) / 3
    assert wall.collision_fraction((2, 2), (5, 5), 1) == pytest.approx(expected)
    assert wall.collision_fraction((2, 3), (8, 3), 1) == pytest.approx(1 / 3)


def test_boundary_sweep_stops_the_body_at_contact():
    course = arena()
    assert course.first_collision((50, 50), (150, 50), 2) == pytest.approx(0.48)
    assert course.first_collision((50, 50), (98, 50), 2) == 1
    assert course.first_collision((2, 50), (50, 50), 2) == 0
    assert course.first_collision((50, 50), (50, 50), 2) is None
    assert course.first_collision((50, 50), (60, 60), 2) is None


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(0, 90), (math.pi / 2, 50), (math.pi, 10), (-math.pi / 2, 50)],
)
def test_rays_measure_distance_to_arena_boundaries(angle, expected):
    assert arena().raycast((10, 50), angle, 200) == pytest.approx(expected)


def test_ray_uses_nearest_obstacle_ignores_walls_behind_and_caps_range():
    course = arena((Rectangle(40, 0, 2, 70), Rectangle(60, 0, 2, 70)))
    assert course.raycast((10, 50), 0, 100) == 30
    assert course.raycast((10, 50), math.pi, 100) == pytest.approx(10)
    assert course.raycast((10, 50), 0, 20) == 20
    assert course.raycast((41, 50), 0, 100) == 0
    assert course.raycast((0, 50), 0, 100) == 0


def test_diagonal_ray_hits_corner_at_known_distance():
    course = arena((Rectangle(40, 40, 10, 10),))
    assert course.raycast((20, 20), math.pi / 4, 100) == pytest.approx(
        20 * math.sqrt(2)
    )


def test_segment_detects_a_goal_crossed_between_endpoints():
    goal = Circle((50, 50), 5)
    assert goal.entry_fraction((10, 50), (90, 50)) == pytest.approx(35 / 80)
    assert goal.entry_fraction((50, 50), (90, 50)) == 0
    assert goal.entry_fraction((10, 10), (10, 10)) is None


def test_checkpoints_are_collected_in_movement_order():
    regions = (Circle((30, 50), 2), Circle((60, 50), 2))
    assert arena(checkpoints=regions).advance_checkpoints((10, 50), (80, 50), 0) == 2
    reverse = arena(checkpoints=regions[::-1])
    assert reverse.advance_checkpoints((10, 50), (80, 50), 0) == 1
    assert reverse.advance_checkpoints((10, 10), (80, 10), 0) == 0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Rectangle(0, 0, 0, 10),
        lambda: Rectangle(math.nan, 0, 10, 10),
        lambda: Circle((0, math.inf), 1),
        lambda: Circle((1, 1), -1),
        lambda: Course(0, 100, (10, 10), Circle((90, 50), 3)),
        lambda: Course(100, 100, (0, 50), Circle((90, 50), 3)),
        lambda: arena((Rectangle(90, 0, 20, 20),)),
        lambda: Course(100, 100, (10, 50), Circle((99, 50), 3)),
        lambda: arena((Rectangle(5, 45, 10, 10),)),
    ],
)
def test_invalid_geometry_is_rejected(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize(
    "query",
    [
        lambda c: c.raycast((-1, 50), 0, 10),
        lambda c: c.raycast((10, 50), math.nan, 10),
        lambda c: c.raycast((10, 50), 0, 0),
        lambda c: c.first_collision((10, 50), (20, 50), -1),
        lambda c: c.advance_checkpoints((10, 50), (20, 50), 1),
    ],
)
def test_invalid_geometry_queries_are_rejected(query):
    with pytest.raises(ValueError):
        query(arena())

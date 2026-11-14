"""Movement, terminal-state, and full-course traversal checks."""

import math

import numpy as np
import pytest

from neuroevolution.agent import (
    Action,
    AgentConfig,
    AgentState,
    spawn_agent,
    step_agent,
)
from neuroevolution.course import Circle, Course, Rectangle, slalom_course


def arena(obstacles=(), goal=None):
    return Course(100, 100, (10, 50), goal or Circle((90, 50), 3), obstacles)


def test_forward_movement_uses_fixed_timestep_and_preserves_previous_state():
    course = arena()
    config = AgentConfig(radius=2, max_speed=20, timestep=0.5)
    initial = spawn_agent(course, config)
    moved = step_agent(course, initial, Action(0, 1), config)
    assert moved.position == pytest.approx((20, 50))
    assert moved.speed == 20
    assert moved.steps == 1
    assert moved.distance_travelled == pytest.approx(10)
    assert initial.position == (10, 50)
    assert initial.steps == 0


def test_turning_is_counterclockwise_and_can_happen_in_place():
    course = arena()
    config = AgentConfig(radius=2, max_speed=20, timestep=0.5, turn_rate=math.pi)
    state = spawn_agent(course, config)
    turned = step_agent(course, state, Action(1, 0), config)
    assert turned.position == state.position
    assert turned.heading == pytest.approx(math.pi / 2)
    moved = step_agent(course, turned, Action(0, 1), config)
    assert moved.position == pytest.approx((10, 60))


def test_high_speed_cannot_skip_a_wall_or_collect_checkpoints_beyond_it():
    course = Course(
        100,
        100,
        (10, 50),
        Circle((90, 50), 3),
        (Rectangle(40, 0, 0.1, 90),),
        (Circle((70, 50), 2),),
    )
    config = AgentConfig(radius=2, max_speed=200, timestep=1)
    state = step_agent(course, spawn_agent(course, config), Action(0, 1), config)
    assert state.status == "crashed"
    assert state.position == pytest.approx((38, 50))
    assert state.distance_travelled == pytest.approx(28)
    assert state.checkpoints_passed == 0
    assert state.speed == 0
    assert step_agent(course, state, Action(1, 1), config) is state


def test_goal_crossed_between_frames_wins_before_a_later_boundary_collision():
    course = arena()
    config = AgentConfig(radius=2, max_speed=200, timestep=1)
    state = step_agent(course, spawn_agent(course, config), Action(0, 1), config)
    assert state.status == "reached"
    assert state.position == pytest.approx((87, 50))
    assert state.speed == 0
    assert step_agent(course, state, Action(), config) is state


def test_arena_boundary_stops_the_body():
    course = arena(goal=Circle((90, 80), 3))
    config = AgentConfig(radius=2, max_speed=200, timestep=1)
    state = step_agent(course, spawn_agent(course, config), Action(0, 1), config)
    assert state.status == "crashed"
    assert state.position == pytest.approx((98, 50))


def test_episode_limit_ends_stationary_attempts_and_does_not_advance_again():
    course = arena()
    config = AgentConfig(max_steps=2)
    state = spawn_agent(course, config)
    state = step_agent(course, state, Action(), config)
    assert state.status == "active"
    state = step_agent(course, state, Action(), config)
    assert state.status == "timed_out"
    assert state.steps == 2
    assert state.position == course.spawn
    assert step_agent(course, state, Action(), config) is state


def test_goal_on_last_allowed_step_is_successful():
    course = arena()
    config = AgentConfig(radius=2, max_speed=77, timestep=1, max_steps=1)
    state = step_agent(course, spawn_agent(course, config), Action(0, 1), config)
    assert state.status == "reached"
    assert state.steps == 1


def test_spawn_inside_goal_is_already_finished_and_large_bodies_are_rejected():
    course = Course(100, 100, (90, 50), Circle((90, 50), 3))
    assert spawn_agent(course).status == "reached"
    with pytest.raises(ValueError, match="does not fit"):
        spawn_agent(arena(), AgentConfig(radius=10))


def test_scripted_route_proves_default_course_is_traversable_within_the_limit():
    course, config = slalom_course(), AgentConfig()
    state = spawn_agent(course, config)
    # Cardinal legs and stationary quarter-turns, not a learned controller.
    segments = [
        (0, 1, 32),
        (1, 0, 30),
        (0, 1, 52),
        (-1, 0, 30),
        (0, 1, 46),
        (-1, 0, 30),
        (0, 1, 104),
        (1, 0, 30),
        (0, 1, 50),
        (1, 0, 30),
        (0, 1, 104),
        (-1, 0, 30),
        (0, 1, 40),
        (-1, 0, 30),
        (0, 1, 52),
        (1, 0, 30),
        (0, 1, 16),
    ]
    for steering, throttle, count in segments:
        for _ in range(count):
            state = step_agent(course, state, Action(steering, throttle), config)
            assert state.status != "crashed"
    assert state.status == "reached"
    assert state.steps < config.max_steps
    assert state.checkpoints_passed == 3


def test_same_actions_produce_identical_independent_trajectories():
    course = slalom_course()
    first, second = spawn_agent(course), spawn_agent(course)
    for action in [Action(0.2, 0.5)] * 20:
        first = step_agent(course, first, action)
        second = step_agent(course, second, action)
        assert first == second


@pytest.mark.parametrize(
    ("outputs", "expected"),
    [
        ([0, -1], Action(0, 0)),
        ([0, 0], Action(0, 0.5)),
        ([-1, 1], Action(-1, 1)),
        ([-5, 5], Action(-1, 1)),
    ],
)
def test_signed_network_outputs_map_to_bounded_controls(outputs, expected):
    assert Action.from_network(outputs) == expected


@pytest.mark.parametrize("outputs", [[0], [[0, 0]], [np.nan, 0], [0, np.inf], [1j, 0]])
def test_invalid_controller_outputs_are_rejected(outputs):
    with pytest.raises(ValueError):
        Action.from_network(outputs)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Action(2, 0),
        lambda: Action(0, -1),
        lambda: Action(math.nan, 0),
        lambda: AgentConfig(radius=0),
        lambda: AgentConfig(max_speed=math.inf),
        lambda: AgentConfig(timestep=-1),
        lambda: AgentConfig(max_steps=0),
        lambda: AgentConfig(max_steps=1.5),
        lambda: AgentState((math.nan, 0)),
        lambda: AgentState((10, 10), speed=-1),
        lambda: AgentState((10, 10), steps=-1),
        lambda: AgentState((10, 10), status="unknown"),
    ],
)
def test_invalid_actions_configuration_and_states_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()

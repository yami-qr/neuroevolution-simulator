"""Analytical sensor distances, local coordinates, and observation layout."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from neuroevolution import FeedForwardNetwork
from neuroevolution.agent import (
    Action,
    AgentConfig,
    AgentState,
    spawn_agent,
    step_agent,
)
from neuroevolution.course import Circle, Course, Rectangle, slalom_course
from neuroevolution.sensors import SensorConfig, observe, sense


def test_default_observation_matches_analytical_slalom_distances():
    course = slalom_course()
    observation = observe(course, spawn_agent(course))
    diagonal_ray = 170 / (math.cos(math.pi / 6) * 200)
    expected = [
        1,
        1,
        diagonal_ray,
        0.85,
        diagonal_ray,
        1,
        1,
        760 / math.hypot(900, 600),
        0,
        0,
    ]
    assert observation.shape == (10,)
    assert observation.dtype == np.float64
    assert_allclose(observation, expected)
    assert np.isfinite(observation).all()


def test_rays_rotate_with_heading_and_provide_world_space_endpoints():
    course = Course(200, 100, (50, 30), Circle((180, 50), 5))
    config = SensorConfig(ray_angles=(0,), max_distance=200)
    east = sense(course, AgentState((50, 30)), config)[0]
    north = sense(course, AgentState((50, 30), heading=math.pi / 2), config)[0]
    south = sense(course, AgentState((50, 30), heading=-math.pi / 2), config)[0]
    assert east.distance == pytest.approx(150)
    assert east.endpoint == pytest.approx((200, 30))
    assert north.distance == pytest.approx(70)
    assert north.endpoint == pytest.approx((50, 100))
    assert south.distance == pytest.approx(30)
    assert south.endpoint == pytest.approx((50, 0))


def test_goal_vector_is_local_and_speed_is_normalized():
    course = Course(200, 100, (50, 30), Circle((180, 50), 5))
    state = AgentState((50, 30), heading=math.pi / 2, speed=120)
    observation = observe(course, state, sensor_config=SensorConfig(ray_angles=(0,)))
    diagonal = math.hypot(200, 100)
    assert_allclose(observation[-3:], [20 / diagonal, -130 / diagonal, 0.5])
    assert observation.shape == (4,)


def test_distances_update_after_movement_and_ignore_goal_and_checkpoints():
    course = Course(
        100,
        100,
        (10, 50),
        Circle((90, 80), 3),
        (Rectangle(40, 40, 1, 20),),
        (Circle((20, 50), 2),),
    )
    agent_config = AgentConfig(radius=2, max_speed=20, timestep=0.5)
    sensors = SensorConfig(ray_angles=(0,), max_distance=100)
    state = spawn_agent(course, agent_config)
    assert sense(course, state, sensors)[0].distance == 30
    moved = step_agent(course, state, Action(0, 1), agent_config)
    assert sense(course, moved, sensors)[0].distance == 20
    assert state.position == (10, 50)


def test_default_observation_connects_to_existing_controller_and_is_independent():
    course = slalom_course()
    state = spawn_agent(course)
    network = FeedForwardNetwork((10, 12, 2), seed=42)
    observation = observe(course, state)
    action = Action.from_network(network.forward(observation))
    assert -1 <= action.steering <= 1
    assert 0 <= action.throttle <= 1
    expected = observation.copy()
    observation[:] = -99
    assert_array_equal(observe(course, state), expected)


def test_sensor_configuration_copies_angle_containers():
    angles = [0, math.pi / 2]
    config = SensorConfig(ray_angles=angles)
    angles.clear()
    assert config.ray_angles == (0, math.pi / 2)


@pytest.mark.parametrize(
    "config",
    [
        lambda: SensorConfig(ray_angles=()),
        lambda: SensorConfig(ray_angles=(math.nan,)),
        lambda: SensorConfig(max_distance=0),
        lambda: SensorConfig(max_distance=math.inf),
    ],
)
def test_invalid_sensor_configuration_is_rejected(config):
    with pytest.raises(ValueError):
        config()


def test_speed_outside_the_configured_range_is_rejected():
    course = slalom_course()
    with pytest.raises(ValueError, match="maximum speed"):
        observe(course, AgentState(course.spawn, speed=241))

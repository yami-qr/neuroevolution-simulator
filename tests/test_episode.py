"""End-to-end episodes using real neural controllers and sensor feedback."""

import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from neuroevolution import FeedForwardNetwork
from neuroevolution.agent import AgentConfig
from neuroevolution.course import Circle, Course, Rectangle, slalom_course
from neuroevolution.episode import run_episode
from neuroevolution.sensors import SensorConfig


def constant_controller(steering=0, signed_throttle=1):
    controller = FeedForwardNetwork((10, 2), output_activation="linear")
    controller.set_parameters([np.zeros((10, 2))], [[steering, signed_throttle]])
    return controller


def test_neural_controller_reaches_goal_and_records_complete_history():
    course = Course(100, 100, (10, 50), Circle((90, 50), 3))
    config = AgentConfig(radius=2, max_speed=10, timestep=1, max_steps=10)
    result = run_episode(
        course, constant_controller(), agent_config=config, record=True
    )
    assert result.final_state.status == "reached"
    assert result.final_state.position == pytest.approx((87, 50))
    assert result.elapsed_seconds == 8
    assert len(result.trajectory) == result.final_state.steps + 1
    assert result.trajectory[0].position == course.spawn
    assert result.trajectory[-1] == result.final_state


def test_obstacle_ends_a_neural_episode_before_the_goal():
    course = Course(100, 100, (10, 50), Circle((90, 50), 3), (Rectangle(40, 0, 1, 90),))
    config = AgentConfig(radius=2, max_speed=10, timestep=1)
    result = run_episode(course, constant_controller(), agent_config=config)
    assert result.final_state.status == "crashed"
    assert result.final_state.position == pytest.approx((38, 50))
    assert result.elapsed_seconds == 3
    assert result.trajectory == ()


def test_stationary_neural_episode_is_bounded_by_tick_limit():
    config = AgentConfig(max_steps=3)
    result = run_episode(
        slalom_course(), constant_controller(signed_throttle=-1), agent_config=config
    )
    assert result.final_state.status == "timed_out"
    assert result.final_state.steps == 3
    assert result.elapsed_seconds == pytest.approx(3 / 60)


def test_same_seeded_controller_reproduces_trajectory_without_mutating_genome():
    controller = FeedForwardNetwork((10, 12, 2), seed=42)
    genome = controller.to_genome()
    course = slalom_course()
    first = run_episode(course, controller, record=True)
    second = run_episode(course, controller, record=True)
    assert first == second
    assert first.final_state.status in ("reached", "crashed", "timed_out")
    assert first.final_state.steps <= 900
    assert_array_equal(controller.to_genome(), genome)


def test_controller_steering_uses_current_sensor_feedback():
    course = Course(100, 100, (10, 50), Circle((90, 80), 3))
    controller = FeedForwardNetwork((10, 2), output_activation="linear")
    weights = np.zeros((10, 2))
    weights[3, 0] = 1  # Steering responds to the forward ray distance.
    controller.set_parameters([weights], [[0, 1]])
    config = AgentConfig(radius=2, max_speed=10, timestep=0.1, max_steps=2)
    result = run_episode(course, controller, agent_config=config, record=True)
    first = result.trajectory[1]
    assert first.heading == pytest.approx((90 / 200) * math.pi * 0.1)
    assert first.position[1] > course.spawn[1]
    assert result.final_state.status == "timed_out"


def test_spawn_inside_goal_has_zero_elapsed_time_and_one_recorded_state():
    course = Course(100, 100, (90, 50), Circle((90, 50), 3))
    result = run_episode(course, constant_controller(), record=True)
    assert result.final_state.status == "reached"
    assert result.elapsed_seconds == 0
    assert len(result.trajectory) == 1


@pytest.mark.parametrize("sizes", [(9, 2), (10, 1), (11, 3)])
def test_incompatible_controller_shapes_are_rejected_before_running(sizes):
    with pytest.raises(ValueError, match="10 inputs and 2 outputs"):
        run_episode(slalom_course(), FeedForwardNetwork(sizes))


def test_custom_ray_fan_works_with_matching_controller():
    sensors = SensorConfig(ray_angles=(0,))
    config = AgentConfig(max_steps=1)
    controller = FeedForwardNetwork((4, 3, 2), seed=42)
    result = run_episode(
        slalom_course(), controller, sensor_config=sensors, agent_config=config
    )
    assert result.final_state.steps == 1
    assert result.final_state.status == "timed_out"

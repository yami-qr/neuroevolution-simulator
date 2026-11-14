"""Agent-local obstacle rays and normalized neural-controller observations."""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neuroevolution._validation import Point, finite_float, positive_float
from neuroevolution.agent import AgentConfig, AgentState
from neuroevolution.course import Course


@dataclass(frozen=True, slots=True)
class SensorConfig:
    """Default fan: seven rays from the right side through forward to the left."""

    ray_angles: tuple[float, ...] = (
        -math.pi / 2,
        -math.pi / 3,
        -math.pi / 6,
        0,
        math.pi / 6,
        math.pi / 3,
        math.pi / 2,
    )
    max_distance: float = 200.0

    def __post_init__(self) -> None:
        angles = tuple(finite_float(angle, "Ray angle") for angle in self.ray_angles)
        if not angles:
            raise ValueError("At least one sensor ray is required.")
        object.__setattr__(self, "ray_angles", angles)
        object.__setattr__(
            self, "max_distance", positive_float(self.max_distance, "Sensor range")
        )


@dataclass(frozen=True, slots=True)
class RayReading:
    """A world-space direction, raw distance, and endpoint for visualization."""

    angle: float
    distance: float
    endpoint: Point


_DEFAULT_AGENT_CONFIG = AgentConfig()
_DEFAULT_SENSOR_CONFIG = SensorConfig()


def sense(
    course: Course, state: AgentState, config: SensorConfig = _DEFAULT_SENSOR_CONFIG
) -> tuple[RayReading, ...]:
    """Cast rays from the agent's center against obstacles and arena boundaries.

    Distances measure from the center, not from the circular body's surface.
    The goal and other agents do not obstruct rays.
    """
    readings = []
    for relative_angle in config.ray_angles:
        angle = state.heading + relative_angle
        distance = course.raycast(state.position, angle, config.max_distance)
        endpoint = (
            state.position[0] + math.cos(angle) * distance,
            state.position[1] + math.sin(angle) * distance,
        )
        readings.append(RayReading(angle, distance, endpoint))
    return tuple(readings)


def observe(
    course: Course,
    state: AgentState,
    agent_config: AgentConfig = _DEFAULT_AGENT_CONFIG,
    sensor_config: SensorConfig = _DEFAULT_SENSOR_CONFIG,
) -> NDArray[np.float64]:
    """Build [ray distances..., local goal forward, local goal left, speed].

    Ray distances divide by their range; speed divides by maximum speed.
    Goal displacement is rotated into the agent's frame and divided by the
    arena diagonal, retaining both direction and distance in [-1, 1]. Ordered
    scoring checkpoints are deliberately not provided as navigation inputs.
    """
    if state.speed > agent_config.max_speed:
        raise ValueError("Agent speed exceeds the configured maximum speed.")
    readings = sense(course, state, sensor_config)
    dx = course.goal.center[0] - state.position[0]
    dy = course.goal.center[1] - state.position[1]
    cosine, sine = math.cos(state.heading), math.sin(state.heading)
    diagonal = math.hypot(course.width, course.height)
    return np.array(
        [
            *(reading.distance / sensor_config.max_distance for reading in readings),
            (cosine * dx + sine * dy) / diagonal,
            (-sine * dx + cosine * dy) / diagonal,
            state.speed / agent_config.max_speed,
        ],
        dtype=np.float64,
    )

"""Deterministic kinematic agents with bounded controls and terminal states."""

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from neuroevolution._validation import (
    Point,
    finite_float,
    nonnegative_float,
    nonnegative_int,
    point,
    positive_float,
)
from neuroevolution.course import Course

AgentStatus = Literal["active", "reached", "crashed", "timed_out"]


@dataclass(frozen=True, slots=True)
class AgentConfig:
    radius: float = 8.0
    max_speed: float = 240.0
    turn_rate: float = math.pi
    timestep: float = 1.0 / 60.0
    max_steps: int = 900

    def __post_init__(self) -> None:
        for name in ("radius", "max_speed", "turn_rate", "timestep"):
            object.__setattr__(self, name, positive_float(getattr(self, name), name))
        limit = nonnegative_int(self.max_steps, "max_steps")
        if limit == 0:
            raise ValueError("max_steps must be positive.")
        object.__setattr__(self, "max_steps", limit)


@dataclass(frozen=True, slots=True)
class Action:
    """Steering in [-1, 1] and forward throttle in [0, 1]."""

    steering: float = 0.0
    throttle: float = 0.0

    def __post_init__(self) -> None:
        steering = finite_float(self.steering, "Steering")
        throttle = finite_float(self.throttle, "Throttle")
        if not -1 <= steering <= 1 or not 0 <= throttle <= 1:
            raise ValueError("Steering must be in [-1, 1] and throttle in [0, 1].")
        object.__setattr__(self, "steering", steering)
        object.__setattr__(self, "throttle", throttle)

    @classmethod
    def from_network(cls, outputs: ArrayLike) -> "Action":
        """Map two signed network outputs to safe steering and throttle.

        The first output steers, the second maps [-1, 1] to [0, 1]. Values
        outside [-1, 1] saturate, allowing controllers with linear outputs too.
        """
        values = np.asarray(outputs)
        if values.shape != (2,) or values.dtype.kind not in "biuf":
            raise ValueError(
                "Controller outputs must be a real numeric vector of size 2."
            )
        values = np.asarray(values, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Controller outputs must be finite.")
        steering, throttle = np.clip(values, -1.0, 1.0)
        return cls(float(steering), float((throttle + 1.0) / 2.0))


@dataclass(frozen=True, slots=True)
class AgentState:
    position: Point
    heading: float = 0.0
    speed: float = 0.0
    steps: int = 0
    status: AgentStatus = "active"
    checkpoints_passed: int = 0
    distance_travelled: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", point(self.position))
        heading = finite_float(self.heading, "Heading")
        object.__setattr__(self, "heading", (heading + math.pi) % math.tau - math.pi)
        for name in ("speed", "distance_travelled"):
            object.__setattr__(self, name, nonnegative_float(getattr(self, name), name))
        for name in ("steps", "checkpoints_passed"):
            object.__setattr__(self, name, nonnegative_int(getattr(self, name), name))
        if self.status not in ("active", "reached", "crashed", "timed_out"):
            raise ValueError("Unknown agent status.")


def spawn_agent(course: Course, config: AgentConfig = AgentConfig()) -> AgentState:
    """Start a fresh, independent agent and verify its body fits at spawn."""
    if course.collides_circle(course.spawn, config.radius):
        raise ValueError("Agent body does not fit at the course spawn.")
    return AgentState(
        position=course.spawn,
        heading=course.spawn_heading,
        status="reached" if course.goal.contains(course.spawn) else "active",
        checkpoints_passed=course.advance_checkpoints(course.spawn, course.spawn, 0),
    )


def step_agent(
    course: Course,
    state: AgentState,
    action: Action,
    config: AgentConfig = AgentConfig(),
) -> AgentState:
    """Turn, then sweep a straight movement segment for one fixed timestep.

    Throttle immediately sets speed; an agent can rotate while stationary.
    The earliest goal or wall contact ends an attempt, with collisions winning
    ties. Terminal agents stay unchanged. Agents never collide with each other.
    """
    if state.status != "active":
        return state
    if state.steps >= config.max_steps:
        return replace(state, status="timed_out", speed=0.0)
    heading = state.heading + action.steering * config.turn_rate * config.timestep
    speed = action.throttle * config.max_speed
    distance = speed * config.timestep
    end = (
        state.position[0] + math.cos(heading) * distance,
        state.position[1] + math.sin(heading) * distance,
    )
    collision = course.first_collision(state.position, end, config.radius)
    goal = course.goal.entry_fraction(state.position, end)
    fraction, status = 1.0, "active"
    if collision is not None and (goal is None or collision <= goal):
        fraction, status = collision, "crashed"
    elif goal is not None:
        fraction, status = goal, "reached"

    position = (
        state.position[0] + (end[0] - state.position[0]) * fraction,
        state.position[1] + (end[1] - state.position[1]) * fraction,
    )
    steps = state.steps + 1
    if status == "active" and steps >= config.max_steps:
        status = "timed_out"
    return AgentState(
        position=position,
        heading=heading,
        speed=speed if status == "active" else 0.0,
        steps=steps,
        status=status,
        checkpoints_passed=course.advance_checkpoints(
            state.position, position, state.checkpoints_passed
        ),
        distance_travelled=state.distance_travelled
        + math.hypot(position[0] - state.position[0], position[1] - state.position[1]),
    )

"""Bounded, deterministic episodes connecting sensors, controllers, and motion."""

from dataclasses import dataclass

from neuroevolution.agent import (
    Action,
    AgentConfig,
    AgentState,
    spawn_agent,
    step_agent,
)
from neuroevolution.course import Course
from neuroevolution.network import FeedForwardNetwork
from neuroevolution.sensors import SensorConfig, observe

_DEFAULT_AGENT_CONFIG = AgentConfig()
_DEFAULT_SENSOR_CONFIG = SensorConfig()


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    final_state: AgentState
    elapsed_seconds: float
    trajectory: tuple[AgentState, ...] = ()


def run_episode(
    course: Course,
    controller: FeedForwardNetwork,
    *,
    agent_config: AgentConfig = _DEFAULT_AGENT_CONFIG,
    sensor_config: SensorConfig = _DEFAULT_SENSOR_CONFIG,
    record: bool = False,
) -> EpisodeResult:
    """Evaluate one unchanged controller until success, collision, or timeout.

    Elapsed time is measured in fixed simulation ticks, including the tick of
    a terminal contact. Recording includes spawn and terminal states, and is
    disabled by default to avoid retaining every agent's complete trajectory.
    """
    input_size = len(sensor_config.ray_angles) + 3
    if controller.layer_sizes[0] != input_size or controller.layer_sizes[-1] != 2:
        raise ValueError(
            f"Controller must have {input_size} inputs and 2 outputs for these sensors."
        )
    state = spawn_agent(course, agent_config)
    history = [state] if record else []
    while state.status == "active":
        observation = observe(course, state, agent_config, sensor_config)
        action = Action.from_network(controller.forward(observation))
        state = step_agent(course, state, action, agent_config)
        if record:
            history.append(state)
    return EpisodeResult(
        final_state=state,
        elapsed_seconds=state.steps * agent_config.timestep,
        trajectory=tuple(history),
    )

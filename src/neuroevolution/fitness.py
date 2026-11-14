"""Explainable episode fitness based on ordered course progress and completion."""

import math
from dataclasses import dataclass

from neuroevolution._validation import nonnegative_float
from neuroevolution.agent import AgentConfig
from neuroevolution.course import Course
from neuroevolution.episode import EpisodeResult


@dataclass(frozen=True, slots=True)
class FitnessConfig:
    progress_weight: float = 1000.0
    completion_bonus: float = 1000.0
    time_penalty: float = 10.0
    collision_penalty: float = 25.0

    def __post_init__(self) -> None:
        for name in (
            "progress_weight",
            "completion_bonus",
            "time_penalty",
            "collision_penalty",
        ):
            object.__setattr__(self, name, nonnegative_float(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class FitnessBreakdown:
    progress: float
    progress_reward: float
    completion_reward: float
    time_penalty: float
    collision_penalty: float

    @property
    def total(self) -> float:
        return (
            self.progress_reward
            + self.completion_reward
            - self.time_penalty
            - self.collision_penalty
        )


_DEFAULT_AGENT_CONFIG = AgentConfig()
_DEFAULT_FITNESS_CONFIG = FitnessConfig()


def score_episode(
    course: Course,
    result: EpisodeResult,
    *,
    agent_config: AgentConfig = _DEFAULT_AGENT_CONFIG,
    config: FitnessConfig = _DEFAULT_FITNESS_CONFIG,
) -> FitnessBreakdown:
    """Score a terminal attempt without rewarding survival or travelled distance.

    Each checkpoint and the goal form one stage. Closeness to the next region
    is measured relative to spawn or the previous checkpoint center and clipped
    to [0, 1]. Completion sets full progress even if a valid alternate route
    bypassed checkpoint regions. The default completion bonus exceeds all
    possible time costs, so any success outranks any default failure.
    """
    state = result.final_state
    if state.status == "active":
        raise ValueError("Only terminal episodes can be scored.")
    if state.checkpoints_passed > len(course.checkpoints):
        raise ValueError(
            "Episode checkpoint count exceeds the course checkpoint count."
        )
    if state.steps > agent_config.max_steps:
        raise ValueError("Episode exceeds the configured step limit.")
    elapsed = nonnegative_float(result.elapsed_seconds, "Elapsed time")
    if not math.isclose(
        elapsed, state.steps * agent_config.timestep, rel_tol=1e-9, abs_tol=1e-12
    ):
        raise ValueError("Episode elapsed time does not match its simulation ticks.")

    if state.status == "reached":
        progress = 1.0
    else:
        passed = state.checkpoints_passed
        target = (*course.checkpoints, course.goal)[passed]
        origin = course.spawn if passed == 0 else course.checkpoints[passed - 1].center
        baseline = max(math.dist(origin, target.center) - target.radius, math.ulp(1.0))
        remaining = max(0.0, math.dist(state.position, target.center) - target.radius)
        partial = max(0.0, min(1.0, 1.0 - remaining / baseline))
        progress = (passed + partial) / (len(course.checkpoints) + 1)

    breakdown = FitnessBreakdown(
        progress=progress,
        progress_reward=progress * config.progress_weight,
        completion_reward=config.completion_bonus if state.status == "reached" else 0.0,
        time_penalty=config.time_penalty * state.steps / agent_config.max_steps,
        collision_penalty=config.collision_penalty
        if state.status == "crashed"
        else 0.0,
    )
    if not math.isfinite(breakdown.total):
        raise ValueError("Fitness weights produced a non-finite score.")
    return breakdown

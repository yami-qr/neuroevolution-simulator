"""Known fitness values and guards against rewarding ineffective movement."""

import math

import pytest

from neuroevolution.agent import AgentConfig, AgentState
from neuroevolution.course import Circle, Course
from neuroevolution.episode import EpisodeResult
from neuroevolution.fitness import FitnessConfig, score_episode

CONFIG = AgentConfig(radius=2, timestep=1, max_steps=100)


def arena(checkpoints=()):
    return Course(100, 100, (10, 50), Circle((90, 50), 3), checkpoints=checkpoints)


def episode(position, *, status="crashed", steps=50, passed=0, distance=0):
    state = AgentState(
        position,
        status=status,
        steps=steps,
        checkpoints_passed=passed,
        distance_travelled=distance,
    )
    return EpisodeResult(state, steps)


def test_halfway_progress_and_penalties_match_hand_calculated_score():
    score = score_episode(arena(), episode((48.5, 50)), agent_config=CONFIG)
    assert score.progress == pytest.approx(0.5)
    assert score.progress_reward == pytest.approx(500)
    assert score.completion_reward == 0
    assert score.time_penalty == 5
    assert score.collision_penalty == 25
    assert score.total == pytest.approx(470)


def test_progress_targets_next_checkpoint_instead_of_the_final_goal():
    course = arena((Circle((30, 80), 5),))
    towards_checkpoint = score_episode(course, episode((20, 65)), agent_config=CONFIG)
    towards_goal = score_episode(course, episode((70, 50)), agent_config=CONFIG)
    assert towards_checkpoint.total > towards_goal.total
    assert towards_checkpoint.progress < 0.5
    assert towards_goal.progress == 0
    after_checkpoint = score_episode(
        course, episode((30, 80), passed=1), agent_config=CONFIG
    )
    assert after_checkpoint.progress == pytest.approx(0.5)


def test_success_gets_full_progress_and_outscores_every_default_failure():
    course = arena((Circle((30, 80), 5),))
    success = score_episode(
        course, episode((87, 50), status="reached", steps=100), agent_config=CONFIG
    )
    assert success.progress == 1
    assert success.total == 1990
    failure = score_episode(
        course, episode((87, 50), passed=1, status="timed_out"), agent_config=CONFIG
    )
    assert failure.total <= 1000
    assert success.total > failure.total


def test_faster_completion_scores_higher():
    fast = score_episode(
        arena(), episode((87, 50), status="reached", steps=10), agent_config=CONFIG
    )
    slow = score_episode(
        arena(), episode((87, 50), status="reached", steps=90), agent_config=CONFIG
    )
    assert fast.total == 1999
    assert fast.total > slow.total


def test_distance_travelled_is_not_a_reward_and_loitering_costs_time():
    course = arena()
    first = score_episode(course, episode((40, 50), distance=30), agent_config=CONFIG)
    looping = score_episode(
        course, episode((40, 50), distance=3000), agent_config=CONFIG
    )
    later = score_episode(course, episode((40, 50), steps=90), agent_config=CONFIG)
    assert first.total == looping.total
    assert first.total > later.total


def test_backwards_movement_does_not_give_negative_progress():
    score = score_episode(arena(), episode((5, 50)), agent_config=CONFIG)
    assert score.progress == 0


def test_overlapping_stage_regions_keep_progress_finite():
    course = arena((Circle((10, 50), 5), Circle((10, 50), 5)))
    score = score_episode(course, episode((10, 50), passed=1), agent_config=CONFIG)
    assert math.isfinite(score.total)
    assert 0 <= score.progress <= 1


def test_zero_weights_are_allowed_for_control_experiments():
    score = score_episode(
        arena(),
        episode((40, 50)),
        agent_config=CONFIG,
        config=FitnessConfig(0, 0, 0, 0),
    )
    assert score.total == 0


@pytest.mark.parametrize("weight", [math.nan, math.inf, -1, True])
def test_invalid_fitness_configuration_is_rejected(weight):
    with pytest.raises(ValueError):
        FitnessConfig(progress_weight=weight)


@pytest.mark.parametrize(
    "result",
    [
        episode((40, 50), status="active"),
        episode((40, 50), passed=1),
        episode((40, 50), steps=101),
        EpisodeResult(AgentState((40, 50), status="crashed"), 1),
        EpisodeResult(AgentState((40, 50), status="crashed"), math.nan),
    ],
)
def test_incompatible_or_incomplete_episodes_are_rejected(result):
    with pytest.raises(ValueError):
        score_episode(arena(), result, agent_config=CONFIG)

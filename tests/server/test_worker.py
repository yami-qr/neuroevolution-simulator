"""Real engine execution through persisted jobs, with failure and cancellation."""

import threading

import pytest
from fastapi.testclient import TestClient

from neuroevolution import (
    EvolutionRunner,
    FeedForwardNetwork,
    run_episode,
    score_episode,
)
from neuroevolution.server.api import create_app
from neuroevolution.server.repository import Repository
from neuroevolution.server.schemas import (
    ExperimentCreate,
    ExperimentSettings,
    RunCreate,
)
from neuroevolution.server.worker import run_once


@pytest.fixture
def queued(database):
    repository = Repository(database)
    request = ExperimentCreate(
        name="Worker test",
        settings={
            "generations": 2,
            "evolution": {"population_size": 2, "elite_count": 1, "tournament_size": 2},
            "agent": {"max_steps": 3},
        },
    )
    experiment = repository.create_experiment(request)
    run = repository.create_run(experiment.id, RunCreate())
    return repository, run


def test_worker_persists_generation_results_and_replay_accessible_from_api(
    queued, database
):
    repo, queued_run = queued
    result = run_once(repo)
    assert result.id == queued_run.id
    assert result.status == "completed"
    assert result.completed_generations == 2
    assert result.environment["numpy"]
    assert run_once(repo) is None
    with TestClient(create_app(database)) as client:
        details = client.get(f"/api/v1/runs/{result.id}").json()
        assert details["status"] == "completed" and details["progress"] == 1
        history = client.get(f"/api/v1/runs/{result.id}/generations").json()
        assert [row["generation"] for row in history["items"]] == [0, 1]
        champion = client.get(f"/api/v1/runs/{result.id}/champion").json()
        assert champion["replay_available"]
        replay = client.get(f"/api/v1/runs/{result.id}/replay").json()
        assert len(replay["trajectory"]) == replay["final_state"]["steps"] + 1
        restored = FeedForwardNetwork.from_dict(champion["controller"])
        runner = ExperimentSettings.model_validate(details["settings"]).runner()
        episode = run_episode(
            runner.course,
            restored,
            agent_config=runner.agent_config,
            sensor_config=runner.sensor_config,
        )
        assert (
            score_episode(
                runner.course,
                episode,
                agent_config=runner.agent_config,
                config=runner.fitness_config,
            ).total
            == champion["fitness"]
        )
        assert client.post(f"/api/v1/runs/{result.id}/cancel").status_code == 409


def test_worker_failure_marks_run_failed_and_retains_finished_generations(
    queued, monkeypatch
):
    repo, run = queued
    original = EvolutionRunner.step
    calls = 0

    def fail_second(runner):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("private failure diagnostic")
        return original(runner)

    monkeypatch.setattr(EvolutionRunner, "step", fail_second)
    result = run_once(repo)
    assert result.status == "failed"
    assert result.completed_generations == 1
    assert repo.list_generations(run.id)[1] == 1
    assert repo.get_champion(run.id)["controller"]
    assert "RuntimeError" in result.error
    assert "private failure diagnostic" not in result.error


def test_cancellation_during_generation_preserves_it_and_stops_before_next(
    queued, monkeypatch
):
    repo, run = queued
    original = EvolutionRunner.step

    def cancel_after_step(runner):
        result = original(runner)
        repo.cancel(run.id)
        return result

    monkeypatch.setattr(EvolutionRunner, "step", cancel_after_step)
    result = run_once(repo)
    assert result.status == "cancelled"
    assert result.completed_generations == 1
    assert repo.get_replay(run.id)["trajectory"]


def test_heartbeat_renews_ownership_while_generation_is_executing(queued, monkeypatch):
    repo, _ = queued
    renewed = threading.Event()
    original_renew, original_step = repo.renew, EvolutionRunner.step

    def renew(*args, **kwargs):
        result = original_renew(*args, **kwargs)
        renewed.set()
        return result

    def wait_for_heartbeat(runner):
        if not renewed.wait(timeout=3):
            raise RuntimeError("Heartbeat did not run during the generation")
        return original_step(runner)

    monkeypatch.setattr(repo, "renew", renew)
    monkeypatch.setattr(EvolutionRunner, "step", wait_for_heartbeat)
    assert (
        run_once(repo, lease_seconds=30, heartbeat_seconds=0.01).status == "completed"
    )
    assert renewed.is_set()


def test_invalid_heartbeat_configuration_is_rejected(queued):
    with pytest.raises(ValueError):
        run_once(queued[0], lease_seconds=5, heartbeat_seconds=5)

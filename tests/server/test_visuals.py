"""Replay inspection uses actual saved controllers; fitness exports are real PNGs."""

from fastapi.testclient import TestClient

from neuroevolution.server.api import create_app
from neuroevolution.server.repository import Repository
from neuroevolution.server.schemas import (
    ExperimentCreate,
    ExperimentSettings,
    RunCreate,
)
from neuroevolution.server.worker import run_once


def test_saved_population_can_be_selected_replayed_and_inspected(database):
    repository = Repository(database)
    experiment = repository.create_experiment(
        ExperimentCreate(
            name="Replay check",
            settings={
                "generations": 2,
                "evolution": {
                    "population_size": 3,
                    "elite_count": 1,
                    "tournament_size": 2,
                },
                "agent": {"max_steps": 5},
            },
        )
    )
    run = repository.create_run(experiment.id, RunCreate())
    assert run_once(repository).status == "completed"
    with TestClient(create_app(database)) as client:
        path = f"/api/v1/runs/{run.id}"
        population = client.get(path + "/generations/0/population").json()
        assert len(population["agents"]) == 3
        assert "controller" not in population["agents"][0]
        replay = client.get(
            path + "/generations/0/replay", params={"agent": 0, "max_frames": 3}
        )
        assert replay.status_code == 200
        payload = replay.json()
        assert payload["agent"]["index"] == 0
        assert payload["fitness"]["total"] == population["agents"][0]["fitness"]
        assert len(payload["frames"]) == 3
        assert payload["frames"][0]["time"] == 0
        assert payload["frames"][-1]["state"] == payload["final_state"]
        assert len(payload["frames"][0]["inputs"]) == 10
        assert len(payload["frames"][0]["outputs"]) == 2
        assert payload["frames"][-1]["outputs"] is None
        assert payload["sampled"]
        assert (
            client.get(path + "/generations/0/replay", params={"agent": 5}).status_code
            == 404
        )
        assert client.get(path + "/generations/9/replay").status_code == 404
        chart = client.get(path + "/fitness.png")
        assert chart.status_code == 200
        assert chart.headers["content-type"] == "image/png"
        assert chart.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_unfinished_runs_have_no_chart_or_population_to_invent(database):
    repository = Repository(database)
    experiment = repository.create_experiment(ExperimentCreate(name="No results"))
    run = repository.create_run(experiment.id, RunCreate())
    with TestClient(create_app(database)) as client:
        assert client.get(f"/api/v1/runs/{run.id}/fitness.png").status_code == 409
        assert (
            client.get(f"/api/v1/runs/{run.id}/generations/0/population").status_code
            == 404
        )


def test_pre_dashboard_generations_retain_champion_inspection(database):
    repository = Repository(database)
    experiment = repository.create_experiment(
        ExperimentCreate(
            name="Legacy champion",
            settings={
                "generations": 1,
                "evolution": {
                    "population_size": 2,
                    "elite_count": 1,
                    "tournament_size": 2,
                },
                "agent": {"max_steps": 3},
            },
        )
    )
    queued = repository.create_run(experiment.id, RunCreate())
    claimed = repository.claim_next()
    runner = ExperimentSettings.model_validate(claimed.settings).runner()
    result = runner.step()
    # The previous worker stored the champion without a population snapshot.
    repository.save_generation(claimed.id, claimed.lease_token, result)
    with TestClient(create_app(database)) as client:
        path = f"/api/v1/runs/{queued.id}/generations/0"
        assert client.get(path + "/population").json() == {
            "agents": [],
            "champion_index": None,
        }
        replay = client.get(path + "/replay")
        assert replay.status_code == 200
        assert replay.json()["agent"]["index"] is None
        assert replay.json()["fitness"]["total"] == result.stats.best_fitness
        assert client.get(path + "/replay?agent=0").status_code == 404

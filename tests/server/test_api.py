"""HTTP validation, durable resource creation, pagination, and lifecycle errors."""

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from neuroevolution.server.api import create_app
from neuroevolution.server.database import Database


@pytest.fixture
def client(database):
    with TestClient(create_app(database)) as client:
        yield client


def tiny_experiment(client):
    response = client.post(
        "/api/v1/experiments",
        json={
            "name": "HTTP experiment",
            "settings": {
                "generations": 1,
                "evolution": {
                    "population_size": 2,
                    "elite_count": 1,
                    "tournament_size": 2,
                },
                "agent": {"max_steps": 1},
            },
        },
    )
    assert response.status_code == 201
    return response.json()


def test_create_list_and_inspect_experiments_and_runs(client):
    experiment = tiny_experiment(client)
    assert client.get("/ready").json() == {"status": "ready"}
    assert client.get("/health").status_code == 200
    assert client.get("/api/v1/defaults").json()["network"]["layer_sizes"] == [
        10,
        12,
        2,
    ]
    assert client.get("/api/v1/experiments").json()["total"] == 1
    assert client.get(f"/api/v1/experiments/{experiment['id']}").json() == experiment
    response = client.post(
        f"/api/v1/experiments/{experiment['id']}/runs", json={"seed": 7}
    )
    assert response.status_code == 202
    run = response.json()
    assert run["status"] == "queued" and run["settings"]["seed"] == 7
    assert run["progress"] == 0
    assert "lease_token" not in run
    assert client.get(f"/api/v1/runs/{run['id']}").json() == run
    assert (
        client.get("/api/v1/runs", params={"experiment_id": experiment["id"]}).json()[
            "total"
        ]
        == 1
    )
    assert client.get(f"/api/v1/runs/{run['id']}/generations").json()["items"] == []
    assert client.get(f"/api/v1/runs/{run['id']}/champion").status_code == 409
    assert client.get(f"/api/v1/runs/{run['id']}/replay").status_code == 409
    assert client.get("/openapi.json").status_code == 200


def test_queued_runs_can_be_cancelled_without_starting_a_worker(client):
    experiment = tiny_experiment(client)
    run = client.post(f"/api/v1/experiments/{experiment['id']}/runs").json()
    cancelled = client.post(f"/api/v1/runs/{run['id']}/cancel")
    assert cancelled.status_code == 202
    assert cancelled.json()["status"] == "cancelled"
    assert cancelled.json()["finished_at"] is not None


def test_experiments_survive_application_restart(database):
    with TestClient(create_app(database)) as first:
        experiment = tiny_experiment(first)
    with TestClient(create_app(database)) as second:
        assert (
            second.get(f"/api/v1/experiments/{experiment['id']}").json() == experiment
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"name": " "},
        {"name": "Bad", "unknown": 1},
        {"name": "Bad", "settings": {"network": {"layer_sizes": [9, 2]}}},
        {"name": "Bad", "settings": {"agent": {"radius": 100}}},
    ],
)
def test_invalid_experiments_are_rejected_before_persistence(client, payload):
    assert client.post("/api/v1/experiments", json=payload).status_code == 422
    assert client.get("/api/v1/experiments").json()["total"] == 0


def test_nan_validation_error_is_json_safe(client):
    response = client.post(
        "/api/v1/experiments",
        content='{"name":"Bad","settings":{"fitness":{"time_penalty":NaN}}}',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]


def test_missing_ids_bad_ids_and_invalid_pagination(client):
    assert client.get(f"/api/v1/runs/{uuid4()}").status_code == 404
    assert client.get("/api/v1/runs/not-a-uuid").status_code == 422
    assert client.get("/api/v1/experiments", params={"limit": 0}).status_code == 422
    assert client.get("/api/v1/experiments", params={"offset": -1}).status_code == 422
    assert client.post(f"/api/v1/experiments/{uuid4()}/runs").status_code == 404


def test_missing_migrations_report_unready_instead_of_silently_creating_tables(
    tmp_path,
):
    database = Database(f"sqlite:///{(tmp_path / 'empty.sqlite').as_posix()}")
    with TestClient(create_app(database)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
        assert client.get("/api/v1/experiments").status_code == 503

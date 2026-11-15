"""Run ownership, atomic generation persistence, and cancellation transitions."""

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from uuid import uuid4

import pytest

from neuroevolution.server.database import utcnow
from neuroevolution.server.repository import Conflict, LeaseLost, NotFound, Repository
from neuroevolution.server.schemas import ExperimentCreate, RunCreate


@pytest.fixture
def setup(database):
    repository = Repository(database)
    request = ExperimentCreate(
        name="Tiny experiment",
        settings={
            "generations": 2,
            "evolution": {"population_size": 2, "elite_count": 1, "tournament_size": 2},
            "agent": {"max_steps": 1},
        },
    )
    experiment = repository.create_experiment(request)
    run = repository.create_run(experiment.id, RunCreate())
    return repository, request, experiment, run


def test_run_overrides_are_snapshots_and_experiment_settings_stay_unchanged(setup):
    repo, request, experiment, first = setup
    second = repo.create_run(experiment.id, RunCreate(seed=7, generations=1))
    assert second.id != first.id
    assert second.settings["seed"] == 7
    assert second.requested_generations == 1
    assert repo.get_experiment(experiment.id).settings == request.settings.model_dump(
        mode="json"
    )
    assert repo.list_runs(experiment_id=experiment.id, limit=1)[1] == 2
    assert repo.list_experiments()[1] == 1


def test_claim_owner_and_generation_progress_persist_atomically(setup):
    repo, request, _, run = setup
    claimed = repo.claim_next()
    assert claimed.id == run.id
    assert claimed.status == "running"
    assert repo.claim_next() is None
    result = request.settings.runner().step()
    with pytest.raises(LeaseLost):
        repo.save_generation(run.id, uuid4(), result)
    assert repo.list_generations(run.id)[1] == 0
    repo.save_generation(run.id, claimed.lease_token, result)
    assert repo.get_run(run.id).completed_generations == 1
    rows, total = repo.list_generations(run.id)
    assert total == 1
    assert rows[0].best_fitness == result.stats.best_fitness
    assert rows[0].champion == result.champion.to_dict()
    with pytest.raises(Conflict):
        repo.save_generation(run.id, claimed.lease_token, result)
    assert repo.get_run(run.id).completed_generations == 1


def test_successful_completion_persists_champion_replay(setup):
    repo, request, _, run = setup
    claimed, runner = repo.claim_next(), request.settings.runner()
    for _ in range(2):
        repo.save_generation(run.id, claimed.lease_token, runner.step())
    finished = repo.finish(run.id, claimed.lease_token, {"status": "verified"})
    assert finished.status == "completed"
    assert finished.lease_token is None
    assert repo.get_replay(run.id) == {"status": "verified"}
    assert not repo.renew(run.id, claimed.lease_token)
    with pytest.raises(Conflict):
        repo.cancel(run.id)


def test_cancellation_is_immediate_when_queued_and_cooperative_when_running(setup):
    repo, request, experiment, run = setup
    assert repo.cancel(run.id).status == "cancelled"
    assert repo.cancel(run.id).status == "cancelled"
    second = repo.create_run(experiment.id, RunCreate())
    claimed = repo.claim_next()
    assert repo.cancel(second.id).cancel_requested
    assert repo.get_run(second.id).status == "running"
    repo.save_generation(
        second.id, claimed.lease_token, request.settings.runner().step()
    )
    assert repo.finish(second.id, claimed.lease_token, {}).status == "cancelled"


def test_expired_worker_is_failed_and_cannot_overwrite_retained_results(setup):
    repo, request, _, run = setup
    now = utcnow()
    claimed = repo.claim_next(now=now, lease_seconds=2)
    result = request.settings.runner().step()
    repo.save_generation(run.id, claimed.lease_token, result, now=now)
    assert repo.claim_next(now=now + timedelta(seconds=3)) is None
    assert repo.get_run(run.id).status == "failed"
    assert repo.get_run(run.id).completed_generations == 1
    assert repo.list_generations(run.id)[1] == 1
    assert not repo.renew(run.id, claimed.lease_token, now=now + timedelta(seconds=3))
    with pytest.raises(LeaseLost):
        repo.save_generation(run.id, claimed.lease_token, result)


def test_premature_completion_and_missing_ids_are_rejected(setup):
    repo, _, _, run = setup
    claimed = repo.claim_next()
    with pytest.raises(Conflict):
        repo.finish(run.id, claimed.lease_token, None)
    with pytest.raises(NotFound):
        repo.get_run(uuid4())
    with pytest.raises(Conflict):
        repo.get_replay(run.id)


def test_postgres_claims_one_job_only_once_under_concurrency(setup, database):
    if database.engine.dialect.name != "postgresql":
        pytest.skip("Row-lock concurrency is exercised by PostgreSQL integration CI.")
    repo, _, _, run = setup
    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(executor.map(lambda _: repo.claim_next(), range(2)))
    assert [claim.id for claim in claims if claim is not None] == [run.id]

"""Standalone durable-queue worker: python -m neuroevolution.server.worker."""

import argparse
import logging
import platform
import threading
from dataclasses import asdict
from importlib.metadata import version

import numpy as np

from neuroevolution import run_episode, score_episode
from neuroevolution._validation import positive_float
from neuroevolution.server.database import Database
from neuroevolution.server.repository import LeaseLost, Repository
from neuroevolution.server.schemas import ExperimentSettings

logger = logging.getLogger(__name__)


class _Heartbeat:
    def __init__(self, repository, run, duration, interval):
        self.repository, self.run = repository, run
        self.duration, self.interval = duration, interval
        self.stop = threading.Event()
        self.lost = threading.Event()
        self.thread = threading.Thread(target=self._renew, daemon=True)

    def _renew(self):
        while not self.stop.wait(self.interval):
            try:
                valid = self.repository.renew(
                    self.run.id, self.run.lease_token, lease_seconds=self.duration
                )
            except Exception:
                logger.exception("Worker heartbeat failed")
                valid = False
            if not valid:
                self.lost.set()
                return

    def check(self):
        if self.lost.is_set():
            raise LeaseLost("Worker heartbeat lost its lease.")

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)


def run_once(repository: Repository, *, lease_seconds=30, heartbeat_seconds=5):
    """Process at most one job. Return its latest persisted state, or None."""
    duration = positive_float(lease_seconds, "Lease duration")
    interval = positive_float(heartbeat_seconds, "Heartbeat interval")
    if interval >= duration / 2:
        raise ValueError(
            "Heartbeat interval must be less than half the lease duration."
        )
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "simulator": version("neuroevolution-simulator"),
    }
    run = repository.claim_next(lease_seconds=duration, environment=environment)
    if run is None:
        return None
    logger.info("Running experiment job %s", run.id)
    try:
        with _Heartbeat(repository, run, duration, interval) as heartbeat:
            settings = ExperimentSettings.model_validate(run.settings)
            runner = settings.runner()
            for _ in range(settings.generations):
                heartbeat.check()
                if repository.get_run(run.id).cancel_requested:
                    break
                result = runner.step()
                heartbeat.check()
                repository.save_generation(
                    run.id, run.lease_token, result, population=runner.population
                )
            replay = None
            if runner.history:
                episode = run_episode(
                    runner.course,
                    runner.champion,
                    agent_config=runner.agent_config,
                    sensor_config=runner.sensor_config,
                    record=True,
                )
                score = score_episode(
                    runner.course,
                    episode,
                    agent_config=runner.agent_config,
                    config=runner.fitness_config,
                )
                if score.total != max(stats.best_fitness for stats in runner.history):
                    raise RuntimeError(
                        "Champion replay did not reproduce the saved fitness."
                    )
                replay = {
                    "final_state": asdict(episode.final_state),
                    "elapsed_seconds": episode.elapsed_seconds,
                    "fitness": {**asdict(score), "total": score.total},
                    "trajectory": [asdict(state) for state in episode.trajectory],
                }
            heartbeat.check()
            repository.finish(run.id, run.lease_token, replay)
    except LeaseLost:
        logger.warning("Job %s stopped after losing ownership", run.id)
    except Exception as error:
        logger.exception("Job %s failed", run.id)
        try:
            repository.fail(
                run.id,
                run.lease_token,
                f"Worker failed ({type(error).__name__}); inspect worker logs.",
            )
        except LeaseLost:
            logger.warning("Job %s no longer belongs to this worker", run.id)
    return repository.get_run(run.id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--once", action="store_true", help="Process one queued job and exit."
    )
    parser.add_argument("--poll-interval", type=float, default=1)
    parser.add_argument("--lease-seconds", type=float, default=30)
    parser.add_argument("--heartbeat-seconds", type=float, default=5)
    args = parser.parse_args()
    try:
        interval = positive_float(args.poll_interval, "Poll interval")
        duration = positive_float(args.lease_seconds, "Lease duration")
        heartbeat = positive_float(args.heartbeat_seconds, "Heartbeat interval")
        if heartbeat >= duration / 2:
            raise ValueError(
                "Heartbeat interval must be less than half the lease duration."
            )
    except ValueError as error:
        parser.error(str(error))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    database = Database()
    try:
        repository = Repository(database)
        while True:
            result = run_once(
                repository, lease_seconds=duration, heartbeat_seconds=heartbeat
            )
            if args.once:
                if result is not None and result.status == "failed":
                    raise SystemExit(1)
                break
            if result is None:
                threading.Event().wait(interval)
    except KeyboardInterrupt:
        logger.info(
            "Worker stopped; unfinished leases will be recovered on a later poll"
        )
    finally:
        database.close()


if __name__ == "__main__":
    main()

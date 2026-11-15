"""Exercise the real container stack and verify saved results after restart."""

import argparse
import json
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--run-file", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    def request(path, body=None):
        payload = None if body is None else json.dumps(body).encode()
        with urlopen(
            Request(
                args.url + path,
                data=payload,
                headers={"Content-Type": "application/json"},
            ),
            timeout=10,
        ) as response:
            return response.read()

    def data(path, body=None):
        return json.loads(request(path, body))

    deadline = time.monotonic() + 90
    while True:
        try:
            assert data("/ready")["status"] == "ready"
            break
        except (URLError, TimeoutError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(1)
    assert b"Evolution lab" in request("/")
    for asset in ("app.js", "data.js", "arena.js", "charts.js", "styles.css"):
        assert len(request("/assets/" + asset)) > 100
    if args.verify:
        identifier = args.run_file.read_text().strip()
    else:
        experiment = data(
            "/api/v1/experiments",
            {
                "name": "Compose lifecycle smoke",
                "settings": {
                    "generations": 2,
                    "evolution": {
                        "population_size": 4,
                        "elite_count": 1,
                        "tournament_size": 2,
                    },
                    "agent": {"max_steps": 10},
                },
            },
        )
        identifier = data(f"/api/v1/experiments/{experiment['id']}/runs", {})["id"]
        args.run_file.write_text(identifier)
    path = f"/api/v1/runs/{identifier}"
    deadline = time.monotonic() + 90
    while True:
        run = data(path)
        if run["status"] == "completed":
            break
        if run["status"] in ("failed", "cancelled"):
            raise RuntimeError(f"Container run failed: {run}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Worker did not finish: {run['status']}")
        time.sleep(1)
    assert run["completed_generations"] == 2
    assert data(path + "/generations")["total"] == 2
    assert len(data(path + "/generations/0/population")["agents"]) == 4
    replay = data(path + "/generations/0/replay?agent=0")
    assert len(replay["frames"][0]["inputs"]) == 10
    assert replay["frames"][-1]["outputs"] is None
    assert data(path + "/champion")["replay_available"]
    assert request(path + "/fitness.png").startswith(b"\x89PNG\r\n\x1a\n")
    print(
        f"{'Persistence verified' if args.verify else 'Stack verified'}: {identifier}"
    )


if __name__ == "__main__":
    main()

"""Start the local dashboard, API, and worker from a source checkout."""

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from alembic import command
from alembic.config import Config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sqlite",
        action="store_true",
        help="Use artifacts/dashboard.sqlite for development.",
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    root = Path(__file__).resolve().parents[3]
    if not (root / "alembic.ini").exists():
        parser.error("Run this development launcher from an installed source checkout.")
    if args.sqlite:
        (root / "artifacts").mkdir(exist_ok=True)
        os.environ["DATABASE_URL"] = (
            f"sqlite:///{(root / 'artifacts' / 'dashboard.sqlite').as_posix()}"
        )
    # Fail before starting a worker when another server already owns the port.
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError:
            parser.error(
                f"Port {args.port} is already in use; choose --port or stop it."
            )
    command.upgrade(Config(str(root / "alembic.ini")), "head")
    url = f"http://127.0.0.1:{args.port}"
    processes = []
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        api = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "neuroevolution.server.api:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.port),
            ],
            cwd=root,
            creationflags=flags,
        )
        processes.append(api)
        deadline = time.monotonic() + 20
        while True:
            if api.poll() is not None:
                raise RuntimeError("The API did not start; inspect the server output.")
            try:
                with urlopen(f"{url}/ready", timeout=1) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError):
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "The API did not become ready in 20 seconds."
                    ) from None
                time.sleep(0.2)
        worker = subprocess.Popen(
            [sys.executable, "-m", "neuroevolution.server.worker"],
            cwd=root,
            creationflags=flags,
        )
        processes.append(worker)
        print(
            f"\nLocal: {url}\nDashboard and worker ready. Press Ctrl+C to stop.\n",
            flush=True,
        )
        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
        raise RuntimeError("A service stopped; inspect the server output.")
    except KeyboardInterrupt:
        print(
            "\nStopping the dashboard and worker; saved results are retained.",
            flush=True,
        )
    finally:
        for process in reversed(processes):
            if process.poll() is None:
                if os.name == "nt":
                    # Windows venv executables may launch another Python process.
                    # Terminating only the launcher can leave its service running.
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=flags,
                        check=False,
                    )
                else:
                    process.terminate()
        for process in reversed(processes):
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()

"""Run directories: one folder per pipeline run with its config and environment."""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml

from src.config import Config

_PACKAGES = {
    "opencv": "opencv-contrib-python",
    "numpy": "numpy",
    "scipy": "scipy",
    "open3d": "open3d",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
}


def create_run_dir(root: str | Path, label: str) -> Path:
    """Create ``<root>/<YYYYmmdd-HHMMSS>_<label>`` (suffixed if it already exists)."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(root) / f"{stamp}_{label}"
    path, n = base, 1
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}_{n}")
    path.mkdir(parents=True)
    return path


def worker_count(workers: int) -> int:
    """``performance.workers``: 0 means all CPU cores."""
    return max(1, os.cpu_count() or 1) if workers <= 0 else workers


def parallel_map[T, U](fn: Callable[[T], U], items: Iterable[T], workers: int) -> list[U]:
    """``[fn(x) for x in items]`` on a thread pool, results in input order.

    Threads (not processes) suffice because OpenCV releases the GIL in its heavy calls.
    ``fn`` must not depend on shared mutable state; seeded OpenCV RNG calls must seed
    inside ``fn`` (OpenCV's RNG is per thread).
    """
    items = list(items)
    n = worker_count(workers)
    if n == 1 or len(items) < 2:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=min(n, len(items))) as pool:
        return list(pool.map(fn, items))


def atomic_savez(path: str | Path, compressed: bool = False, **arrays) -> Path:
    """``np.savez`` into ``path`` atomically: readers never see a partial file.

    Writes to a uniquely named temporary file in the same directory and renames it into
    place, so concurrent writers of the same path (threads, processes) cannot clash; the
    last complete write wins.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            (np.savez_compressed if compressed else np.savez)(f, **arrays)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def write_json(path: str | Path, data) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: str | Path, header: Sequence[str], rows: Iterable[Sequence]) -> Path:
    """Write a CSV file (creating parent folders) and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def write_run_metadata(run_dir: Path, config: Config) -> None:
    with (run_dir / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, sort_keys=False)
    write_json(run_dir / "environment.json", environment_info())


def environment_info() -> dict:
    packages = {}
    for key, dist in _PACKAGES.items():
        try:
            packages[key] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            packages[key] = None
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_commit": _git_commit(),
        "packages": packages,
    }


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None

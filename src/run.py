"""Run directories: one folder per pipeline run with its config and environment."""

from __future__ import annotations

import csv
import json
import platform
import subprocess
from collections.abc import Iterable, Sequence
from datetime import datetime
from importlib import metadata
from pathlib import Path

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
    (run_dir / "environment.json").write_text(
        json.dumps(environment_info(), indent=2), encoding="utf-8"
    )


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

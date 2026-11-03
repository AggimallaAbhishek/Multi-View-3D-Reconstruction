"""Command-line scripts: run from anywhere, with standard argument handling."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def run_script(name, *args):
    """The script as a user runs it, from a folder other than the repository (tests run in
    a temporary working directory)."""
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *args],
        capture_output=True, text=True, timeout=120,
    )  # fmt: skip


@pytest.mark.parametrize("name", ["track_coverage.py", "download_eth3d.py"])
def test_help_works_from_outside_the_repository(name):
    done = run_script(name, "--help")
    assert done.returncode == 0, done.stderr
    assert "usage:" in done.stdout


def test_eth3d_unknown_scene_is_an_argument_error():
    done = run_script("download_eth3d.py", "no-such-scene")
    assert done.returncode == 2
    assert "invalid choice" in done.stderr and "pipes" in done.stderr


def test_track_coverage_default_config_is_the_repositorys():
    import scripts.track_coverage as script  # the repository is on pythonpath (pyproject)

    assert script.parse_args(["images"]).config is None  # load_config(None): the default

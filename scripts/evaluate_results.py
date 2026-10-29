"""Redraw the reports of finished experiments without rerunning the pipeline.

Usage:
    python scripts/evaluate_results.py outputs/experiments/<run>  [or one experiment dir]

Regenerates metrics.png, runtime.png and results.md from every experiment_results.csv
under the given directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluation.reports import make_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    found = sorted(p.parent for p in args.directory.rglob("experiment.json"))
    if not found:
        print(f"no experiment.json under {args.directory}")
        return 1
    for directory in found:
        for path in make_report(directory):
            print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

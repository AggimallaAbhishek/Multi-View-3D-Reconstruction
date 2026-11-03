"""Turn a finished web run into the intro page's showcase (src/web/static/showcase/).

Usage:
    python scripts/make_showcase.py outputs/<stamp>_web_reconstruct data/raw/temple \
        --up-axis y [--name "Middlebury Temple"] [--out src/web/static/showcase]

Uses the run's own settings (its config.yaml) to recompute the features and matches
drawn in the "how it works" frames; see src/web/showcase.py. The output is committed so
the intro page works on a fresh clone.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.web.showcase import make_showcase  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir", type=Path, help="a finished web run (outputs/*_web_reconstruct)")
    parser.add_argument("images", type=Path, help="the photos the run was made from")
    parser.add_argument(
        "--name", default="", help="name shown on the page (default: the photo folder)"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "src" / "web" / "static" / "showcase")
    parser.add_argument(
        "--up-axis",
        choices=["x", "y", "z"],
        help="the ground-truth frame's up axis (gt_cameras.json next to the photos); "
        "stands the model upright on the page (Middlebury: y)",
    )
    args = parser.parse_args(argv)
    run_config = args.run_dir / "config.yaml"
    config = load_config(run_config if run_config.is_file() else None)
    up_axis = None if args.up_axis is None else "xyz".index(args.up_axis)
    try:
        out = make_showcase(
            args.run_dir,
            args.images,
            args.out,
            config,
            name=args.name or args.images.name,
            up_axis=up_axis,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    total = sum(p.stat().st_size for p in out.iterdir())
    print(f"Wrote {out} ({total / 2**20:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

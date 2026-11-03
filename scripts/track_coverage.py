"""Tracks without a landmark after reconstruction, and why: for every track seen by at
least ``--min-views`` registered images but left without a landmark, triangulate it
with the final cameras and count the filter that rejects it and how many of its
observations reproject beyond the threshold (one bad keypoint, or a wrongly merged track).

    python scripts/track_coverage.py data/raw/templeRing
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # runnable from any folder, like the other scripts

from src.config import load_config  # noqa: E402
from src.geometry.triangulation import Observations, triangulate_and_filter  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402
from src.reconstruction.landmarks import keypoint_pixels  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", type=Path)
    parser.add_argument("--config", type=Path, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--min-views", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    config = load_config(args.config)
    result = run_pipeline(args.images, config, until="reconstruct")
    rec, tracks = result.reconstruction, result.tracks
    if rec is None:
        print("no reconstruction")
        return 1
    images = rec.registry.images
    in_registered = np.isin(tracks.image, images)
    has_landmark = np.zeros(tracks.n_tracks, bool)
    has_landmark[rec.landmarks.track] = True
    views = np.bincount(tracks.track[in_registered], minlength=tracks.n_tracks)
    long = views >= args.min_views
    missing = np.flatnonzero(long & ~has_landmark)
    print(
        f"tracks seen by >= {args.min_views} registered images: {long.sum()}, "
        f"without a landmark: {len(missing)}"
    )
    if not len(missing):
        return 0

    point_of_track = np.full(tracks.n_tracks, -1)
    point_of_track[missing] = np.arange(len(missing))
    sel = in_registered & (point_of_track[tracks.track] >= 0)
    view_of_image = np.full(tracks.image.max() + 1, -1)
    view_of_image[images] = np.arange(len(images))
    obs = Observations(
        view_of_image[tracks.image[sel]],
        point_of_track[tracks.track[sel]],
        keypoint_pixels(result.features, tracks.image[sel], tracks.keypoint[sel]),
    )
    cameras = [rec.registry.camera(m) for m in images]
    tri = triangulate_and_filter(cameras, obs, config.triangulation, len(missing))
    bad = tri.errors > config.triangulation.max_reprojection_error_px
    n_bad = np.bincount(obs.point, weights=bad, minlength=len(missing)).astype(int)
    print(f"rejected by: {dict(Counter(tri.rejected_by))}")
    per_track = dict(sorted(Counter(n_bad.tolist()).items()))
    print(f"observations beyond the threshold per track: {per_track}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Sparse vs dense on a dark-background dataset (Middlebury temple/dino): point counts,
image coverage, run time, and silhouette consistency — every true surface point projects
inside the object's silhouette in every view, so the fraction of points falling on the
background in some view measures outliers without a ground-truth surface.

    python scripts/evaluate_dense.py data/raw/templeRing [--threshold 15]

Silhouettes: grey level above ``--threshold``, closed, then dilated by ``--margin`` px
(silhouette edges and the camera poses are not exact). Shaded parts of the object fall
below the threshold too, so even good points score a few per cent: compare the sparse
and dense rows rather than reading either as an absolute outlier rate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.pipeline import run_pipeline  # noqa: E402


def silhouettes(images, threshold: int, margin: int) -> list[np.ndarray]:
    masks = []
    for image in images:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = cv2.morphologyEx((grey > threshold).astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((7, 7), np.uint8))  # fmt: skip
        masks.append(cv2.dilate(mask, np.ones((2 * margin + 1,) * 2, np.uint8)).astype(bool))
    return masks


def outside_fraction(points: np.ndarray, cameras, masks) -> float:
    """Fraction of points that project onto the background of at least one view."""
    bad = np.zeros(len(points), bool)
    for cam, mask in zip(cameras, masks, strict=True):
        x, z = cam.project(points)
        h, w = mask.shape
        with np.errstate(invalid="ignore"):
            u, v = np.round(x[:, 0]), np.round(x[:, 1])
            inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        bad[inside] |= ~mask[v[inside].astype(int), u[inside].astype(int)]
    return float(bad.mean()) if len(points) else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--threshold", type=int, default=15, help="background grey level")
    parser.add_argument("--margin", type=int, default=2, help="silhouette dilation (px)")
    args = parser.parse_args(argv)

    result = run_pipeline(args.images, load_config(args.config), until="dense")
    rec, dense = result.reconstruction, result.dense
    if dense is None:
        print("no reconstruction")
        return 1
    cameras = [rec.registry.camera(m) for m in rec.registry.images]
    masks = silhouettes([result.images[m].image for m in rec.registry.images],
                        args.threshold, args.margin)  # fmt: skip
    sparse = rec.landmarks.position[result.filtering.keep]
    coverage = [c for c in dense.coverage if c is not None]
    sparse_s = sum(result.timings[s] for s in result.timings if s not in ("dense",))
    print(f"{'':10s}{'points':>10s}{'outside a silhouette':>22s}{'time (s)':>10s}")
    print(f"{'sparse':10s}{len(sparse):10d}{outside_fraction(sparse, cameras, masks):22.3f}"
          f"{sparse_s:10.1f}")  # fmt: skip
    print(f"{'dense':10s}{len(dense.cloud.points):10d}"
          f"{outside_fraction(dense.cloud.points, cameras, masks):22.3f}"
          f"{result.timings['dense']:10.1f}")  # fmt: skip
    print(f"depth maps: {len(coverage)} of {len(cameras)} images; pixels with a depth: "
          f"median {np.median(coverage):.2f}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Cleaning a sparse point cloud before export.

Per-point filters (non-finite, reprojection error, track length) come first, then two
neighbourhood filters through Open3D's tensor API: statistical outlier removal (mean
distance to the k nearest neighbours far above the cloud's average) and radius outlier
removal (too few neighbours within a radius). An SfM reconstruction is known only up to
scale, so the radius is relative: ``radius_scale`` times the median distance of a point
to its ``radius_nb_points``-th nearest neighbour.

Each step sees only the survivors of the previous one; ``FilterStep`` records how many
points it saw and removed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from src.config import FilteringConfig

FILTER_STEPS = ("non_finite", "reprojection", "track_length", "statistical", "radius")


@dataclass(frozen=True)
class FilterStep:
    name: str
    before: int
    removed: int

    @property
    def after(self) -> int:
        return self.before - self.removed


@dataclass(frozen=True, eq=False)
class CloudFilterResult:
    keep: np.ndarray  # (N,) bool: survived every filter
    steps: list[FilterStep]  # in FILTER_STEPS order

    def summary(self) -> dict:
        return {
            "points": len(self.keep),
            "kept": int(self.keep.sum()),
            "steps": [asdict(s) for s in self.steps],
        }


def filter_cloud(
    points: np.ndarray,
    config: FilteringConfig,
    errors: np.ndarray | None = None,
    track_lengths: np.ndarray | None = None,
) -> CloudFilterResult:
    """Which of ``points`` (N, 3) to keep. ``errors`` (mean reprojection error per point,
    px) and ``track_lengths`` (observations per point) enable their filters; without
    them those steps remove nothing. A neighbourhood filter is skipped (removes nothing)
    if it is disabled (0 neighbours) or the cloud has too few points for it."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(points)
    for name, values in (("errors", errors), ("track_lengths", track_lengths)):
        if values is not None and len(values) != n:
            raise ValueError(f"{n} points but {len(values)} {name}")

    keep = np.ones(n, bool)
    steps: list[FilterStep] = []

    def apply(name: str, bad: np.ndarray) -> None:
        """Drop the surviving points where ``bad`` (N,) is true."""
        removed = keep & bad
        steps.append(FilterStep(name, int(keep.sum()), int(removed.sum())))
        keep[removed] = False

    apply("non_finite", ~np.isfinite(points).all(axis=1))
    apply("reprojection", np.zeros(n, bool) if errors is None
          else ~(np.asarray(errors) <= config.max_reprojection_error_px))  # fmt: skip
    apply("track_length", np.zeros(n, bool) if track_lengths is None
          else np.asarray(track_lengths) < config.min_track_length)  # fmt: skip
    apply("statistical", _neighbourhood_outliers(points, keep, config, "statistical"))
    apply("radius", _neighbourhood_outliers(points, keep, config, "radius"))
    return CloudFilterResult(keep, steps)


def _neighbourhood_outliers(
    points: np.ndarray, keep: np.ndarray, config: FilteringConfig, kind: str
) -> np.ndarray:
    """(N,) true for surviving points that Open3D's statistical or radius filter rejects."""
    import open3d as o3d

    k = config.statistical_nb_neighbors if kind == "statistical" else config.radius_nb_points
    idx = np.flatnonzero(keep)
    bad = np.zeros(len(points), bool)
    if k <= 0 or len(idx) <= k:
        return bad
    # Centred and scaled to unit RMS spread: float32 keeps its precision and the
    # filters behave the same at any reconstruction scale.
    X = points[idx] - points[idx].mean(axis=0)
    X /= np.sqrt((X**2).sum(axis=1).mean()) or 1.0
    positions = o3d.core.Tensor(X.astype(np.float32))
    cloud = o3d.t.geometry.PointCloud(positions)
    if kind == "statistical":
        _, mask = cloud.remove_statistical_outliers(k, config.statistical_std_ratio)
    else:
        nns = o3d.core.nns.NearestNeighborSearch(positions)
        nns.knn_index()
        _, d2 = nns.knn_search(positions, k + 1)  # the first neighbour is the point itself
        radius = config.radius_scale * float(np.median(np.sqrt(d2.numpy()[:, -1])))
        _, mask = cloud.remove_radius_outliers(k, radius)
    bad[idx[~mask.numpy()]] = True
    return bad

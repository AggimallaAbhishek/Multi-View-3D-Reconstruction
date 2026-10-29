"""Triangulation of 3-D points from image observations, with quality filters.

Observations are an *observation list*: one row per (camera, 3-D point, pixel). A point
can be seen by any number of cameras, so the same data structure serves two-view
initialisation, multi-view tracks and bundle adjustment.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.config import TriangulationConfig
from src.geometry.camera import Camera
from src.geometry.reprojection import (
    ReprojectionStats,
    project_observations,
    reprojection_errors,
)


@dataclass(frozen=True, eq=False)
class Observations:
    view: np.ndarray  # (M,) camera index of each observation
    point: np.ndarray  # (M,) 3-D point index of each observation
    xy: np.ndarray  # (M, 2) observed pixel

    def __post_init__(self) -> None:
        object.__setattr__(self, "view", np.asarray(self.view, dtype=np.int64))
        object.__setattr__(self, "point", np.asarray(self.point, dtype=np.int64))
        object.__setattr__(self, "xy", np.asarray(self.xy, dtype=np.float64).reshape(-1, 2))
        if not len(self.view) == len(self.point) == len(self.xy):
            raise ValueError("view, point and xy must have the same length")

    def __len__(self) -> int:
        return len(self.view)

    @property
    def n_points(self) -> int:
        return int(self.point.max()) + 1 if len(self.point) else 0

    def views_per_point(self, n_points: int | None = None) -> np.ndarray:
        return np.bincount(self.point, minlength=n_points or self.n_points)

    def subset(self, mask: np.ndarray) -> Observations:
        return Observations(self.view[mask], self.point[mask], self.xy[mask])


# --- linear triangulation --------------------------------------------------------------


def triangulate_dlt(
    cameras: Sequence[Camera], obs: Observations, n_points: int | None = None
) -> np.ndarray:
    """Linear (DLT) triangulation of every point; (N, 3), nan for points seen < 2 times.

    Each observation x ~ K [R | t] X gives two equations in X. They are written in
    normalised coordinates K^-1 x with P = [R | t] and scaled to unit length, which keeps
    the system well conditioned. For each point, X is the smallest-eigenvalue
    eigenvector of A^T A, accumulated over all its observations, so points with any
    number of observations are solved together in one batched eigendecomposition.
    """
    n_points = obs.n_points if n_points is None else n_points
    K_inv = np.stack([np.linalg.inv(c.K) for c in cameras])[obs.view]
    Rt = np.stack([np.hstack([c.R, c.t.reshape(3, 1)]) for c in cameras])[obs.view]
    x = np.einsum("mij,mj->mi", K_inv, np.hstack([obs.xy, np.ones((len(obs), 1))]))
    x = x[:, :2] / x[:, 2:3]

    rows = np.concatenate(
        [x[:, :1] * Rt[:, 2] - Rt[:, 0], x[:, 1:2] * Rt[:, 2] - Rt[:, 1]]
    )  # (2M, 4)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    normal = np.zeros((n_points, 4, 4))
    np.add.at(normal, np.concatenate([obs.point, obs.point]), rows[:, :, None] * rows[:, None])

    X = np.full((n_points, 3), np.nan)
    solvable = obs.views_per_point(n_points) >= 2
    _, vectors = np.linalg.eigh(normal[solvable])
    X_h = vectors[:, :, 0]  # eigenvalues ascending: the first is the null vector
    X[solvable] = _dehomogenise(X_h)
    return X


def triangulate_two_view(
    P_i: np.ndarray, P_j: np.ndarray, x_i: np.ndarray, x_j: np.ndarray
) -> np.ndarray:
    """Linear (DLT) triangulation of N correspondences from two views; returns (N, 3).

    ``x_i``/``x_j`` are in whatever coordinates the projection matrices use (pixels
    with P = K [R | t], normalised coordinates with P = [R | t]). Used for the
    cheirality test of relative-pose candidates, where only P matrices are at hand.
    """
    A = np.stack(
        [
            x_i[:, 0, None] * P_i[2] - P_i[0],
            x_i[:, 1, None] * P_i[2] - P_i[1],
            x_j[:, 0, None] * P_j[2] - P_j[0],
            x_j[:, 1, None] * P_j[2] - P_j[1],
        ],
        axis=1,
    )  # (N, 4, 4)
    return _dehomogenise(np.linalg.svd(A)[2][:, -1])


def _dehomogenise(X_h: np.ndarray) -> np.ndarray:
    """(N, 4) -> (N, 3). Points at infinity (parallel rays) come back very far away
    rather than as nan/inf: w keeps its sign but is bounded away from zero."""
    w = X_h[:, 3:]
    w = np.where(np.abs(w) < 1e-12, np.copysign(1e-12, w), w)
    return X_h[:, :3] / w


# --- nonlinear refinement --------------------------------------------------------------


def refine_points(
    cameras: Sequence[Camera], obs: Observations, X: np.ndarray, iterations: int = 5
) -> np.ndarray:
    """Gauss-Newton refinement of each point's reprojection error in pixels.

    DLT minimises an algebraic error; this minimises the geometric one with cameras
    fixed (each point is an independent 3-parameter problem, solved in a batch). An
    update is kept only for points whose total squared error decreases.
    """
    X = X.copy()
    n = len(X)
    ok = np.isfinite(X).all(axis=1)
    K = np.stack([c.K for c in cameras])[obs.view]
    KR = np.einsum("mij,mjk->mik", K, np.stack([c.R for c in cameras])[obs.view])

    Kt = np.stack([c.K @ c.t for c in cameras])[obs.view]

    def cost(points: np.ndarray) -> np.ndarray:
        e = reprojection_errors(cameras, obs, points) ** 2
        return np.bincount(obs.point, weights=np.where(np.isfinite(e), e, 1e30), minlength=n)

    current = cost(X)
    for _ in range(iterations):
        pixels, _ = project_observations(cameras, obs, X)
        residual = pixels - obs.xy  # (M, 2)
        p = np.einsum("mij,mj->mi", KR, X[obs.point]) + Kt  # homogeneous pixel K(RX + t)
        # d(u, v)/dX for u = p0/p2, v = p1/p2, with dp/dX = K R.
        J = (KR[:, :2] * p[:, 2, None, None] - p[:, :2, None] * KR[:, 2][:, None]) / (
            p[:, 2, None, None] ** 2
        )  # (M, 2, 3)
        JtJ = np.zeros((n, 3, 3))
        Jtr = np.zeros((n, 3))
        good = np.isfinite(residual).all(axis=1) & np.isfinite(J).all(axis=(1, 2))
        np.add.at(JtJ, obs.point[good], np.einsum("mki,mkj->mij", J[good], J[good]))
        np.add.at(Jtr, obs.point[good], np.einsum("mki,mk->mi", J[good], residual[good]))
        damping = 1e-9 * np.trace(JtJ, axis1=1, axis2=2)[:, None, None] * np.eye(
            3
        ) + 1e-12 * np.eye(3)
        step = np.zeros_like(X)
        step[ok] = -np.linalg.solve(JtJ[ok] + damping[ok], Jtr[ok][:, :, None])[:, :, 0]
        candidate = X + step
        new = cost(candidate)
        better = ok & (new < current)
        X[better], current[better] = candidate[better], new[better]
    return X


# --- quality ---------------------------------------------------------------------------


def triangulation_angles_deg(
    cameras: Sequence[Camera], obs: Observations, X: np.ndarray
) -> np.ndarray:
    """Largest angle between any two viewing rays of each point (nan if seen < 2 times).

    Small angles mean the rays are nearly parallel and depth is poorly determined.
    Points are grouped by track length so each group is one vectorised computation.
    """
    n = len(X)
    centres = np.stack([c.centre for c in cameras])
    rays = X[obs.point] - centres[obs.view]
    with np.errstate(invalid="ignore", divide="ignore"):
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)

    order = np.argsort(obs.point, kind="stable")
    counts = obs.views_per_point(n)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    angles = np.full(n, np.nan)
    for length in np.unique(counts[counts >= 2]):
        pts = np.flatnonzero(counts == length)
        idx = order[starts[pts][:, None] + np.arange(length)]  # (n_L, L)
        r = rays[idx]  # (n_L, L, 3)
        cos = np.einsum("pai,pbi->pab", r, r).reshape(len(pts), -1).min(axis=1)
        angles[pts] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return angles


FILTER_ORDER = ("too_few_views", "non_finite", "behind_camera", "reprojection", "low_angle")


@dataclass(frozen=True, eq=False)
class TriangulationResult:
    points: np.ndarray  # (N, 3); invalid points are kept so indices stay aligned
    valid: np.ndarray  # (N,) passed every filter
    rejected_by: np.ndarray  # (N,) object: first failed filter name, or "" if valid
    angles_deg: np.ndarray  # (N,) largest ray angle
    max_errors: np.ndarray  # (N,) largest reprojection error over the point's observations
    errors: np.ndarray  # (M,) reprojection error per observation
    reprojection: ReprojectionStats  # over observations of valid points

    def summary(self) -> dict:
        """Counts per filter, reprojection statistics and triangulation angles (valid points)."""
        angles = self.angles_deg[self.valid]
        return {
            "counts": self.counts,
            "reprojection_valid_points": self.reprojection.to_dict(),
            "triangulation_angle_deg": {
                "median": float(np.median(angles)) if angles.size else None,
                "min": float(angles.min()) if angles.size else None,
            },
        }

    @property
    def counts(self) -> dict[str, int]:
        counts = {"points": len(self.points)}
        counts.update({name: int(np.sum(self.rejected_by == name)) for name in FILTER_ORDER})
        counts["valid"] = int(self.valid.sum())
        return counts


def triangulate_and_filter(
    cameras: Sequence[Camera],
    obs: Observations,
    config: TriangulationConfig,
    n_points: int | None = None,
) -> TriangulationResult:
    """Triangulate every point (DLT, optionally refined) and apply the quality filters.

    A point is valid if it has at least two observations, finite coordinates, positive
    depth in every observing camera, every reprojection error at most
    ``max_reprojection_error_px`` and a triangulation angle of at least
    ``min_angle_deg``. Each rejected point records the first filter it failed.
    """
    n = obs.n_points if n_points is None else n_points
    X = triangulate_dlt(cameras, obs, n)
    if config.refine:
        X = refine_points(cameras, obs, X, config.refine_iterations)

    _, depths = project_observations(cameras, obs, X)
    errors = reprojection_errors(cameras, obs, X)
    behind = np.bincount(obs.point, weights=~(depths > 0), minlength=n) > 0
    max_errors = np.full(n, -np.inf)
    np.maximum.at(max_errors, obs.point, np.where(np.isfinite(errors), errors, np.inf))
    angles = triangulation_angles_deg(cameras, obs, X)

    failed = {
        "too_few_views": obs.views_per_point(n) < 2,
        "non_finite": ~np.isfinite(X).all(axis=1),
        "behind_camera": behind,
        "reprojection": max_errors > config.max_reprojection_error_px,
        "low_angle": ~(angles >= config.min_angle_deg),
    }
    rejected_by = np.full(n, "", dtype=object)
    for name in reversed(FILTER_ORDER):  # earlier filters overwrite later ones
        rejected_by[failed[name]] = name
    valid = rejected_by == ""
    return TriangulationResult(
        points=X,
        valid=valid,
        rejected_by=rejected_by,
        angles_deg=angles,
        max_errors=max_errors,
        errors=errors,
        reprojection=ReprojectionStats.from_errors(errors[valid[obs.point]]),
    )

"""Reprojection of 3-D points into the cameras that observe them."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.geometry.camera import Camera

if TYPE_CHECKING:
    from src.geometry.triangulation import Observations


def project_observations(
    cameras: Sequence[Camera], obs: Observations, X: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Projected pixels (M, 2) and depths (M,) of each observation's 3-D point."""
    P = np.stack([c.P for c in cameras])[obs.view]  # (M, 3, 4)
    Rt = np.stack([np.hstack([c.R, c.t.reshape(3, 1)]) for c in cameras])[obs.view]
    X_h = np.hstack([X[obs.point], np.ones((len(obs.point), 1))])
    p = np.einsum("mij,mj->mi", P, X_h)
    depths = np.einsum("mj,mj->m", Rt[:, 2], X_h)
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / p[:, 2:3], depths


def reprojection_errors(cameras: Sequence[Camera], obs: Observations, X: np.ndarray) -> np.ndarray:
    """Pixel distance between each observation and its point's projection, (M,)."""
    pixels, _ = project_observations(cameras, obs, X)
    return np.linalg.norm(pixels - obs.xy, axis=1)


def per_point_mean(values: np.ndarray, point: np.ndarray, n_points: int) -> np.ndarray:
    """Mean of per-observation ``values`` for each point (nan for unobserved points)."""
    total = np.bincount(point, weights=values, minlength=n_points)
    count = np.bincount(point, minlength=n_points)
    with np.errstate(divide="ignore", invalid="ignore"):
        return total / count


@dataclass(frozen=True)
class ReprojectionStats:
    n: int
    mean: float
    median: float
    p90: float
    max: float

    @classmethod
    def from_errors(cls, errors: np.ndarray) -> ReprojectionStats:
        errors = np.asarray(errors, dtype=float)
        if errors.size == 0:
            return cls(0, np.nan, np.nan, np.nan, np.nan)
        return cls(
            n=int(errors.size),
            mean=float(errors.mean()),
            median=float(np.median(errors)),
            p90=float(np.percentile(errors, 90)),
            max=float(errors.max()),
        )

    def to_dict(self) -> dict[str, float]:
        return {k if k == "n" else f"{k}_px": v for k, v in asdict(self).items()}

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
    """Projected pixels (M, 2) and depths (M,) of each observation's 3-D point.

    Works view by view (at most a few hundred cameras), so no per-observation copies of
    the camera matrices are built.
    """
    pixels = np.empty((len(obs), 2))
    depths = np.empty(len(obs))
    for v, idx in obs.by_view():
        pixels[idx], depths[idx] = cameras[v].project(X[obs.point[idx]])
    return pixels, depths


def reprojection_errors(cameras: Sequence[Camera], obs: Observations, X: np.ndarray) -> np.ndarray:
    """Pixel distance between each observation and its point's projection, (M,)."""
    pixels, _ = project_observations(cameras, obs, X)
    return np.linalg.norm(pixels - obs.xy, axis=1)


def per_point_mean(values: np.ndarray, point: np.ndarray, n_points: int) -> np.ndarray:
    """Mean of per-observation ``values`` ((M,) or (M, k)) for each point.

    Unobserved points get nan.
    """
    values = np.asarray(values, dtype=float)
    totals = np.zeros((n_points, *values.shape[1:]))
    np.add.at(totals, point, values)
    count = np.bincount(point, minlength=n_points).reshape(-1, *[1] * (values.ndim - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        return totals / count


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

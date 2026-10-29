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
        cam = cameras[v]
        X_cam = X[obs.point[idx]] @ cam.R.T + cam.t
        p = X_cam @ cam.K.T
        with np.errstate(divide="ignore", invalid="ignore"):
            pixels[idx] = p[:, :2] / p[:, 2:3]
        depths[idx] = X_cam[:, 2]
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
    columns = values.reshape(len(values), -1)
    count = np.bincount(point, minlength=n_points).astype(float)
    totals = np.stack(
        [np.bincount(point, weights=c, minlength=n_points) for c in columns.T], axis=1
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = totals / count[:, None]
    return mean.reshape((n_points, *values.shape[1:]))


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

"""Pinhole camera: intrinsics K and world -> camera pose (R, t)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def camera_centre(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Centre C = -R^T t of a camera with world -> camera transform (R, t)."""
    return -R.T @ t


@dataclass(frozen=True, eq=False)
class Camera:
    K: np.ndarray  # (3, 3)
    R: np.ndarray  # (3, 3) world -> camera
    t: np.ndarray  # (3,)

    @property
    def Rt(self) -> np.ndarray:
        """Extrinsics [R | t], (3, 4)."""
        return np.hstack([self.R, self.t.reshape(3, 1)])

    @property
    def P(self) -> np.ndarray:
        """Projection matrix K [R | t], (3, 4)."""
        return self.K @ self.Rt

    @property
    def centre(self) -> np.ndarray:
        return camera_centre(self.R, self.t)

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Pixels (N, 2) and depths (N,) of world points X (N, 3)."""
        X_cam = X @ self.R.T + self.t
        p = X_cam @ self.K.T
        with np.errstate(divide="ignore", invalid="ignore"):
            return p[:, :2] / p[:, 2:3], X_cam[:, 2]

    def nearest_pixels(self, X: np.ndarray, shape: tuple[int, ...]):
        """Row and column indices (N,) of the pixel nearest each world point's projection,
        whether it falls in an image of ``shape`` in front of the camera (N,), and the
        depths (N,). Indices of points outside are 0."""
        x, z = self.project(X)
        h, w = shape[:2]
        with np.errstate(invalid="ignore"):
            u, v = np.round(x[:, 0]), np.round(x[:, 1])
            inside = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        rows = np.where(inside, v, 0).astype(int)
        cols = np.where(inside, u, 0).astype(int)
        return rows, cols, inside, z

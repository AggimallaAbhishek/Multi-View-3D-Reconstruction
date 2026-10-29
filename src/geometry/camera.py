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

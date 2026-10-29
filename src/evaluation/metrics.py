"""Pose accuracy metrics (comparison against ground truth)."""

from __future__ import annotations

import numpy as np

from src.geometry.epipolar import angle_between_deg
from src.geometry.pose_estimation import rotation_angle_deg


def rotation_error_deg(R_est: np.ndarray, R_true: np.ndarray) -> float:
    """Angle of the rotation taking R_est to R_true, in degrees."""
    return rotation_angle_deg(R_est.T @ R_true)


def translation_direction_error_deg(t_est: np.ndarray, t_true: np.ndarray) -> float:
    """Angle between translation directions (translation is only known up to scale)."""
    return float(angle_between_deg(np.ravel(t_est), np.ravel(t_true)))

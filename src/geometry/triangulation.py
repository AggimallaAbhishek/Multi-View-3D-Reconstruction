"""Triangulation of 3-D points from image correspondences.

Phase 5 needs only two-view linear triangulation (for the cheirality test);
Phase 6 extends this module to N views and quality filters.
"""

from __future__ import annotations

import numpy as np


def triangulate_two_view(
    P_i: np.ndarray, P_j: np.ndarray, x_i: np.ndarray, x_j: np.ndarray
) -> np.ndarray:
    """Linear (DLT) triangulation of N correspondences; returns (N, 3).

    Each observation x ~ P X gives two equations, x * P[2] - P[0] and y * P[2] - P[1];
    the four rows from both views form A X = 0, solved by SVD per point (vectorised).
    ``x_i``/``x_j`` are in whatever coordinates the projection matrices use (pixels
    with P = K [R | t], normalised coordinates with P = [R | t]). Points at infinity
    (parallel rays) come back very far away rather than as nan/inf.
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
    X_h = np.linalg.svd(A)[2][:, -1]  # (N, 4)
    w = X_h[:, 3:]
    # Keep the sign of w, but bound it away from zero so parallel rays stay finite.
    w = np.where(np.abs(w) < 1e-12, np.copysign(1e-12, w), w)
    return X_h[:, :3] / w

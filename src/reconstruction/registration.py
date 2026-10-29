"""Registering a new image: its camera pose from links between its keypoints and
existing landmarks (perspective-n-point with RANSAC, then refinement)."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from src.config import RegistrationConfig
from src.features.extractor import Features
from src.geometry.camera import Camera
from src.reconstruction.camera_registry import CameraRegistry
from src.reconstruction.landmarks import Landmarks
from src.reconstruction.tracks import Tracks

PNP_MIN_POINTS = 4  # fewest 2D-3D links any PnP solver accepts


@dataclass(frozen=True, eq=False)
class Registration:
    """Result of registering one image; failures are results, not exceptions."""

    image: int
    camera: Camera | None
    keypoints: np.ndarray  # (I,) inlier keypoints of the image...
    landmarks: np.ndarray  # (I,) ...and the landmarks they observe
    n_correspondences: int  # 2D-3D links tried
    success: bool
    reason: str | None  # "too_few_correspondences" | "pnp_failed" | "too_few_inliers"

    @property
    def n_inliers(self) -> int:
        return len(self.keypoints)


def register_image(
    image: int,
    landmarks: Landmarks,
    tracks: Tracks,
    features: Sequence[Features],
    K: np.ndarray,
    config: RegistrationConfig,
) -> Registration:
    keypoints, ids = landmarks.correspondences(image, tracks)
    n = len(keypoints)
    none = np.empty(0, np.int64)

    def failed(reason: str, camera: Camera | None = None) -> Registration:
        return Registration(image, camera, none, none, n, False, reason)

    if n < max(PNP_MIN_POINTS, config.min_inliers):
        return failed("too_few_correspondences")

    X = np.ascontiguousarray(landmarks.position[ids], dtype=np.float64)
    x = np.ascontiguousarray(features[image].points[keypoints], dtype=np.float64)
    cv2.setRNGSeed(0)  # RANSAC sampling uses OpenCV's RNG: make runs reproducible
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        X, x, K, None,
        iterationsCount=config.max_iterations,
        reprojectionError=config.reprojection_error_px,
        confidence=config.confidence,
        flags=cv2.SOLVEPNP_SQPNP,
    )  # fmt: skip
    if not ok or inliers is None or len(inliers) < PNP_MIN_POINTS:
        return failed("pnp_failed")
    if config.refine:
        k = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(X[k], x[k], K, None, rvec, tvec)
    camera = Camera(K, cv2.Rodrigues(rvec)[0], tvec.ravel())

    # Inliers of the final pose: in front of the camera and within the threshold.
    pixels, depth = camera.project(X)
    good = (depth > 0) & (np.linalg.norm(pixels - x, axis=1) <= config.reprojection_error_px)
    if good.sum() < config.min_inliers or good.mean() < config.min_inlier_ratio:
        return failed("too_few_inliers", camera)
    return Registration(image, camera, keypoints[good], ids[good], n, True, None)


def next_best_view(
    registry: CameraRegistry,
    landmarks: Landmarks,
    tracks: Tracks,
    n_images: int,
    exclude: Collection[int] = (),
) -> int | None:
    """The unregistered image (not in ``exclude``) with the most 2D-3D links, or None if
    no such image sees any landmark. Ties go to the lower image index."""
    has_landmark = np.zeros(tracks.n_tracks, bool)
    has_landmark[landmarks.track] = True
    links = np.bincount(tracks.image[has_landmark[tracks.track]], minlength=n_images)
    links[list(registry.images) + list(exclude)] = 0
    best = int(np.argmax(links))
    return best if links[best] > 0 else None

"""Registering a new image: its camera pose from links between its keypoints and
existing landmarks (perspective-n-point with RANSAC, then refinement) or, when there are
too few links, from its relative pose to a registered image with the scale fixed by the
few landmarks they share."""

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
    # "too_few_correspondences" | "pnp_failed" | "too_few_inliers" (PnP);
    # "too_few_matches" | "no_relative_pose" | "too_few_links" (relative pose)
    reason: str | None
    method: str = "pnp"  # "pnp" | "relative_pose"

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


def register_from_neighbour(
    image: int,
    registry: CameraRegistry,
    landmarks: Landmarks,
    tracks: Tracks,
    features: Sequence[Features],
    K: np.ndarray,
    config: RegistrationConfig,
) -> Registration:
    """Register ``image`` from the registered image it shares most tracks with.

    The essential matrix of their shared keypoints gives the rotation and the direction
    of the baseline: ``R = R_rel R_n``, ``t = R_rel t_n + s t_rel``. Only the baseline
    length ``s`` is unknown, and each landmark the image sees fixes it linearly
    (``x̂ × (R X + R_rel t_n + s t_rel) = 0``): the median over the links, then a least-
    squares fit on those that reproject within ``reprojection_error_px``. Needs
    ``min_inliers`` essential-matrix inliers but only ``fallback_min_links`` landmarks,
    where PnP needs ``min_inliers`` landmarks.
    """
    keypoints, ids = landmarks.correspondences(image, tracks)
    none = np.empty(0, np.int64)

    def failed(reason: str, camera: Camera | None = None) -> Registration:
        return Registration(image, camera, none, none, len(ids), False, reason, "relative_pose")

    # Keypoint of ``image`` on each track, and the registered image sharing most tracks.
    in_image = tracks.image == image
    keypoint_of_track = np.full(tracks.n_tracks, -1, np.int64)
    keypoint_of_track[tracks.track[in_image]] = tracks.keypoint[in_image]
    shared = np.isin(tracks.image, registry.images) & (keypoint_of_track[tracks.track] >= 0)
    if not shared.any():
        return failed("too_few_matches")
    neighbour = int(np.bincount(tracks.image[shared]).argmax())
    pair = shared & (tracks.image == neighbour)
    if pair.sum() < config.min_inliers:
        return failed("too_few_matches")

    cam_n = registry.camera(neighbour)
    x_n = _normalised(features[neighbour].points[tracks.keypoint[pair]], cam_n.K)
    x_m = _normalised(features[image].points[keypoint_of_track[tracks.track[pair]]], K)
    focal = float(np.sqrt(K[0, 0] * K[1, 1]))
    cv2.setRNGSeed(0)  # USAC with local optimisation: plain RANSAC's E was 1-3 deg off
    E, mask = cv2.findEssentialMat(
        x_n, x_m, np.eye(3), method=cv2.USAC_ACCURATE, prob=config.confidence,
        threshold=config.fallback_epipolar_threshold_px / focal,
    )  # fmt: skip
    if E is None or E.shape != (3, 3):
        return failed("no_relative_pose")
    n_good, R_rel, t_rel, _ = cv2.recoverPose(E, x_n, x_m, np.eye(3), mask=mask)
    if n_good < config.min_inliers:
        return failed("no_relative_pose")

    R, t0, direction = R_rel @ cam_n.R, R_rel @ cam_n.t, t_rel.ravel()
    if len(ids) < config.fallback_min_links:
        return failed("too_few_links")
    rays = np.column_stack([_normalised(features[image].points[keypoints], K), np.ones(len(ids))])
    c = np.cross(rays, landmarks.position[ids] @ R.T + t0)  # x̂ × (R X + t0) ...
    d = np.cross(rays, direction)  # ... + s x̂ × t_rel = 0
    per_link = -np.sum(c * d, axis=1) / np.maximum(np.sum(d * d, axis=1), 1e-300)
    s = float(np.median(per_link))
    for _ in range(2):  # inliers of the scale, then the scale refitted on them
        camera = Camera(K, R, t0 + s * direction)
        pixels, depth = camera.project(landmarks.position[ids])
        x = features[image].points[keypoints]
        good = (depth > 0) & (np.linalg.norm(pixels - x, axis=1) <= config.reprojection_error_px)
        if good.sum() < config.fallback_min_links:
            return failed("too_few_links", camera)
        s = float(-np.sum(c[good] * d[good]) / np.sum(d[good] * d[good]))
    camera = Camera(K, R, t0 + s * direction)
    return Registration(
        image, camera, keypoints[good], ids[good], len(ids), True, None, "relative_pose"
    )


def _normalised(pixels: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixels (N, 2) to normalised image coordinates ``K^-1 x`` (N, 2)."""
    return (pixels - K[:2, 2]) @ np.linalg.inv(K[:2, :2]).T


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

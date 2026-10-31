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
from src.geometry.epipolar import ESTIMATORS, homogeneous, normalize_points
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
    # "too_few_matches" | "no_relative_pose" | "bad_scale" | "too_few_links" (relative pose)
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
    """Register ``image`` from its relative pose to a registered image.

    Registered images are tried in order of shared tracks (at least ``min_inliers``).
    The essential matrix of the shared keypoints gives the rotation and the direction
    of the baseline: ``R = R_rel R_n``, ``t = R_rel t_n + s t_rel``. Only the baseline
    length ``s`` is unknown, and each landmark the image sees fixes it linearly
    (``x̂ × (R X + R_rel t_n + s t_rel) = 0``): the median over the links, refitted by
    least squares on those that reproject within ``reprojection_error_px``. It must be
    positive (the image in front of its neighbour's baseline, as ``recoverPose`` chose),
    and at least ``fallback_min_links`` links, and ``min_inlier_ratio`` of them, must
    fit the final pose — where PnP needs ``min_inliers`` links. The pose is not refined
    here; the bundle adjustment that follows refines it.
    """
    keypoints, ids = landmarks.correspondences(image, tracks)
    no_links = np.empty(0, np.int64)

    def failed(reason: str, camera: Camera | None = None) -> Registration:
        return Registration(image, camera, no_links, no_links, len(ids), False, reason,
                            "relative_pose")  # fmt: skip

    if len(ids) < config.fallback_min_links:
        return failed("too_few_links")
    pixels = features[image].points[keypoints]
    rays = homogeneous(normalize_points(pixels, K))
    reason = "too_few_matches"
    for neighbour, x_n, x_m in _shared_keypoints(image, registry, tracks, features, config):
        cam_n = registry.camera(neighbour)
        relative = _relative_pose(normalize_points(x_n, cam_n.K), normalize_points(x_m, K), K,
                                  config)  # fmt: skip
        if relative is None:
            reason = "no_relative_pose"
            continue
        R_rel, direction = relative
        R, offset = R_rel @ cam_n.R, R_rel @ cam_n.t
        # x̂ × (R X + offset) + s x̂ × direction = 0, one linear equation in s per link.
        a = np.cross(rays, landmarks.position[ids] @ R.T + offset)
        b = np.cross(rays, direction)
        per_link = -np.sum(a * b, axis=1) / np.maximum(np.sum(b * b, axis=1), np.finfo(float).tiny)
        s = float(np.median(per_link))
        good = _fits(Camera(K, R, offset + s * direction), landmarks.position[ids], pixels, config)
        if good.sum() >= config.fallback_min_links:
            s = float(-np.sum(a[good] * b[good]) / np.sum(b[good] * b[good]))
        camera = Camera(K, R, offset + s * direction)
        good = _fits(camera, landmarks.position[ids], pixels, config)  # of the final pose
        if s <= 0:
            reason = "bad_scale"
        elif good.sum() < config.fallback_min_links or good.mean() < config.min_inlier_ratio:
            reason = "too_few_links"
        else:
            return Registration(image, camera, keypoints[good], ids[good], len(ids), True,
                                None, "relative_pose")  # fmt: skip
    return failed(reason)


def _shared_keypoints(image, registry, tracks, features, config):
    """(neighbour, its keypoints, the image's keypoints) for every registered image that
    shares at least ``min_inliers`` tracks with ``image``, most shared first."""
    in_image = tracks.image == image
    keypoint_of_track = np.full(tracks.n_tracks, -1, np.int64)
    keypoint_of_track[tracks.track[in_image]] = tracks.keypoint[in_image]
    shared = np.isin(tracks.image, registry.images) & (keypoint_of_track[tracks.track] >= 0)
    counts = np.bincount(tracks.image[shared], minlength=len(features))
    for neighbour in np.argsort(-counts, kind="stable"):  # ties: lower index first
        if counts[neighbour] < config.min_inliers:
            break
        pair = shared & (tracks.image == neighbour)
        yield (int(neighbour), features[neighbour].points[tracks.keypoint[pair]],
               features[image].points[keypoint_of_track[tracks.track[pair]]])  # fmt: skip


def _relative_pose(x_n, x_m, K, config):
    """(R, unit t) from normalised points of the neighbour to those of the new image, or
    None without ``min_inliers`` inliers in front of both cameras."""
    focal = float(np.sqrt(K[0, 0] * K[1, 1]))
    cv2.setRNGSeed(0)
    E, mask = cv2.findEssentialMat(
        x_n, x_m, np.eye(3), method=ESTIMATORS[config.fallback_estimator],
        prob=config.confidence, threshold=config.fallback_epipolar_threshold_px / focal,
    )  # fmt: skip
    if E is None or E.shape != (3, 3):
        return None
    n_good, R, t, _ = cv2.recoverPose(E, x_n, x_m, np.eye(3), mask=mask)
    return (R, t.ravel()) if n_good >= config.min_inliers else None


def _fits(camera: Camera, X: np.ndarray, pixels: np.ndarray, config) -> np.ndarray:
    """Links in front of ``camera`` that reproject within ``reprojection_error_px``."""
    projected, depth = camera.project(X)
    return (depth > 0) & (
        np.linalg.norm(projected - pixels, axis=1) <= config.reprojection_error_px
    )


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

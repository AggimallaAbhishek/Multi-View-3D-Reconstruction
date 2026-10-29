"""Incremental structure from motion: start from the initial pair, then grow.

The world frame is the camera of the initial pair's image i, and the initial baseline
has length 1, so the reconstruction is known up to a global similarity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.config import BundleAdjustmentConfig, Config, TriangulationConfig
from src.features.extractor import Features
from src.geometry.camera import Camera
from src.geometry.pose_estimation import InitialPair
from src.geometry.reprojection import per_point_mean, project_observations, reprojection_errors
from src.geometry.triangulation import (
    Observations,
    triangulate_and_filter,
    triangulation_angles_deg,
)
from src.reconstruction.bundle_adjustment import BundleAdjustmentResult, bundle_adjust
from src.reconstruction.camera_registry import CameraRegistry
from src.reconstruction.landmarks import Landmarks, keypoint_pixels
from src.reconstruction.registration import Registration, next_best_view, register_image
from src.reconstruction.tracks import Tracks

FRAME = "camera of the initial pair's image i at the origin, initial baseline length 1"


@dataclass(frozen=True, eq=False)
class Reconstruction:
    registry: CameraRegistry
    landmarks: Landmarks
    attempts: list[Registration]  # every registration tried, in order, failures included

    def reprojection_errors(self, features: Sequence[Features]) -> np.ndarray:
        """Pixel error of every observation, in ``landmarks`` observation order, (M,)."""
        cameras, obs, X = self.landmarks.as_problem(self.registry, features)
        return reprojection_errors(cameras, obs, X)

    def unregistered(self, n_images: int) -> list[int]:
        """Indices of the images that are not part of the reconstruction."""
        return [m for m in range(n_images) if m not in self.registry]


def reconstruct(
    pair: InitialPair,
    intrinsics: Sequence[np.ndarray],
    tracks: Tracks,
    features: Sequence[Features],
    config: Config,
) -> Reconstruction:
    """Start from the initial pair, then register images one by one, best-linked first.

    An image that fails to register is skipped until another image succeeds (its new
    landmarks may give it enough links). Stops when no remaining image can be added.
    With bundle adjustment enabled, the newest ``local_window`` cameras and the points
    they see are adjusted after each registration, and everything every
    ``global_every`` registrations. With ``refine_structure`` on, each registration (and
    its adjustment) is followed by ``refine_structure``. At the end, global adjustment
    and refinement alternate until refinement changes nothing (at most
    ``max_refinements`` rounds); either can be off.
    """
    ba = config.bundle_adjustment
    registry, landmarks = initialize(pair, intrinsics, tracks, features, config.triangulation)
    attempts: list[Registration] = []
    failed: set[int] = set()
    while (m := next_best_view(registry, landmarks, tracks, len(features), failed)) is not None:
        reg = register_image(m, landmarks, tracks, features, intrinsics[m], config.registration)
        attempts.append(reg)
        if not reg.success:
            failed.add(m)
            continue
        registry.register(m, reg.camera)
        landmarks.add_observations(m, reg.keypoints, reg.landmarks)
        triangulate_new_tracks(registry, landmarks, tracks, features, config.triangulation)
        failed.clear()
        if ba.enabled:
            n_added = len(registry) - 2
            if ba.global_every > 0 and n_added % ba.global_every == 0:
                adjust(registry, landmarks, features, ba)
            elif ba.local_window > 0:
                newest = range(len(registry) - ba.local_window, len(registry))
                adjust(registry, landmarks, features, ba, free_views=newest)
        if ba.refine_structure:
            refine_structure(registry, landmarks, tracks, features, config.triangulation)
    for _ in range(ba.max_refinements if ba.enabled or ba.refine_structure else 0):
        if ba.enabled:
            adjust(registry, landmarks, features, ba)
        if not ba.refine_structure or not refine_structure(
            registry, landmarks, tracks, features, config.triangulation
        ):
            break
    return Reconstruction(registry, landmarks, attempts)


def adjust(
    registry: CameraRegistry,
    landmarks: Landmarks,
    features: Sequence[Features],
    config: BundleAdjustmentConfig,
    free_views: Sequence[int] | None = None,
) -> BundleAdjustmentResult | None:
    """Bundle-adjust the reconstruction in place. ``free_views`` (registration order)
    limits it to those cameras and the landmarks they see; the first camera always stays
    fixed. ``None`` adjusts every camera and landmark."""
    cameras, obs, X = landmarks.as_problem(registry, features)
    free = set(range(1, len(cameras)))
    if free_views is not None:
        free &= {v for v in free_views if v >= 0}
    if not free or not len(obs):
        return None
    points = np.unique(obs.point[np.isin(obs.view, list(free))])
    new_index = np.full(len(X), -1, np.int64)
    new_index[points] = np.arange(len(points))
    sub = obs.subset(new_index[obs.point] >= 0)
    sub = Observations(sub.view, new_index[sub.point], sub.xy)
    fixed = [v for v in range(len(cameras)) if v not in free]
    result = bundle_adjust(cameras, sub, X[points], config, fixed=fixed)
    for v in free:
        registry.update(registry.images[v], result.cameras[v])
    landmarks.position[points] = result.points
    errors = reprojection_errors(result.cameras, sub, result.points)
    landmarks.error[points] = per_point_mean(errors, sub.point, len(points))
    return result


def initialize(
    pair: InitialPair,
    intrinsics: Sequence[np.ndarray],
    tracks: Tracks,
    features: Sequence[Features],
    config: TriangulationConfig,
) -> tuple[CameraRegistry, Landmarks]:
    """Register the initial pair (image i at the origin, image j at the relative pose)
    and triangulate every track the two images share."""
    registry = CameraRegistry()
    registry.register(pair.i, Camera(intrinsics[pair.i], np.eye(3), np.zeros(3)))
    registry.register(pair.j, Camera(intrinsics[pair.j], pair.pose.R, pair.pose.t))
    landmarks = Landmarks()
    triangulate_new_tracks(registry, landmarks, tracks, features, config)
    return registry, landmarks


def triangulate_new_tracks(
    registry: CameraRegistry,
    landmarks: Landmarks,
    tracks: Tracks,
    features: Sequence[Features],
    config: TriangulationConfig,
) -> np.ndarray:
    """Add a landmark for every track without one that at least two registered images
    observe and whose triangulation passes the filters. Returns the new landmark ids."""
    images = registry.images
    in_registered = np.isin(tracks.image, images)
    has_landmark = np.zeros(tracks.n_tracks, bool)
    has_landmark[landmarks.track] = True
    views = np.bincount(tracks.track[in_registered], minlength=tracks.n_tracks)
    candidates = np.flatnonzero((views >= 2) & ~has_landmark)

    point_of_track = np.full(tracks.n_tracks, -1, np.int64)
    point_of_track[candidates] = np.arange(len(candidates))
    sel = in_registered & (point_of_track[tracks.track] >= 0)
    view_of_image = np.full(int(tracks.image.max(initial=0)) + 1, -1, np.int64)
    view_of_image[images] = np.arange(len(images))
    obs = Observations(
        view=view_of_image[tracks.image[sel]],
        point=point_of_track[tracks.track[sel]],
        xy=keypoint_pixels(features, tracks.image[sel], tracks.keypoint[sel]),
    )
    cameras = [registry.camera(m) for m in images]
    result = triangulate_and_filter(cameras, obs, config, n_points=len(candidates))

    valid = result.valid
    ids = landmarks.add(candidates[valid], result.points[valid], tracks, set(images))
    landmarks.error[ids] = per_point_mean(result.errors, obs.point, len(candidates))[valid]
    return ids


def refine_structure(
    registry: CameraRegistry,
    landmarks: Landmarks,
    tracks: Tracks,
    features: Sequence[Features],
    config: TriangulationConfig,
) -> int:
    """After bundle adjustment: attach track keypoints that now fit their landmark, drop
    observations and landmarks that no longer fit, and triangulate the tracks left
    without a landmark (the dropped ones included). Returns how many observations
    were added or removed."""
    changed = complete_tracks(registry, landmarks, tracks, features, config)
    changed += filter_landmarks(registry, landmarks, features, config)
    new = triangulate_new_tracks(registry, landmarks, tracks, features, config)
    return changed + int(np.isin(landmarks.obs_landmark, new).sum())


def complete_tracks(
    registry: CameraRegistry,
    landmarks: Landmarks,
    tracks: Tracks,
    features: Sequence[Features],
    config: TriangulationConfig,
) -> int:
    """Attach to each landmark the keypoints of its track, in registered images, that it
    does not observe yet (e.g. links PnP rejected) and that reproject in front of the
    camera within ``max_reprojection_error_px``. Returns how many were attached.
    Landmark errors are not updated (``filter_landmarks`` does that)."""
    images = registry.images
    landmark_of_track = np.full(tracks.n_tracks, -1, np.int64)
    landmark_of_track[landmarks.track] = np.arange(len(landmarks))
    lid = landmark_of_track[tracks.track]
    # A track has at most one keypoint per image, so (landmark, image) is an observation.
    n = int(tracks.image.max(initial=0)) + 1
    observed = np.isin(lid * n + tracks.image, landmarks.obs_landmark * n + landmarks.obs_image)
    candidate = (lid >= 0) & np.isin(tracks.image, images) & ~observed
    added = 0
    for m in np.unique(tracks.image[candidate]).tolist():
        sel = np.flatnonzero(candidate & (tracks.image == m))
        pixels, depth = registry.camera(m).project(landmarks.position[lid[sel]])
        error = np.linalg.norm(pixels - features[m].points[tracks.keypoint[sel]], axis=1)
        fits = sel[(depth > 0) & (error <= config.max_reprojection_error_px)]
        landmarks.add_observations(m, tracks.keypoint[fits], lid[fits])
        added += len(fits)
    return added


def filter_landmarks(
    registry: CameraRegistry,
    landmarks: Landmarks,
    features: Sequence[Features],
    config: TriangulationConfig,
) -> int:
    """Drop single observations that lie behind their camera or reproject farther than
    ``max_reprojection_error_px``, then whole landmarks left with fewer than two
    observations or a triangulation angle below ``min_angle_deg``. Unlike the filter at
    triangulation, one bad observation does not cost a long track its landmark.
    Recomputes the landmark errors; returns how many observations were removed."""
    n_obs = len(landmarks.obs_image)
    cameras, obs, X = landmarks.as_problem(registry, features)
    pixels, depth = project_observations(cameras, obs, X)
    errors = np.linalg.norm(pixels - obs.xy, axis=1)
    good = (depth > 0) & (errors <= config.max_reprojection_error_px)
    landmarks.keep_observations(good)
    obs = obs.subset(good)
    landmarks.error = per_point_mean(errors[good], obs.point, len(X))
    angles = triangulation_angles_deg(cameras, obs, X)  # nan below two observations
    landmarks.keep(angles >= config.min_angle_deg)
    return n_obs - len(landmarks.obs_image)

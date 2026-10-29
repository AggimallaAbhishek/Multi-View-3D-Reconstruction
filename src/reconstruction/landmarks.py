"""3-D landmarks of an incremental reconstruction and the keypoints that observe them.

Each landmark comes from one feature track. Observations are a flat list: observation k
is keypoint ``obs_keypoint[k]`` of image ``obs_image[k]``, seeing landmark
``obs_landmark[k]``; only registered images have observations.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

import numpy as np

from src.features.extractor import Features
from src.geometry.camera import Camera
from src.geometry.triangulation import Observations
from src.reconstruction.camera_registry import CameraRegistry
from src.reconstruction.tracks import Tracks

_INT = np.int64


def keypoint_pixels(
    features: Sequence[Features], image: np.ndarray, keypoint: np.ndarray
) -> np.ndarray:
    """(K, 2) pixel coordinates of keypoint ``keypoint[k]`` of image ``image[k]``."""
    xy = np.empty((len(image), 2))
    for m in np.unique(image).tolist():
        k = image == m
        xy[k] = features[m].points[keypoint[k]]
    return xy


class Landmarks:
    def __init__(self) -> None:
        self.position = np.empty((0, 3))  # (N, 3) world coordinates
        self.color = np.zeros((0, 3), np.uint8)  # (N, 3) RGB, filled in when coloured
        self.error = np.empty(0)  # (N,) mean reprojection error in px (nan until computed)
        self.track = np.empty(0, _INT)  # (N,) track each landmark was triangulated from
        self.obs_landmark = np.empty(0, _INT)
        self.obs_image = np.empty(0, _INT)
        self.obs_keypoint = np.empty(0, _INT)

    def __len__(self) -> int:
        return len(self.position)

    def add(
        self,
        track_ids: Sequence[int] | np.ndarray,
        positions: np.ndarray,
        tracks: Tracks,
        registered: Collection[int],
    ) -> np.ndarray:
        """New landmarks for ``track_ids`` at ``positions`` (N, 3), observed by every
        keypoint of their track that lies in a registered image. Returns their ids."""
        track_ids = np.asarray(track_ids, _INT)
        ids = np.arange(len(self), len(self) + len(track_ids))
        landmark_of_track = np.full(tracks.n_tracks, -1, _INT)
        landmark_of_track[track_ids] = ids
        obs = (landmark_of_track[tracks.track] >= 0) & np.isin(tracks.image, list(registered))

        self.position = np.vstack([self.position, np.asarray(positions, float).reshape(-1, 3)])
        self.color = np.vstack([self.color, np.zeros((len(ids), 3), np.uint8)])
        self.error = np.concatenate([self.error, np.full(len(ids), np.nan)])
        self.track = np.concatenate([self.track, track_ids])
        self._append_observations(
            landmark_of_track[tracks.track[obs]], tracks.image[obs], tracks.keypoint[obs]
        )
        return ids

    def add_observations(self, image: int, keypoints: np.ndarray, landmarks: np.ndarray) -> None:
        """Keypoints of a newly registered ``image`` that see existing ``landmarks``."""
        self._append_observations(
            np.asarray(landmarks, _INT), np.full(len(landmarks), image, _INT), keypoints
        )

    def correspondences(self, image: int, tracks: Tracks) -> tuple[np.ndarray, np.ndarray]:
        """2D-3D links of an unregistered ``image``: (keypoints, landmark ids) for every
        keypoint whose track already has a landmark."""
        landmark_of_track = np.full(tracks.n_tracks, -1, _INT)
        landmark_of_track[self.track] = np.arange(len(self))
        in_image = tracks.image == image
        landmarks = landmark_of_track[tracks.track[in_image]]
        found = landmarks >= 0
        return tracks.keypoint[in_image][found], landmarks[found]

    def as_problem(
        self, registry: CameraRegistry, features: Sequence[Features]
    ) -> tuple[list[Camera], Observations, np.ndarray]:
        """(cameras, observations, points) for triangulation, reprojection or bundle
        adjustment: view v is the v-th registered image, point p is landmark p."""
        images = registry.images
        view_of_image = {image: v for v, image in enumerate(images)}
        view = np.array([view_of_image[m] for m in self.obs_image.tolist()], _INT)
        xy = keypoint_pixels(features, self.obs_image, self.obs_keypoint)
        cameras = [registry.camera(m) for m in images]
        return cameras, Observations(view, self.obs_landmark, xy), self.position

    def keep(self, mask: np.ndarray) -> None:
        """Keep only the landmarks where ``mask`` is true (renumbered in order)."""
        mask = np.asarray(mask, bool)
        new_id = np.cumsum(mask) - 1
        obs = mask[self.obs_landmark]
        self.position, self.color = self.position[mask], self.color[mask]
        self.error, self.track = self.error[mask], self.track[mask]
        self.obs_landmark = new_id[self.obs_landmark[obs]]
        self.obs_image, self.obs_keypoint = self.obs_image[obs], self.obs_keypoint[obs]

    def _append_observations(self, landmark, image, keypoint) -> None:
        self.obs_landmark = np.concatenate([self.obs_landmark, np.asarray(landmark, _INT)])
        self.obs_image = np.concatenate([self.obs_image, np.asarray(image, _INT)])
        self.obs_keypoint = np.concatenate([self.obs_keypoint, np.asarray(keypoint, _INT)])

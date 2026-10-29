"""Synthetic scenes wired up as SfM inputs: keypoints, features and tracks with known truth.

Unlike tests/synthetic.py this module uses the pipeline's own types (Features, Tracks).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.features.extractor import Features
from src.geometry.camera import Camera
from src.geometry.pose_estimation import InitialPair, RelativePose
from src.reconstruction.tracks import Edge, Tracks, ViewGraph, build_tracks
from tests.synthetic import SyntheticScene, exact_matches, make_scene, shuffled_keypoints


@dataclass(frozen=True, eq=False)
class SfmScene:
    scene: SyntheticScene
    point_of: list[np.ndarray]  # point_of[m][k]: scene point behind keypoint k of image m
    features: list[Features]  # keypoint k of image m at scene.observations[m, point_of[m][k]]
    tracks: Tracks  # from exact matches between every pair of images
    truth: np.ndarray  # (n_tracks,) scene point of each track

    def tracks_seen_by(self, images: set[int], at_least: int = 2) -> list[int]:
        """Tracks observed in at least ``at_least`` of ``images``."""
        t = self.tracks
        return [k for k in range(t.n_tracks)
                if len(set(t.image[t.track == k].tolist()) & images) >= at_least]  # fmt: skip


def make_sfm_scene(n_cameras: int = 4, n_points: int = 250, seed: int = 5, **kwargs) -> SfmScene:
    scene = make_scene(n_points=n_points, n_cameras=n_cameras, seed=seed, **kwargs)
    point_of = shuffled_keypoints(scene)
    edges = [
        Edge(i, j, *exact_matches(point_of, i, j))
        for i in range(n_cameras)
        for j in range(i + 1, n_cameras)
    ]
    tracks = build_tracks(ViewGraph(edges))
    first = np.unique(tracks.track, return_index=True)[1]
    truth = np.array(
        [point_of[m][k] for m, k in zip(tracks.image[first], tracks.keypoint[first], strict=True)]
    )
    features = []
    for m, points in enumerate(point_of):
        n = len(points)
        features.append(
            Features(scene.observations[m, points], np.zeros((n, 128), np.float32),
                     np.ones(n), np.zeros(n), np.ones(n), "synthetic")
        )  # fmt: skip
    return SfmScene(scene, point_of, features, tracks, truth)


def initial_pair(scene, i, j):
    """InitialPair carrying the true relative pose (unit baseline), as Phase 5 would."""
    R, t = scene.relative_pose(i, j)
    ok = np.ones(1, bool)
    pose = RelativePose(R, t / np.linalg.norm(t), ok, ok, 10.0, True, None)
    return InitialPair(i, j, pose, None, np.empty((0, 2)), np.empty((0, 2)))


def in_pair_frame(scene, m, i=0, j=1):
    """True camera m in the initial pair's frame (camera i at origin, baseline 1)."""
    b = np.linalg.norm(scene.relative_pose(i, j)[1])
    R = scene.rotations[m] @ scene.rotations[i].T
    return Camera(scene.K, R, (scene.translations[m] - R @ scene.translations[i]) / b)

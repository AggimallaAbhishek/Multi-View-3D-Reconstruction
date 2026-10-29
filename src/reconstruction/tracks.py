"""Feature tracks: the same scene point observed across several images.

Verified matches form a view graph; union-find over its (image, keypoint) nodes links
chains of matches (a-b, b-c) into tracks.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.features.matcher import Matches
from src.geometry.epipolar import TwoViewGeometry


@dataclass(frozen=True, eq=False)
class Edge:
    """Verified correspondences of one image pair: keypoint ``idx_i[k]`` <-> ``idx_j[k]``."""

    i: int
    j: int
    idx_i: np.ndarray  # (M,) keypoint indices in image i
    idx_j: np.ndarray  # (M,) keypoint indices in image j


@dataclass(frozen=True, eq=False)
class ViewGraph:
    """Image pairs that passed geometric verification, with their inlier correspondences."""

    edges: list[Edge]

    @classmethod
    def from_verified(
        cls, matches: Sequence[Matches], geometries: Sequence[TwoViewGeometry]
    ) -> ViewGraph:
        return cls(
            [
                Edge(m.i, m.j, m.idx_i[g.inlier_mask], m.idx_j[g.inlier_mask])
                for m, g in zip(matches, geometries, strict=True)
                if g.success
            ]
        )


@dataclass(frozen=True, eq=False)
class Tracks:
    """Flat observation list, sorted by track then image: observation k is keypoint
    ``keypoint[k]`` of image ``image[k]``, belonging to track ``track[k]``."""

    track: np.ndarray  # (K,) int
    image: np.ndarray  # (K,) int
    keypoint: np.ndarray  # (K,) int

    @property
    def n_tracks(self) -> int:
        return int(self.track.max()) + 1 if len(self.track) else 0


def build_tracks(graph: ViewGraph) -> Tracks:
    """Link verified matches into tracks with union-find over (image, keypoint) nodes."""
    nodes: dict[tuple[int, int], int] = {}
    parent: list[int] = []

    def node(image: int, keypoint: int) -> int:
        key = (image, keypoint)
        if key not in nodes:
            nodes[key] = len(parent)
            parent.append(len(parent))
        return nodes[key]

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]  # path halving
            a = parent[a]
        return a

    for e in graph.edges:
        for a, b in zip(e.idx_i.tolist(), e.idx_j.tolist(), strict=True):
            ra, rb = find(node(e.i, a)), find(node(e.j, b))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)

    keys = np.array(list(nodes), dtype=np.int64).reshape(-1, 2)
    roots = np.array([find(n) for n in range(len(parent))], dtype=np.int64)
    _, component = np.unique(roots, return_inverse=True)

    # A component reaching two keypoints of one image is ambiguous: drop all of it.
    image_pairs = np.unique(np.stack([component, keys[:, 0]], axis=1), axis=0)
    per_component = np.bincount(component)
    images_per_component = np.bincount(image_pairs[:, 0], minlength=len(per_component))
    keep = (images_per_component == per_component)[component]

    _, track = np.unique(component[keep], return_inverse=True)
    keys = keys[keep]
    order = np.lexsort((keys[:, 0], track))
    return Tracks(track[order], keys[order, 0], keys[order, 1])

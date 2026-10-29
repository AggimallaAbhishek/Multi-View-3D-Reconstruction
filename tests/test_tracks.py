import numpy as np

from src.features.matcher import Matches
from src.geometry.epipolar import TwoViewGeometry
from src.reconstruction.tracks import Edge, ViewGraph, build_tracks
from tests.synthetic import exact_matches, make_scene, shuffled_keypoints


def _graph(*edges):
    """Edges as (i, j, [(keypoint_i, keypoint_j), ...])."""
    return ViewGraph([Edge(i, j, *np.array(kp).reshape(-1, 2).T) for i, j, kp in edges])


def _as_sets(tracks):
    """Each track as a frozenset of (image, keypoint) observations."""
    pairs = zip(tracks.track.tolist(), tracks.image.tolist(), tracks.keypoint.tolist(), strict=True)
    by_track: dict[int, set] = {}
    for t, image, keypoint in pairs:
        by_track.setdefault(t, set()).add((image, keypoint))
    return {frozenset(obs) for obs in by_track.values()}


def test_chained_matches_merge_into_one_track():
    # Keypoint 5 in image 0 <-> 7 in image 1 <-> 2 in image 2.
    tracks = build_tracks(_graph((0, 1, [(5, 7)]), (1, 2, [(7, 2)])))
    assert tracks.n_tracks == 1
    assert _as_sets(tracks) == {frozenset({(0, 5), (1, 7), (2, 2)})}


def test_track_with_two_keypoints_in_one_image_is_dropped():
    # 0:1 <-> 1:1 <-> 2:1 <-> 0:9 closes a loop onto a *different* keypoint of image 0.
    graph = _graph(
        (0, 1, [(1, 1), (3, 3)]),
        (1, 2, [(1, 1), (3, 3)]),
        (0, 2, [(9, 1), (3, 3)]),
    )
    tracks = build_tracks(graph)
    assert _as_sets(tracks) == {frozenset({(0, 3), (1, 3), (2, 3)})}
    assert tracks.n_tracks == 1  # ids stay dense after dropping


def test_no_verified_pairs_gives_no_tracks():
    tracks = build_tracks(ViewGraph([]))
    assert tracks.n_tracks == 0
    assert len(tracks.track) == len(tracks.image) == len(tracks.keypoint) == 0


def test_tracks_recover_true_points_of_a_synthetic_scene():
    scene = make_scene(n_points=300, n_cameras=5, seed=3)
    point_of = shuffled_keypoints(scene)
    edges = [Edge(i, j, *exact_matches(point_of, i, j)) for i in range(5) for j in range(i + 1, 5)]
    tracks = build_tracks(ViewGraph(edges))

    seen_twice = np.flatnonzero(scene.visible.sum(axis=0) >= 2)
    assert tracks.n_tracks == len(seen_twice)
    recovered = set()
    for t in range(tracks.n_tracks):
        k = tracks.track == t
        obs = zip(tracks.image[k], tracks.keypoint[k], strict=True)
        points = {int(point_of[m][kp]) for m, kp in obs}
        assert len(points) == 1  # one track, one scene point
        (p,) = points
        assert set(tracks.image[k].tolist()) == set(np.flatnonzero(scene.visible[:, p]).tolist())
        recovered.add(p)
    assert recovered == set(seen_twice.tolist())


def _geometry(inlier_mask, success=True):
    mask = np.asarray(inlier_mask, dtype=bool)
    return TwoViewGeometry(
        "essential", np.eye(3), np.eye(3), mask, np.zeros(len(mask)), 0, False, success,
        None if success else "too_few_inliers",
    )  # fmt: skip


def test_view_graph_keeps_only_inliers_of_verified_pairs():
    def matches(i, j, idx_i, idx_j):
        return Matches(i, j, np.array(idx_i), np.array(idx_j), np.zeros(len(idx_i), np.float32))

    graph = ViewGraph.from_verified(
        [matches(0, 1, [0, 1, 2], [5, 6, 7]), matches(0, 2, [3, 4], [8, 9])],
        [_geometry([True, False, True]), _geometry([True, True], success=False)],
    )
    (edge,) = graph.edges
    assert (edge.i, edge.j) == (0, 1)
    np.testing.assert_array_equal(edge.idx_i, [0, 2])
    np.testing.assert_array_equal(edge.idx_j, [5, 7])

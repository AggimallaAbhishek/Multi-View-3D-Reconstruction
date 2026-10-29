import cv2
import numpy as np
import pytest

from src.config import PoseConfig, VerificationConfig, load_config
from src.evaluation.metrics import rotation_error_deg, translation_direction_error_deg
from src.geometry.epipolar import normalize_points, skew, verify_pair
from src.geometry.pose_estimation import (
    PoseStats,
    RelativePose,
    decompose_essential,
    essential_for_pose,
    estimate_pair_poses,
    estimate_relative_pose,
    is_initial_candidate,
    rotation_angle_deg,
    select_initial_pair,
)
from src.geometry.triangulation import triangulate_two_view
from tests.synthetic import intrinsics, look_at, make_scene, project

CFG = load_config().pose  # the defaults users actually run with


def _true_E(scene, i, j):
    R, t = scene.relative_pose(i, j)
    return skew(t) @ R


@pytest.fixture(scope="module")
def scene():
    return make_scene(n_points=400, n_cameras=3, noise_px=0.0, seed=0)


@pytest.fixture(scope="module")
def noisy():
    return make_scene(n_points=400, n_cameras=3, noise_px=1.0, seed=0)


# --- building blocks -----------------------------------------------------------------------


def test_rotation_metrics():
    R = cv2.Rodrigues(np.array([0.0, np.deg2rad(30), 0.0]))[0]
    assert rotation_angle_deg(R) == pytest.approx(30.0)
    assert rotation_error_deg(R, np.eye(3)) == pytest.approx(30.0)
    assert translation_direction_error_deg(np.array([1, 0, 0]), np.array([2, 0, 0])) == 0.0
    assert translation_direction_error_deg(np.array([1, 0, 0]), np.array([0, 3, 0])) == (
        pytest.approx(90.0)
    )


def test_decompose_essential_contains_true_pose(scene):
    R_true, t_true = scene.relative_pose(0, 1)
    candidates = decompose_essential(_true_E(scene, 0, 1))
    assert len(candidates) == 4
    for R, t in candidates:
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.linalg.det(R) == pytest.approx(1.0)
        assert np.linalg.norm(t) == pytest.approx(1.0)
    errors = [
        rotation_error_deg(R, R_true) + translation_direction_error_deg(t, t_true)
        for R, t in candidates
    ]
    assert min(errors) < 1e-4  # degrees; arccos near 1 limits precision to ~1e-6


def test_triangulate_two_view_recovers_points(scene):
    # Camera 0 at the world origin: express everything in camera-0 coordinates.
    R, t = scene.relative_pose(0, 1)
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = np.hstack([R, t[:, None]])
    x0 = normalize_points(scene.observations[0], scene.K)
    x1 = normalize_points(scene.observations[1], scene.K)
    X = triangulate_two_view(P0, P1, x0, x1)
    X_true = scene.points3d @ scene.rotations[0].T + scene.translations[0]
    np.testing.assert_allclose(X, X_true, atol=1e-8)


def test_triangulate_two_view_point_at_infinity_does_not_warn():
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = np.hstack([np.eye(3), np.array([[1.0], [0.0], [0.0]])])
    x = np.array([[0.1, 0.2]])
    with np.errstate(all="raise"):
        X = triangulate_two_view(P0, P1, x, x)  # identical rays: parallel, meet at infinity
    assert X.shape == (1, 3)


# --- relative pose against ground truth ---------------------------------------------------


@pytest.mark.parametrize("pair", [(0, 1), (0, 2), (1, 2)])
def test_noise_free_pose_is_exact(scene, pair):
    i, j = pair
    x_i, x_j = scene.pair(i, j)
    pose = estimate_relative_pose(x_i, x_j, scene.K, scene.K, _true_E(scene, i, j), CFG)
    R_true, t_true = scene.relative_pose(i, j)
    assert pose.success and pose.reason is None
    assert rotation_error_deg(pose.R, R_true) < 0.1
    assert translation_direction_error_deg(pose.t, t_true) < 0.1
    assert pose.cheirality_ratio == 1.0
    assert np.linalg.norm(pose.t) == pytest.approx(1.0)


def test_noisy_pose_from_estimated_essential(noisy):
    x_i, x_j = noisy.pair(0, 1)
    g = verify_pair(x_i, x_j, noisy.K, noisy.K, VerificationConfig(), model="essential")
    E = essential_for_pose(g, noisy.K, noisy.K)
    pose = estimate_relative_pose(x_i[g.inlier_mask], x_j[g.inlier_mask], noisy.K, noisy.K, E, CFG)
    R_true, t_true = noisy.relative_pose(0, 1)
    assert pose.success
    # With 1 px noise and a narrow field of view, rotation about y and sideways
    # translation are nearly confounded (bas-relief ambiguity): measured 0.5-1.3 deg
    # rotation and < 1.1 deg translation error over seeds. Bundle adjustment refines this.
    assert rotation_error_deg(pose.R, R_true) < 2.0
    assert translation_direction_error_deg(pose.t, t_true) < 2.0
    assert pose.cheirality_ratio > 0.95


def test_pose_from_fundamental_with_correct_k(noisy):
    x_i, x_j = noisy.pair(0, 1)
    g = verify_pair(x_i, x_j, noisy.K, noisy.K, VerificationConfig(), model="fundamental")
    E = essential_for_pose(g, noisy.K, noisy.K)
    pose = estimate_relative_pose(x_i[g.inlier_mask], x_j[g.inlier_mask], noisy.K, noisy.K, E, CFG)
    R_true, t_true = noisy.relative_pose(0, 1)
    assert pose.success
    # F has 7 degrees of freedom, E only 5: projecting an F estimate onto the essential
    # manifold is clearly worse than fitting E directly (measured translation errors up
    # to ~9 deg vs ~1 deg). This is why method=auto prefers E when K is known.
    assert rotation_error_deg(pose.R, R_true) < 2.0
    assert translation_direction_error_deg(pose.t, t_true) < 10.0


def test_agrees_with_opencv_recover_pose(noisy):
    x_i, x_j = noisy.pair(0, 2)
    E = _true_E(noisy, 0, 2)
    pose = estimate_relative_pose(x_i, x_j, noisy.K, noisy.K, E, CFG)
    n_i, n_j = normalize_points(x_i, noisy.K), normalize_points(x_j, noisy.K)
    _, R_cv, t_cv, _ = cv2.recoverPose(E, n_i, n_j, np.eye(3))
    assert rotation_error_deg(pose.R, R_cv) < 1e-4
    assert translation_direction_error_deg(pose.t, t_cv.ravel()) < 1e-4


def test_parallax_matches_triangulation_angle(scene):
    x_i, x_j = scene.pair(0, 1)
    pose = estimate_relative_pose(x_i, x_j, scene.K, scene.K, _true_E(scene, 0, 1), CFG)
    C0, C1 = scene.camera_center(0), scene.camera_center(1)
    a, b = scene.points3d - C0, scene.points3d - C1
    cos = np.sum(a * b, axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    expected = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    assert pose.median_parallax_deg == pytest.approx(np.median(expected), rel=1e-6)


def test_rendered_scene_end_to_end():
    from src.config import MatchConfig
    from src.features.extractor import SIFTExtractor
    from src.features.matcher import DescriptorMatcher
    from tests.render import render_box_scene

    scene = render_box_scene(n_cameras=3, arc_deg=20.0)
    extractor = SIFTExtractor(max_features=3000)
    f0, f1 = extractor.detect(scene.images[0]), extractor.detect(scene.images[1])
    m, _ = DescriptorMatcher(MatchConfig()).match(f0, f1)
    x0, x1 = m.points(f0, f1)
    g = verify_pair(x0, x1, scene.K, scene.K, VerificationConfig(), model="essential")
    E = essential_for_pose(g, scene.K, scene.K)
    pose = estimate_relative_pose(x0[g.inlier_mask], x1[g.inlier_mask], scene.K, scene.K, E, CFG)

    R_true, t_true = scene.relative_pose(0, 1)
    assert pose.success
    assert rotation_error_deg(pose.R, R_true) < 0.2
    assert translation_direction_error_deg(pose.t, t_true) < 1.0


# --- degenerate cases ------------------------------------------------------------------------


def test_rotation_only_is_rejected_as_low_parallax():
    rng = np.random.default_rng(0)
    K = intrinsics(600.0, 640, 480)
    X = rng.uniform(-2, 2, size=(400, 3)) + np.array([0, 0, 8.0])
    centre = np.zeros(3)
    R0, t0 = look_at(centre, np.array([0.0, 0.0, 1.0]))
    R1, t1 = look_at(centre, np.array([0.8, 0.1, 6.0]))  # same centre, different direction
    x0, _ = project(K, R0, t0, X)
    x1, _ = project(K, R1, t1, X)
    x0 += rng.normal(0, 0.3, x0.shape)
    x1 += rng.normal(0, 0.3, x1.shape)
    g = verify_pair(x0, x1, K, K, VerificationConfig(), model="essential")
    E = essential_for_pose(g, K, K)
    pose = estimate_relative_pose(x0, x1, K, K, E, CFG)
    assert not pose.success
    assert pose.reason == "low_parallax"
    assert pose.median_parallax_deg < 0.5
    assert rotation_error_deg(pose.R, R1 @ R0.T) < 1.0  # rotation is still recovered


def test_too_few_points(scene):
    x_i, x_j = scene.pair(0, 1)
    pose = estimate_relative_pose(x_i[:10], x_j[:10], scene.K, scene.K, _true_E(scene, 0, 1), CFG)
    assert not pose.success
    assert pose.reason == "too_few_points"
    assert pose.R is None and pose.t is None


def test_points_at_infinity_fail_cheirality(scene):
    # With |t| = 1 the scene is ~2-3 baselines deep. Treating everything beyond half a
    # baseline as "at infinity" leaves no point whose depth sign can be trusted.
    x_i, x_j = scene.pair(0, 1)
    cfg = PoseConfig(max_depth=0.5)
    pose = estimate_relative_pose(x_i, x_j, scene.K, scene.K, _true_E(scene, 0, 1), cfg)
    assert not pose.success
    assert pose.reason == "cheirality"
    assert pose.n_valid == 0


# --- initial pair selection --------------------------------------------------------------


def _stats(i, j, n_valid=500, parallax=8.0, candidate=True):
    return PoseStats(
        i=i, j=j, success=True, reason=None, n_points=n_valid, n_in_front=n_valid,
        n_valid=n_valid, cheirality_ratio=1.0, median_parallax_deg=parallax,
        rotation_deg=10.0, homography_ratio=0.3, initial_candidate=candidate,
    )  # fmt: skip


@pytest.mark.parametrize(
    ("success", "planar", "parallax", "expected"),
    [
        (True, False, 8.0, True),
        (True, False, 2.0, False),  # too little parallax
        (True, True, 8.0, False),  # explained by a homography: planar or rotation only
        (False, False, 8.0, False),  # no valid pose
    ],
)
def test_is_initial_candidate(success, planar, parallax, expected):
    from types import SimpleNamespace

    pose = SimpleNamespace(success=success, median_parallax_deg=parallax)
    assert is_initial_candidate(pose, planar, min_parallax_deg=4.0) is expected


def test_select_initial_pair_prefers_most_points_among_candidates():
    stats = [
        _stats(0, 1, n_valid=2000, candidate=False),
        _stats(1, 2, n_valid=900),
        _stats(2, 3, n_valid=1200),
        _stats(3, 4, n_valid=1200, parallax=12.0),  # tie on points: larger parallax wins
    ]
    best = select_initial_pair(stats)
    assert (best.i, best.j) == (3, 4)


def test_select_initial_pair_returns_none_without_candidates():
    assert select_initial_pair([_stats(0, 1, candidate=False)]) is None


def test_relative_pose_properties(scene):
    x_i, x_j = scene.pair(0, 1)
    pose = estimate_relative_pose(x_i, x_j, scene.K, scene.K, _true_E(scene, 0, 1), CFG)
    assert isinstance(pose, RelativePose)
    assert pose.n_points == len(x_i)
    assert pose.n_valid == int(pose.valid_mask.sum())
    assert pose.rotation_deg == pytest.approx(rotation_angle_deg(pose.R))


# --- batch and CLI on rendered images ------------------------------------------------------


@pytest.fixture(scope="module")
def rendered():
    from tests.render import render_box_scene

    return render_box_scene(n_cameras=4, arc_deg=30.0)


def test_estimate_pair_poses_over_verified_pairs(rendered):
    from types import SimpleNamespace

    from src.config import MatchConfig
    from src.features.extractor import SIFTExtractor
    from src.features.matcher import DescriptorMatcher, match_pairs, select_pairs
    from src.geometry.verification import verify_matches

    extractor = SIFTExtractor(max_features=3000)
    features = [extractor.detect(im) for im in rendered.images]
    views = [SimpleNamespace(K=rendered.K, intrinsics_source="file") for _ in rendered.images]
    matches, _ = match_pairs(
        features, select_pairs(4, MatchConfig()), DescriptorMatcher(MatchConfig())
    )
    geometries, _ = verify_matches(views, features, matches, VerificationConfig())
    poses, stats = estimate_pair_poses(views, features, matches, geometries, CFG)

    assert len(poses) == len(matches)
    assert len(stats) == sum(g.success for g in geometries)
    for m, pose in zip(matches, poses, strict=True):
        if pose is None or not pose.success:
            continue
        R_true, t_true = rendered.relative_pose(m.i, m.j)
        assert rotation_error_deg(pose.R, R_true) < 0.5
        assert translation_direction_error_deg(pose.t, t_true) < 2.0


def test_cli_pose_picks_and_plots_initial_pair(tmp_path, rendered):
    import json

    from src.cli import main
    from tests.images import write_image

    for k, im in enumerate(rendered.images):
        write_image(tmp_path / "scene" / f"view_{k + 1:02d}.png", im)
    K = rendered.K
    (tmp_path / "camera.yaml").write_text(
        f"width: 800\nheight: 600\nfx: {K[0, 0]}\nfy: {K[1, 1]}\ncx: {K[0, 2]}\ncy: {K[1, 2]}\n"
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        f"camera:\n  calibration_file: {tmp_path / 'camera.yaml'}\n"
        f"output:\n  cache_dir: {tmp_path / 'cache'}\nfeatures:\n  max_features: 3000\n"
    )
    out = tmp_path / "outputs"
    assert main(["pose", str(tmp_path / "scene"), "--config", str(config), "--out", str(out)]) == 0

    (run_dir,) = out.iterdir()
    pose_dir = run_dir / "pose"
    assert (pose_dir / "relative_poses.csv").is_file()
    assert (pose_dir / "initial_pair.png").is_file()
    info = json.loads((pose_dir / "initial_pair.json").read_text())
    assert info["model"] == "essential"  # calibrated -> auto uses E
    R_true, t_true = rendered.relative_pose(info["index_i"], info["index_j"])
    assert rotation_error_deg(np.array(info["R"]), R_true) < 0.5
    assert translation_direction_error_deg(np.array(info["t"]), t_true) < 2.0


def test_far_points_do_not_count_as_cheirality_failures():
    # Regression: at a 1.2 deg baseline ~40 % of points lie beyond max_depth (50 baselines)
    # although all are in front of both cameras. Their depth sign is unreliable, so they
    # must be excluded from the cheirality ratio, not counted as failures.
    scene = make_scene(n_points=400, n_cameras=2, noise_px=0.0, seed=0, arc_deg=1.2)
    pose = estimate_relative_pose(*scene.pair(0, 1), scene.K, scene.K, _true_E(scene, 0, 1), CFG)
    assert pose.success, pose.reason
    assert pose.cheirality_ratio == 1.0
    assert pose.n_in_front == pose.n_points  # all points are in front of both cameras
    assert pose.n_valid < pose.n_points  # but some are too far to use


def test_all_points_far_still_picks_the_right_candidate():
    # Regression: at a 0.8 deg baseline every point is beyond max_depth, so all four
    # candidates had 0 valid points and the first (a wrong, ~180 deg "twisted" rotation)
    # won the tie, giving nonsense parallax (~170 deg). Ties are broken by points in front.
    scene = make_scene(n_points=400, n_cameras=2, noise_px=0.0, seed=0, arc_deg=0.8)
    pose = estimate_relative_pose(*scene.pair(0, 1), scene.K, scene.K, _true_E(scene, 0, 1), CFG)
    R_true, _ = scene.relative_pose(0, 1)
    assert rotation_error_deg(pose.R, R_true) < 1e-3
    assert pose.reason == "low_parallax"
    assert pose.median_parallax_deg < 1.0


def test_triangulate_pose_points_of_failed_pose_is_empty(scene):
    from src.geometry.pose_estimation import triangulate_pose_points

    x_i, x_j = scene.pair(0, 1)
    pose = estimate_relative_pose(x_i[:5], x_j[:5], scene.K, scene.K, _true_E(scene, 0, 1), CFG)
    assert pose.reason == "too_few_points"
    X = triangulate_pose_points(pose, x_i[:5], x_j[:5], scene.K, scene.K)
    assert X.shape == (0, 3)

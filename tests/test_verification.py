from types import SimpleNamespace

import numpy as np
import pytest

from src.config import MatchConfig, VerificationConfig
from src.features.extractor import SIFTExtractor
from src.features.matcher import DescriptorMatcher, match_pairs, select_pairs
from src.geometry.calibration import fallback_intrinsics
from src.geometry.verification import (
    inlier_matches,
    verification_matrix,
    verify_matches,
    write_verification_statistics,
)
from src.visualization.epipolar import draw_epipolar_lines
from tests.images import make_natural_texture, make_view_sequence

pytestmark = pytest.mark.filterwarnings("error")


@pytest.fixture(scope="module")
def sequence():
    """Three overlapping views plus one unrelated image, matched exhaustively."""
    images, _ = make_view_sequence(3, size=(640, 480), seed=0)
    images.append(make_natural_texture(42))
    extractor = SIFTExtractor(max_features=2000)
    features = [extractor.detect(im) for im in images]
    pairs = select_pairs(len(images), MatchConfig())
    matches, _ = match_pairs(features, pairs, DescriptorMatcher(MatchConfig()))
    K = fallback_intrinsics(640, 480, 1.2).K
    views = [SimpleNamespace(K=K, intrinsics_source="fallback") for _ in images]
    return images, views, features, matches


def test_verify_matches_over_pairs(sequence):
    _, views, features, matches = sequence
    geometries, stats = verify_matches(views, features, matches, VerificationConfig())

    assert len(geometries) == len(stats) == len(matches)
    by_pair = {(s.i, s.j): s for s in stats}
    for pair in [(0, 1), (0, 2), (1, 2)]:
        s = by_pair[pair]
        assert s.success
        assert s.model == "fundamental"  # fallback intrinsics -> auto picks F
        assert s.inlier_ratio > 0.9
        assert s.median_residual < 1.0
        assert s.homography_ratio > 0.9  # the synthetic scene is a plane
    for pair in [(0, 3), (1, 3), (2, 3)]:
        assert not by_pair[pair].success  # the unrelated image


def test_auto_uses_essential_with_calibrated_intrinsics(sequence):
    _, views, features, matches = sequence
    calibrated = [SimpleNamespace(K=v.K, intrinsics_source="exif") for v in views]
    _, stats = verify_matches(calibrated, features, matches[:1], VerificationConfig())
    assert stats[0].model == "essential"


def test_inlier_matches_keeps_only_verified_correspondences(sequence):
    _, views, features, matches = sequence
    geometries, _ = verify_matches(views, features, matches, VerificationConfig())
    for m, g in zip(matches, geometries, strict=True):
        kept = inlier_matches(m, g)
        assert (kept.i, kept.j) == (m.i, m.j)
        if g.success:
            assert len(kept) == g.n_inliers
            np.testing.assert_array_equal(kept.idx_i, m.idx_i[g.inlier_mask])
        else:
            assert len(kept) == 0


def test_verification_matrix_and_statistics(tmp_path, sequence):
    _, views, features, matches = sequence
    _, stats = verify_matches(views, features, matches, VerificationConfig())
    M = verification_matrix(stats, 4)
    assert M.shape == (4, 4)
    np.testing.assert_array_equal(M, M.T)
    assert M[0, 1] > 0 and M[0, 3] == 0

    names = [f"im{k}.jpg" for k in range(4)]
    path = write_verification_statistics(stats, names, tmp_path / "v.csv")
    header, *rows = path.read_text().strip().splitlines()
    assert header == (
        "image_i,image_j,model,matches,inliers,inlier_ratio,median_residual_px,"
        "homography_ratio,success,reason,time_ms"
    )
    assert len(rows) == len(stats)


def test_draw_epipolar_lines(sequence):
    images, views, features, matches = sequence
    geometries, _ = verify_matches(views, features, matches[:1], VerificationConfig())
    m, g = matches[0], geometries[0]
    x_i, x_j = m.points(features[0], features[1])
    canvas = draw_epipolar_lines(images[0], images[1], x_i[g.inlier_mask], x_j[g.inlier_mask], g.F)
    assert canvas.shape == (480, 1280, 3)
    assert canvas.dtype == np.uint8


def test_cli_verify_writes_outputs(tmp_path):
    from src.cli import main
    from tests.images import write_image

    images, _ = make_view_sequence(3, size=(640, 480), seed=1)
    for k, im in enumerate(images):
        write_image(tmp_path / "scene" / f"view_{k + 1:02d}.png", im)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"output:\n  cache_dir: {tmp_path / 'cache'}\nfeatures:\n  max_features: 1500\n"
    )
    out = tmp_path / "outputs"

    args = ["verify", str(tmp_path / "scene"), "--config", str(config), "--out", str(out)]
    assert main([*args, "--draw", "1"]) == 0

    (run_dir,) = out.iterdir()
    v = run_dir / "verification"
    for name in [
        "verification_statistics.csv",
        "verified_matches.npz",
        "verified_matrix.png",
        "view_01_view_02_inliers.png",
        "view_01_view_02_epipolar.png",
    ]:
        assert (v / name).is_file(), name
    assert (run_dir / "matches" / "match_statistics.csv").is_file()


def test_verification_agrees_with_true_geometry_on_rendered_3d_scene():
    from src.geometry.epipolar import (
        fundamental_from_essential,
        sampson_distance,
        skew,
        verify_pair,
    )
    from tests.render import render_box_scene

    scene = render_box_scene(n_cameras=3, arc_deg=20.0)
    extractor = SIFTExtractor(max_features=3000)
    f0, f1 = extractor.detect(scene.images[0]), extractor.detect(scene.images[1])
    m, _ = DescriptorMatcher(MatchConfig()).match(f0, f1)
    x0, x1 = m.points(f0, f1)

    R, t = scene.relative_pose(0, 1)
    true_residual = sampson_distance(
        fundamental_from_essential(skew(t) @ R, scene.K, scene.K), x0, x1
    )
    cfg = VerificationConfig()
    for model in ("essential", "fundamental"):
        g = verify_pair(x0, x1, scene.K, scene.K, cfg, model=model)
        assert g.success
        assert np.mean(true_residual[g.inlier_mask] <= cfg.ransac_threshold_px) > 0.99
        assert np.mean(g.inlier_mask[true_residual <= 0.5]) > 0.99
        assert g.homography_ratio < 0.9  # a box on a plane is not one plane


def test_parallel_verification_equals_sequential(sequence):
    _, views, features, matches = sequence
    seq, _ = verify_matches(views, features, matches, VerificationConfig(), workers=1)
    par, _ = verify_matches(views, features, matches, VerificationConfig(), workers=4)
    for a, b in zip(seq, par, strict=True):
        np.testing.assert_array_equal(a.inlier_mask, b.inlier_mask)
        assert a.success == b.success and a.homography_inliers == b.homography_inliers


def test_unverified_images_and_planar_flag(sequence):
    from src.geometry.verification import unverified_images

    _, views, features, matches = sequence
    geometries, stats = verify_matches(views, features, matches, VerificationConfig())
    assert unverified_images(stats, 4) == [3]  # the unrelated image
    # The synthetic sequence is a plane: verification marks its pairs planar.
    assert all(g.planar for g in geometries if g.success)
    assert not any(g.planar for g in geometries if not g.success)

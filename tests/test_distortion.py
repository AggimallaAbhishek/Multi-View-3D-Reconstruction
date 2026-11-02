"""Radial lens distortion estimated by bundle adjustment, against synthetic ground truth."""

import numpy as np
import pytest

from src.config import BundleAdjustmentConfig
from src.geometry.camera import Camera
from src.geometry.triangulation import Observations
from src.reconstruction.distortion import RadialDistortion, estimate_radial_distortion
from tests.synthetic import make_scene

CONFIG = BundleAdjustmentConfig()
TRUE = RadialDistortion(-0.12, 0.04)  # barrel, a few per cent at the corners


def _distorted_problem(distortion: RadialDistortion, noise_px=0.3, seed=0):
    """A wide-angle scene (corners at ~1.1 normalised radius, the points filling the
    frame) whose observations are distorted, with the true cameras and slightly
    perturbed points as the start."""
    s = make_scene(n_points=600, n_cameras=8, focal=350.0, radius=2.4, arc_deg=80.0,
                   elevation=0.3, seed=seed)  # fmt: skip
    rng = np.random.default_rng(seed)
    K = s.K
    view, point = np.nonzero(s.visible)
    xn = (s.observations[view, point] - K[:2, 2]) / K[0, 0]
    xy = distortion.apply(xn) * K[0, 0] + K[:2, 2] + rng.normal(0, noise_px, xn.shape)
    cameras = [Camera(K, R, t) for R, t in zip(s.rotations, s.translations, strict=True)]
    points = s.points3d + rng.normal(0, 0.01, s.points3d.shape)
    return cameras, Observations(view, point, xy), points, s


def test_estimate_recovers_radial_distortion():
    cameras, obs, points, s = _distorted_problem(TRUE)
    est = estimate_radial_distortion(cameras, obs, points, s.image_size, CONFIG)
    # The curve, not the coefficients, is what is observable, and only where there are
    # points: compare displacements up to the largest observed radius.
    observed = np.linalg.norm((obs.xy - s.K[:2, 2]) / s.K[0, 0], axis=1).max()
    assert observed > 0.8
    r = np.linspace(0.05, observed, 11)[:, None] * np.array([[1.0, 0.0]])
    np.testing.assert_allclose(est.distortion.apply(r), TRUE.apply(r), atol=2e-3)
    assert est.final_rms_px < 0.5 < est.initial_rms_px
    assert est.focal_scale == pytest.approx(1.0, abs=0.01)


def test_estimate_on_undistorted_data_is_negligible():
    cameras, obs, points, s = _distorted_problem(RadialDistortion(0.0, 0.0))
    est = estimate_radial_distortion(cameras, obs, points, s.image_size, CONFIG)
    # The corner lies beyond the farthest observation (noise extrapolated): ~1 px here,
    # against ~35 px for TRUE, and below the pipeline's distortion_min_shift_px (2).
    assert est.distortion.corner_shift_px(s.K, s.image_size) < 1.5
    assert TRUE.corner_shift_px(s.K, s.image_size) > 30


def test_corner_shift_is_the_displacement_of_the_farthest_corner():
    K = np.array([[500.0, 0, 300], [0, 500, 200], [0, 0, 1]])
    d = RadialDistortion(-0.1, 0.0)
    # farthest corner from (300, 200) in a 640 x 480 image: (639, 479), r = |(339, 279)| / 500
    r = np.hypot(339, 279) / 500
    assert d.corner_shift_px(K, (640, 480)) == pytest.approx(500 * r * 0.1 * r**2)


def test_distortion_coefficients_for_opencv():
    np.testing.assert_array_equal(RadialDistortion(0.1, -0.2).coefficients, [0.1, -0.2, 0, 0, 0])


# --- in the pipeline: estimate, undistort, reconstruct again ---------------------------


@pytest.fixture(scope="module")
def scenes(tmp_path_factory):
    """Rendered wide-angle views (corners at 1.19 normalised radius) through a lens with
    the given distortion, rendered once per module; the runs share one feature cache."""
    from tests.render import distort_image, render_box_scene, write_scene

    root = tmp_path_factory.mktemp("distortion")
    made = {}

    def scene_dir(dist):
        key = tuple(dist)
        if key not in made:
            scene = render_box_scene(n_cameras=10, size=(800, 600), focal=420.0, radius=3.2,
                                     elevation=2.0, arc_deg=90.0)  # fmt: skip
            images = [distort_image(im, scene.K, dist) for im in scene.images]
            scene = type(scene)(scene.K, scene.rotations, scene.translations, images,
                                scene.image_size)  # fmt: skip
            folder, camera = write_scene(scene, root / f"lens{len(made)}")  # K only, no dist
            made[key] = scene, folder, camera
        return made[key]

    scene_dir.cache = str(root / "cache")
    return scene_dir


def _config(scenes, camera, **sections):
    from src.config import config_from_dict

    return config_from_dict({"camera": {"calibration_file": str(camera)},
                             "output": {"cache_dir": scenes.cache}, **sections})  # fmt: skip


def _pose_errors(result, scene):
    from src.evaluation.metrics import camera_pose_errors
    from tests.render import scene_cameras

    truth, rec = scene_cameras(scene), result.reconstruction
    images = rec.registry.images
    return camera_pose_errors([rec.registry.camera(m) for m in images],
                              [truth[result.names[m]] for m in images])  # fmt: skip


@pytest.mark.parametrize("estimate", [False, True])
def test_pipeline_self_calibrates_radial_distortion(scenes, estimate):
    """A wide-angle lens with ~56 px of barrel distortion at the corners and a camera
    file without it: estimating the distortion and reconstructing again from undistorted
    images brings the poses back to the distortion-free accuracy (~0.05 deg)."""
    from src.pipeline import run_pipeline

    scene, folder, camera = scenes([-0.15, 0.05, 0, 0, 0])
    config = _config(scenes, camera, bundle_adjustment={"estimate_distortion": estimate})
    result = run_pipeline(folder, config, until="reconstruct")
    errors = _pose_errors(result, scene)
    assert len(errors.rotation_deg) == 10
    if not estimate:
        assert result.distortion is None
        assert np.median(errors.rotation_deg) > 0.4
        return
    d = result.distortion
    assert d.passes >= 1
    assert d.distortion.k1 == pytest.approx(-0.15, abs=0.01)
    assert d.distortion.k2 == pytest.approx(0.05, abs=0.02)
    assert np.median(errors.rotation_deg) < 0.12
    assert d.estimates[-1].shift_px < config.bundle_adjustment.distortion_min_shift_px


def test_pipeline_does_not_reconstruct_again_without_distortion(scenes):
    from src.pipeline import run_pipeline

    scene, folder, camera = scenes([0, 0, 0, 0, 0])
    config = _config(scenes, camera)
    result = run_pipeline(folder, config, until="reconstruct")
    assert result.distortion.passes == 0
    assert len(result.distortion.estimates) == 1
    assert result.distortion.estimates[0].shift_px < 1.0


def test_pipeline_does_not_self_calibrate_from_too_few_cameras(scenes):
    """Two registered views cannot pin a distortion down (a spurious one once undistorted
    the images so badly that nothing reconstructed): no estimate below
    ``distortion_min_images``."""
    from src.pipeline import run_pipeline

    scene, folder, camera = scenes([0, 0, 0, 0, 0])
    config = _config(scenes, camera, bundle_adjustment={"distortion_min_images": 11})
    result = run_pipeline(folder, config, until="reconstruct")
    assert result.reconstruction is not None and result.distortion is None


def test_pipeline_discards_a_worse_pass(scenes, monkeypatch):
    """A wrong estimate (injected: strong pincushion on distortion-free images) undistorts
    the images into a worse reconstruction (same images, 4x the reprojection error): that
    pass is discarded, the first kept."""
    import src.pipeline as pipeline
    from src.reconstruction.distortion import DistortionEstimate

    scene, folder, camera = scenes([0, 0, 0, 0, 0])
    config = _config(scenes, camera)
    wrong = DistortionEstimate(RadialDistortion(0.6, 0.3), 1.0, 1.0, 1.0, shift_px=1000.0)
    monkeypatch.setattr(pipeline, "estimate_radial_distortion", lambda *a, **k: wrong)
    prepared = []
    original = pipeline.prepare_dataset
    monkeypatch.setattr(pipeline, "prepare_dataset",
                        lambda *a, **k: prepared.append(a) or original(*a, **k))  # fmt: skip
    result = pipeline.run_pipeline(folder, config, until="reconstruct")
    assert len(prepared) == 2  # the wrong pass really ran (and was discarded)
    assert len(result.reconstruction.registry.images) == 10
    assert result.distortion.passes == 0
    assert result.distortion.distortion == RadialDistortion(0.0, 0.0)
    assert np.median(_pose_errors(result, scene).rotation_deg) < 0.1


def test_pipeline_does_not_self_calibrate_a_narrow_field_of_view(scenes):
    """Distortion is unobservable over a narrow view (Middlebury dino, corners at 0.12
    normalised radius: a spurious k2 of -156 made the poses worse). No estimate when
    the farthest corner is nearer the axis than ``distortion_min_field_radius``."""
    from src.pipeline import run_pipeline

    scene, folder, camera = scenes([0, 0, 0, 0, 0])  # corners at 1.19
    config = _config(scenes, camera, bundle_adjustment={"distortion_min_field_radius": 1.2})
    result = run_pipeline(folder, config, until="reconstruct")
    assert result.reconstruction is not None and result.distortion is None


def test_pipeline_does_not_self_calibrate_without_undistortion(scenes):
    """With ``preprocessing.undistort`` off the images cannot be undistorted, so nothing
    may be estimated and reported as applied."""
    from src.pipeline import run_pipeline

    scene, folder, camera = scenes([-0.15, 0.05, 0, 0, 0])
    config = _config(scenes, camera, preprocessing={"undistort": False})
    result = run_pipeline(folder, config, until="reconstruct")
    assert result.reconstruction is not None and result.distortion is None

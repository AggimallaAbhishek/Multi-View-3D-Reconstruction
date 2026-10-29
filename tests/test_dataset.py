import dataclasses
import json
import logging

import numpy as np
import pytest

from src.cli import main
from src.config import config_from_dict
from src.data.dataset import prepare_dataset
from src.data.loader import DatasetError
from src.geometry.calibration import fallback_intrinsics, save_camera_file
from tests.images import write_textured_images

CFG = config_from_dict({"preprocessing": {"max_side": 320}})


@pytest.fixture
def image_dir(tmp_path):
    directory = tmp_path / "scene"
    write_textured_images(directory, 4, size=(640, 480))
    (directory / "image_005.jpg").write_bytes(b"corrupt")
    return directory


def test_prepare_dataset_loads_resizes_and_scales_k(image_dir):
    dataset = prepare_dataset(image_dir, CFG)

    assert [im.name for im in dataset.images] == [f"image_{i:03d}.jpg" for i in range(1, 5)]
    assert [im.index for im in dataset.images] == [0, 1, 2, 3]
    for im in dataset.images:
        assert im.image.shape == (240, 320, 3)
        assert im.original_size == (640, 480)
        assert im.scale == (0.5, 0.5)
        assert im.intrinsics_source == "fallback"
        np.testing.assert_allclose(im.K[0, 0], 1.2 * 640 * 0.5)
        assert im.gray.shape == (240, 320)

    assert dataset.report.ok
    assert [i.code for i in dataset.report.issues] == ["unreadable"]


def test_prepare_dataset_is_deterministic(image_dir):
    a = prepare_dataset(image_dir, CFG)
    b = prepare_dataset(image_dir, CFG)
    assert [im.name for im in a.images] == [im.name for im in b.images]
    for x, y in zip(a.images, b.images, strict=True):
        np.testing.assert_array_equal(x.image, y.image)


def test_prepare_dataset_empty_directory(tmp_path):
    with pytest.raises(DatasetError, match="no images"):
        prepare_dataset(tmp_path, CFG)


def test_prepare_dataset_too_few_images_attaches_report(tmp_path):
    write_textured_images(tmp_path, 1)
    with pytest.raises(DatasetError) as excinfo:
        prepare_dataset(tmp_path, CFG)
    assert excinfo.value.report is not None
    assert "too_few_images" in [i.code for i in excinfo.value.report.issues]


def test_prepare_dataset_uses_calibration_file(image_dir, tmp_path):
    camera_file = save_camera_file(fallback_intrinsics(1280, 960, 2.0), tmp_path / "camera.yaml")
    cfg = config_from_dict(
        {"preprocessing": {"max_side": 320}, "camera": {"calibration_file": str(camera_file)}}
    )
    dataset = prepare_dataset(image_dir, cfg)
    im = dataset.images[0]
    assert im.intrinsics_source == "file"
    # 2.0 * 1280 at calibration size -> x0.5 to 640 -> x0.5 to 320.
    np.testing.assert_allclose(im.K[0, 0], 640.0)


def test_ignored_distortion_is_reported(image_dir, tmp_path, caplog):
    distorted = dataclasses.replace(
        fallback_intrinsics(640, 480, 2.0), dist=np.array([-0.2, 0.0, 0.0, 0.0, 0.0])
    )
    camera_file = save_camera_file(distorted, tmp_path / "camera.yaml")
    cfg = config_from_dict(
        {
            "preprocessing": {"max_side": 320, "undistort": False},
            "camera": {"calibration_file": str(camera_file)},
        }
    )
    with caplog.at_level(logging.WARNING, logger="src.data.dataset"):
        prepare_dataset(image_dir, cfg)
    assert "undistort is off" in caplog.text


def test_dataset_save_writes_json(image_dir, tmp_path):
    dataset = prepare_dataset(image_dir, CFG)
    path = dataset.save(tmp_path / "out" / "dataset.json")
    data = json.loads(path.read_text())
    assert data["n_images"] == 4
    assert data["images"][0]["name"] == "image_001.jpg"
    assert np.array(data["images"][0]["K"]).shape == (3, 3)
    assert data["report"]["ok"] is True


def test_cli_validate_writes_run_directory(image_dir, tmp_path):
    out = tmp_path / "outputs"
    assert main(["validate", str(image_dir), "--out", str(out)]) == 0
    (run_dir,) = out.glob("*_validate")  # outputs/ also holds the default cache
    assert (run_dir / "dataset.json").is_file()
    assert (run_dir / "config.yaml").is_file()
    env = json.loads((run_dir / "environment.json").read_text())
    assert "opencv" in env["packages"]


def test_cli_validate_fails_on_invalid_dataset(tmp_path, capsys):
    write_textured_images(tmp_path / "one", 1)
    assert main(["validate", str(tmp_path / "one"), "--out", str(tmp_path / "o")]) == 1
    assert "too_few_images" in capsys.readouterr().out


def test_image_cache_skips_decoding_on_repeat(image_dir, tmp_path, monkeypatch):
    import src.data.dataset as dataset_module

    cfg = config_from_dict(
        {"preprocessing": {"max_side": 320}, "output": {"cache_dir": str(tmp_path / "c")}}
    )
    first = prepare_dataset(image_dir, cfg)

    def no_decoding(_):
        raise AssertionError("decoded although cached")

    monkeypatch.setattr(dataset_module, "decode_image", no_decoding)
    corrupt = image_dir / "image_005.jpg"
    corrupt.unlink()  # uncached (undecodable) file: remove so nothing needs decoding
    second = prepare_dataset(image_dir, cfg)
    assert [im.name for im in second.images] == [im.name for im in first.images]
    for a, b in zip(first.images, second.images, strict=True):
        np.testing.assert_array_equal(a.gray, b.gray)
        np.testing.assert_array_equal(a.K, b.K)
        assert (a.scale, a.intrinsics_source) == (b.scale, b.intrinsics_source)
    assert [s.blur for s in first.report.images] == [s.blur for s in second.report.images]


def test_image_cache_key_follows_settings(image_dir, tmp_path):
    root = str(tmp_path / "c")
    prepare_dataset(image_dir, config_from_dict({"output": {"cache_dir": root}}))
    small = prepare_dataset(
        image_dir,
        config_from_dict({"output": {"cache_dir": root}, "preprocessing": {"max_side": 200}}),
    )
    assert small.images[0].width == 200  # not served from the full-size cache entry


def test_colour_image_is_rebuilt_on_demand(image_dir):
    import cv2

    from src.data.loader import read_image

    im = prepare_dataset(image_dir, CFG).images[0]
    assert not any(isinstance(v, np.ndarray) and v.ndim == 3 for v in vars(im).values())
    expected = cv2.resize(read_image(im.path), (320, 240), interpolation=cv2.INTER_AREA)
    np.testing.assert_array_equal(im.image, expected)
    np.testing.assert_array_equal(cv2.cvtColor(im.image, cv2.COLOR_BGR2GRAY), im.gray)


def test_camera_yaml_next_to_images_is_used(image_dir, tmp_path):
    # Library behaviour, so the CLI, scripts and notebooks calibrate identically.
    save_camera_file(fallback_intrinsics(640, 480, 2.0), image_dir / "camera.yaml")
    dataset = prepare_dataset(image_dir, CFG)
    assert dataset.calibration_file == image_dir / "camera.yaml"
    assert {im.intrinsics_source for im in dataset.images} == {"file"}
    np.testing.assert_allclose(dataset.images[0].K[0, 0], 2.0 * 640 * 0.5)


def test_configured_calibration_file_wins(image_dir, tmp_path):
    save_camera_file(fallback_intrinsics(640, 480, 2.0), image_dir / "camera.yaml")
    other = save_camera_file(fallback_intrinsics(640, 480, 3.0), tmp_path / "other.yaml")
    cfg = config_from_dict(
        {"preprocessing": {"max_side": 320}, "camera": {"calibration_file": str(other)}}
    )
    dataset = prepare_dataset(image_dir, cfg)
    assert dataset.calibration_file == other
    np.testing.assert_allclose(dataset.images[0].K[0, 0], 3.0 * 640 * 0.5)

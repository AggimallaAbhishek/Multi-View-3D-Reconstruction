import json
from pathlib import Path

import pytest

from src.config import DatasetConfig
from src.data.loader import SkippedFile
from src.data.validation import (
    ImageStats,
    blur_score,
    dhash,
    hash_distance,
    image_stats,
    validate,
)
from tests.images import blur, make_texture

CFG = DatasetConfig(min_images=2)


def _stats(name, seed=0, size=(640, 480), blurred=False):
    image = make_texture(seed, *size)
    if blurred:
        image = blur(image)
    return ImageStats(
        name=name, width=size[0], height=size[1], blur=blur_score(image), dhash=dhash(image)
    )


def _codes(report):
    return [issue.code for issue in report.issues]


def test_blur_score_separates_sharp_from_blurred():
    sharp = make_texture(0)
    assert blur_score(sharp) > CFG.blur_threshold > blur_score(blur(sharp))


def test_blur_score_is_roughly_resolution_independent():
    # Scores are computed at a capped working size so thresholds transfer across cameras.
    small = blur_score(make_texture(0, 800, 600, block=8))
    large = blur_score(make_texture(0, 3200, 2400, block=32))
    assert large == pytest.approx(small, rel=0.35)


def test_dhash_identical_and_distinct_images():
    a, b = make_texture(0), make_texture(1)
    assert hash_distance(dhash(a), dhash(a.copy())) == 0.0
    assert hash_distance(dhash(a), dhash(b)) > 0.3  # unrelated: about half the bits differ


def _object_on_black(angle_deg: float, shift_px: int = 0):
    # Like the Middlebury temple: a textured object filling ~1/3 of a black frame.
    import cv2
    import numpy as np

    from tests.images import make_natural_texture

    image = make_natural_texture(3)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.ellipse(mask, (320, 240), (130, 170), 0, 0, 360, 255, -1)
    image[mask == 0] = 0
    A = cv2.getRotationMatrix2D((320.0, 240.0), angle_deg, 1.0)
    A[:, 2] += shift_px
    return cv2.warpAffine(image, A, (640, 480))


def test_neighbouring_views_of_object_on_plain_background_are_not_duplicates():
    # Regression (Middlebury templeRing): with a 64-bit dHash, 68 % black background made
    # neighbouring views 7.5 deg apart look identical (1-2 bits), flagging 25 false pairs.
    base, neighbour = _object_on_black(0.0), _object_on_black(6.0)
    assert hash_distance(dhash(base), dhash(neighbour)) > CFG.duplicate_max_fraction


def test_shaken_copy_is_still_a_duplicate():
    base, shaken = _object_on_black(0.0), _object_on_black(0.0, shift_px=2)
    assert hash_distance(dhash(base), dhash(shaken)) <= CFG.duplicate_max_fraction


def test_clean_dataset_has_no_issues():
    report = validate([_stats("a", 0), _stats("b", 1), _stats("c", 2)], [], CFG)
    assert report.ok
    assert report.issues == []


def test_too_few_images_is_an_error():
    report = validate([_stats("a")], [], CFG)
    assert not report.ok
    assert "too_few_images" in _codes(report)


def test_unreadable_files_are_warnings():
    skipped = [SkippedFile(Path("x.jpg"), "could not decode image")]
    report = validate([_stats("a", 0), _stats("b", 1)], skipped, CFG)
    assert report.ok
    assert "unreadable" in _codes(report)


def test_mixed_resolutions_are_a_warning():
    report = validate([_stats("a", 0), _stats("b", 1, size=(480, 640))], [], CFG)
    assert report.ok
    assert "mixed_resolution" in _codes(report)


def _with_blur(scores):
    return [
        ImageStats(name=f"im{k}", width=640, height=480, blur=b, dhash=k)
        for k, b in enumerate(scores)
    ]


def test_sharp_low_texture_images_are_not_blurry():
    # Regression (Middlebury templeRing): smooth plaster scores 79-187, below the old
    # absolute threshold of 100, although every image is sharp.
    report = validate(_with_blur([79, 83, 145, 160, 187]), [], CFG)
    assert "blurry" not in _codes(report)


def test_blurry_outlier_is_flagged_relative_to_dataset():
    report = validate(_with_blur([40, 150, 160, 170]), [], CFG)  # 40 < 0.35 * median
    assert [i.message.split(":")[0] for i in report.issues if i.code == "blurry"] == ["im0"]


def test_uniformly_blurry_dataset_is_flagged_by_absolute_floor():
    report = validate(_with_blur([5, 6, 7]), [], CFG)
    assert sum(i.code == "blurry" for i in report.issues) == 3


def test_blurry_image_is_named_in_warning():
    report = validate([_stats("sharp", 0), _stats("soft", 1, blurred=True)], [], CFG)
    blurry = [i for i in report.issues if i.code == "blurry"]
    assert len(blurry) == 1
    assert "soft" in blurry[0].message
    assert blurry[0].severity == "warning"


def test_near_duplicates_are_reported():
    report = validate([_stats("a", 0), _stats("a_copy", 0), _stats("b", 1)], [], CFG)
    dupes = [i for i in report.issues if i.code == "near_duplicate"]
    assert len(dupes) == 1
    assert "a" in dupes[0].message and "a_copy" in dupes[0].message


def test_image_stats_from_record(tmp_path):
    from src.data.loader import iter_images
    from tests.images import write_image

    path = write_image(tmp_path / "a.png", make_texture(0, 64, 48))
    (record,) = iter_images([path])
    stats = image_stats(record)
    assert (stats.name, stats.width, stats.height) == ("a.png", 64, 48)
    assert stats.blur > 0


def test_report_is_json_serialisable():
    skipped = [SkippedFile(Path("x.jpg"), "could not decode image")]
    report = validate([_stats("a", 0), _stats("b", 1, blurred=True)], skipped, CFG)
    data = json.loads(json.dumps(report.to_dict()))
    assert data["n_images"] == 2
    assert data["n_skipped"] == 1
    assert data["resolutions"] == {"640x480": 2}
    assert {i["code"] for i in data["issues"]} == {"unreadable", "blurry"}

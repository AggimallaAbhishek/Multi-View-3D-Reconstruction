import pytest

from src.data.loader import (
    DatasetError,
    ImageRecord,
    SkippedFile,
    discover_images,
    iter_images,
    read_exif,
    read_image,
)
from tests.images import make_texture, write_image

EXTENSIONS = (".jpg", ".jpeg", ".png")


def _touch_images(directory, names):
    for name in names:
        write_image(directory / name, make_texture(0, 64, 48))


def test_discover_uses_natural_order(tmp_path):
    _touch_images(tmp_path, ["img10.jpg", "img2.jpg", "img1.jpg"])
    names = [p.name for p in discover_images(tmp_path, EXTENSIONS)]
    assert names == ["img1.jpg", "img2.jpg", "img10.jpg"]


def test_discover_matches_extensions_case_insensitively(tmp_path):
    _touch_images(tmp_path, ["a.JPG", "b.png"])
    (tmp_path / "notes.txt").write_text("not an image")
    assert [p.name for p in discover_images(tmp_path, EXTENSIONS)] == ["a.JPG", "b.png"]


def test_discover_ignores_hidden_files_and_subdirectories(tmp_path):
    _touch_images(tmp_path, ["a.jpg", "._a.jpg", "sub/b.jpg"])
    assert [p.name for p in discover_images(tmp_path, EXTENSIONS)] == ["a.jpg"]


def test_discover_missing_directory_raises(tmp_path):
    with pytest.raises(DatasetError, match="not a directory"):
        discover_images(tmp_path / "nope", EXTENSIONS)


def test_discover_empty_directory_returns_nothing(tmp_path):
    assert discover_images(tmp_path, EXTENSIONS) == []


def test_read_image_returns_none_for_corrupt_file(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"definitely not a jpeg")
    assert read_image(bad) is None


def test_read_image_handles_non_ascii_path(tmp_path):
    path = write_image(tmp_path / "statue_ü_日本.png", make_texture(1, 64, 48))
    image = read_image(path)
    assert image is not None
    assert image.shape == (48, 64, 3)


def test_iter_images_skips_corrupt_files_and_numbers_loaded_ones(tmp_path):
    _touch_images(tmp_path, ["a.jpg", "c.jpg"])
    (tmp_path / "b.jpg").write_bytes(b"\x00\x01")
    items = list(iter_images(discover_images(tmp_path, EXTENSIONS)))

    records = [i for i in items if isinstance(i, ImageRecord)]
    skipped = [i for i in items if isinstance(i, SkippedFile)]
    assert [r.name for r in records] == ["a.jpg", "c.jpg"]
    assert [r.index for r in records] == [0, 1]
    assert [s.path.name for s in skipped] == ["b.jpg"]
    assert skipped[0].reason


def test_record_reports_dimensions(tmp_path):
    write_image(tmp_path / "a.png", make_texture(0, 80, 40))
    (record,) = iter_images([tmp_path / "a.png"])
    assert (record.width, record.height) == (80, 40)
    assert record.image.dtype.name == "uint8"
    assert record.image.ndim == 3


def test_exif_orientation_is_applied(tmp_path):
    # Orientation 6 = rotate 90 degrees clockwise to display upright.
    path = write_image(tmp_path / "rot.jpg", make_texture(0, 80, 40), orientation=6)
    (record,) = iter_images([path])
    assert (record.width, record.height) == (40, 80)


def test_read_exif_focal_lengths(tmp_path):
    path = write_image(tmp_path / "e.jpg", make_texture(0, 64, 48), focal_35mm=26, focal_mm=4.25)
    exif = read_exif(path)
    assert exif.focal_35mm == 26
    assert exif.focal_mm == pytest.approx(4.25)


def test_read_exif_missing_tags_are_none(tmp_path):
    path = write_image(tmp_path / "plain.png", make_texture(0, 64, 48))
    exif = read_exif(path)
    assert exif.focal_35mm is None
    assert exif.focal_mm is None


def test_read_exif_tolerates_corrupt_file(tmp_path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"junk")
    assert read_exif(bad).focal_35mm is None

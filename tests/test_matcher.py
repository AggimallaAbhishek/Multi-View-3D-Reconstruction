import cv2
import numpy as np
import pytest

from src.config import MatchConfig
from src.features.extractor import ORBExtractor, SIFTExtractor
from src.features.matcher import (
    DescriptorMatcher,
    Matches,
    connectivity,
    load_matches,
    match_pairs,
    save_matches,
    select_pairs,
    write_match_statistics,
)
from src.visualization.matches import draw_matches
from tests.images import make_natural_texture, make_view_sequence, relative_homography

pytestmark = pytest.mark.filterwarnings("error")


@pytest.fixture(scope="module")
def views():
    images, homographies = make_view_sequence(4, size=(640, 480), seed=0)
    return images, homographies


@pytest.fixture(scope="module")
def sift_features(views):
    extractor = SIFTExtractor(max_features=3000)
    return [extractor.detect(im) for im in views[0]]


def _precision(matches, fi, fj, H, tol=3.0):
    pi, pj = matches.points(fi, fj)
    projected = cv2.perspectiveTransform(pi[None], H)[0]
    return float(np.mean(np.linalg.norm(projected - pj, axis=1) < tol))


# --- correctness against ground truth ----------------------------------------------------


@pytest.mark.parametrize("j", [1, 3])
def test_sift_matches_agree_with_true_homography(views, sift_features, j):
    matcher = DescriptorMatcher(MatchConfig(ratio=0.75, mutual=True))
    m, stats = matcher.match(sift_features[0], sift_features[j], 0, j)
    assert len(m) > 300
    assert (
        _precision(m, sift_features[0], sift_features[j], relative_homography(views[1], 0, j))
        > 0.95
    )
    assert (stats.i, stats.j, stats.final) == (0, j, len(m))


def test_orb_matches_agree_with_true_homography(views):
    extractor = ORBExtractor(max_features=3000)
    f0, f1 = extractor.detect(views[0][0]), extractor.detect(views[0][1])
    m, _ = DescriptorMatcher(MatchConfig()).match(f0, f1)
    assert len(m) > 200
    assert _precision(m, f0, f1, relative_homography(views[1], 0, 1)) > 0.9


def test_unrelated_images_give_few_matches():
    # The ratio test exists to reject ambiguous matches; unrelated scenes should barely match.
    extractor = SIFTExtractor(max_features=3000)
    fa = extractor.detect(make_natural_texture(1))
    fb = extractor.detect(make_natural_texture(2))
    m, stats = DescriptorMatcher(MatchConfig(ratio=0.75, mutual=True)).match(fa, fb)
    assert stats.raw > 1000  # every feature has a nearest neighbour...
    assert len(m) < 30  # ...but almost none survive ratio + mutual checks


# --- invariants ---------------------------------------------------------------------------


def test_indices_valid_and_one_to_one_with_mutual_check(sift_features):
    fi, fj = sift_features[0], sift_features[1]
    m, _ = DescriptorMatcher(MatchConfig(mutual=True)).match(fi, fj)
    assert m.idx_i.min() >= 0 and m.idx_i.max() < len(fi)
    assert m.idx_j.min() >= 0 and m.idx_j.max() < len(fj)
    assert len(np.unique(m.idx_i)) == len(m) == len(np.unique(m.idx_j))
    assert m.idx_i.dtype == np.int32 and m.distances.dtype == np.float32
    assert np.all(m.distances >= 0)
    assert np.all(np.diff(m.idx_i) > 0)  # sorted by query index


def test_ratio_test_is_monotonic(sift_features):
    counts = [
        len(DescriptorMatcher(MatchConfig(ratio=r, mutual=False)).match(*sift_features[:2])[0])
        for r in (0.5, 0.6, 0.7, 0.8, 0.9)
    ]
    assert counts == sorted(counts)
    assert counts[0] < counts[-1]


def test_mutual_check_keeps_a_subset(sift_features):
    fi, fj = sift_features[0], sift_features[1]
    loose, s1 = DescriptorMatcher(MatchConfig(mutual=False)).match(fi, fj)
    strict, s2 = DescriptorMatcher(MatchConfig(mutual=True)).match(fi, fj)
    assert len(strict) <= len(loose)
    assert set(zip(strict.idx_i, strict.idx_j, strict=True)) <= set(
        zip(loose.idx_i, loose.idx_j, strict=True)
    )
    assert s1.after_ratio == s2.after_ratio
    assert s2.final <= s2.after_ratio <= s2.raw


def test_matching_an_image_with_itself_is_identity(sift_features):
    f = sift_features[0]
    m, _ = DescriptorMatcher(MatchConfig(mutual=True)).match(f, f)
    assert np.mean(m.idx_i == m.idx_j) > 0.99
    assert len(m) > 0.9 * len(f)


def test_bruteforce_is_deterministic(sift_features):
    matcher = DescriptorMatcher(MatchConfig())
    a, _ = matcher.match(*sift_features[:2])
    b, _ = DescriptorMatcher(MatchConfig()).match(*sift_features[:2])
    np.testing.assert_array_equal(a.idx_i, b.idx_i)
    np.testing.assert_array_equal(a.idx_j, b.idx_j)


def test_flann_is_seeded_and_close_to_bruteforce(sift_features):
    flann = DescriptorMatcher(MatchConfig(method="flann"))
    a, _ = flann.match(*sift_features[:2])
    b, _ = DescriptorMatcher(MatchConfig(method="flann")).match(*sift_features[:2])
    np.testing.assert_array_equal(a.idx_j, b.idx_j)
    exact, _ = DescriptorMatcher(MatchConfig(method="bruteforce")).match(*sift_features[:2])
    assert len(a) == pytest.approx(len(exact), rel=0.05)


@pytest.mark.parametrize("sizes", [(0, 100), (100, 0), (0, 0), (100, 1), (1, 100)])
def test_too_few_features_give_empty_matches(sift_features, sizes):
    f = sift_features[0]
    fi, fj = f.subset(np.arange(sizes[0])), f.subset(np.arange(sizes[1]))
    for mutual in (True, False):
        m, stats = DescriptorMatcher(MatchConfig(mutual=mutual)).match(fi, fj)
        if min(sizes) == 0 or sizes[1] == 1:
            # No neighbours, or no second neighbour for the ratio test.
            assert len(m) == 0
        assert stats.final == len(m)


def test_mixing_descriptor_types_raises(views, sift_features):
    orb = ORBExtractor(max_features=500).detect(views[0][0])
    with pytest.raises(ValueError, match="detector"):
        DescriptorMatcher(MatchConfig()).match(sift_features[0], orb)


# --- pairs, batches, persistence ----------------------------------------------------------


def test_select_pairs_exhaustive():
    pairs = select_pairs(5, MatchConfig(pair_mode="exhaustive"))
    assert len(pairs) == 10
    assert all(i < j for i, j in pairs)


def test_select_pairs_sequential():
    pairs = select_pairs(6, MatchConfig(pair_mode="sequential", sequential_window=2))
    assert pairs == [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4), (3, 5), (4, 5)]
    wide = select_pairs(6, MatchConfig(pair_mode="sequential", sequential_window=10))
    assert wide == select_pairs(6, MatchConfig(pair_mode="exhaustive"))


def test_match_pairs_and_statistics(tmp_path, sift_features):
    pairs = select_pairs(len(sift_features), MatchConfig())
    matches, stats = match_pairs(sift_features, pairs, DescriptorMatcher(MatchConfig()))
    assert [(m.i, m.j) for m in matches] == pairs
    assert all(s.final == len(m) for s, m in zip(stats, matches, strict=True))

    names = [f"image_{k}.jpg" for k in range(len(sift_features))]
    path = write_match_statistics(stats, names, tmp_path / "match_statistics.csv")
    header, *rows = path.read_text().strip().splitlines()
    assert header == (
        "image_i,image_j,features_i,features_j,raw,after_ratio,after_mutual,time_ms,cached"
    )
    assert len(rows) == len(pairs)
    assert rows[0].startswith("image_0.jpg,image_1.jpg,")


def test_matches_save_load_round_trip(tmp_path, sift_features):
    matcher = DescriptorMatcher(MatchConfig())
    matches, _ = match_pairs(sift_features, [(0, 1), (1, 2)], matcher)
    matches.append(Matches.empty(2, 3))
    path = save_matches(matches, tmp_path / "matches.npz")
    loaded = load_matches(path)
    assert [(m.i, m.j, len(m)) for m in loaded] == [(m.i, m.j, len(m)) for m in matches]
    np.testing.assert_array_equal(loaded[0].idx_j, matches[0].idx_j)


def test_connectivity_flags_isolated_images(sift_features):
    extractor = SIFTExtractor(max_features=3000)
    features = [*sift_features[:3], extractor.detect(make_natural_texture(99))]
    pairs = select_pairs(4, MatchConfig())
    _, stats = match_pairs(features, pairs, DescriptorMatcher(MatchConfig()))
    best = connectivity(stats, n_images=4)
    assert best[:3].min() > 300
    assert best[3] < 30


def test_draw_matches(views, sift_features):
    m, _ = DescriptorMatcher(MatchConfig()).match(*sift_features[:2])
    small = cv2.resize(views[0][1], (320, 240))
    canvas = draw_matches(views[0][0], small, sift_features[0], sift_features[1], m, 50)
    assert canvas.shape == (480, 640 + 320, 3)
    assert canvas.dtype == np.uint8


def test_cli_matches_writes_outputs(tmp_path):
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

    args = ["matches", str(tmp_path / "scene"), "--config", str(config), "--out", str(out)]
    assert main([*args, "--draw", "2", "--ratio", "0.8"]) == 0

    (run_dir,) = out.iterdir()
    m = run_dir / "matches"
    assert (m / "match_statistics.csv").is_file()
    assert (m / "match_matrix.png").is_file()
    assert sorted(p.name for p in m.glob("*_matches.png")) == [
        "view_01_view_02_matches.png",
        "view_02_view_03_matches.png",
    ]
    loaded = load_matches(m / "matches.npz")
    assert [(x.i, x.j) for x in loaded] == [(0, 1), (0, 2), (1, 2)]
    assert "ratio: 0.8" in (run_dir / "config.yaml").read_text()


def test_ratios_explain_the_ratio_test(sift_features):
    matcher = DescriptorMatcher(MatchConfig(ratio=0.75, mutual=False))
    ratios = matcher.ratios(*sift_features[:2])
    _, stats = matcher.match(*sift_features[:2])
    assert len(ratios) == stats.raw
    assert np.all((ratios >= 0) & (ratios <= 1))
    assert int(np.sum(ratios < 0.75)) == stats.after_ratio


def test_ratios_separate_overlapping_from_unrelated_images(sift_features):
    extractor = SIFTExtractor(max_features=3000)
    unrelated = extractor.detect(make_natural_texture(5))
    matcher = DescriptorMatcher(MatchConfig())
    overlapping = matcher.ratios(sift_features[0], sift_features[1])
    different = matcher.ratios(sift_features[0], unrelated)
    assert np.median(overlapping) < 0.6 < np.median(different)


# --- efficiency changes keep results identical --------------------------------------------


def test_exact_matching_equals_bfmatcher(sift_features):
    # Regression oracle for the switch from BFMatcher.knnMatch to cv2.batchDistance.
    fi, fj = sift_features[0], sift_features[1]
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(fi.descriptors, fj.descriptors, k=2)
    best = np.array([[m[0].trainIdx, m[0].distance, m[1].distance] for m in knn])
    keep = best[:, 1] < 0.75 * best[:, 2]
    m, _ = DescriptorMatcher(MatchConfig(ratio=0.75, mutual=False)).match(fi, fj)
    np.testing.assert_array_equal(m.idx_i, np.flatnonzero(keep))
    np.testing.assert_array_equal(m.idx_j, best[keep, 0].astype(np.int32))


def test_flann_builds_one_index_per_image(sift_features):
    matcher = DescriptorMatcher(MatchConfig(method="flann"))
    match_pairs(sift_features, select_pairs(len(sift_features), MatchConfig()), matcher)
    assert len(matcher._flann) == len(sift_features)


def test_match_cache_hit_is_identical(tmp_path, sift_features):
    from src.features.matcher import MatchCache

    pairs = select_pairs(len(sift_features), MatchConfig())
    cache = MatchCache(tmp_path / "matches")
    first, s1 = match_pairs(sift_features, pairs, DescriptorMatcher(MatchConfig()), cache)
    second, s2 = match_pairs(sift_features, pairs, DescriptorMatcher(MatchConfig()), cache)
    assert not any(s.cached for s in s1) and all(s.cached for s in s2)
    for a, b in zip(first, second, strict=True):
        np.testing.assert_array_equal(a.idx_i, b.idx_i)
        np.testing.assert_array_equal(a.idx_j, b.idx_j)
    assert [(s.raw, s.after_ratio, s.final) for s in s1] == [
        (s.raw, s.after_ratio, s.final) for s in s2
    ]
    # Different matching settings must not reuse these entries.
    _, s3 = match_pairs(sift_features, pairs, DescriptorMatcher(MatchConfig(ratio=0.7)), cache)
    assert not any(s.cached for s in s3)
    # Pair selection does not change a pair's matches, so it must not invalidate them.
    _, s4 = match_pairs(
        sift_features, pairs, DescriptorMatcher(MatchConfig(pair_mode="sequential")), cache
    )
    assert all(s.cached for s in s4)


def test_match_cache_key_covers_every_matching_setting(tmp_path):
    # Every MatchConfig field that is not pair selection changes the cache key, so a new
    # matching option cannot silently reuse stale entries.
    import dataclasses

    from src.features.matcher import PAIR_SELECTION_FIELDS, MatchCache

    cache, base = MatchCache(tmp_path), MatchConfig()
    for f in dataclasses.fields(MatchConfig):
        if f.name in PAIR_SELECTION_FIELDS:
            continue
        value = getattr(base, f.name)
        if isinstance(value, bool):
            other = not value
        elif isinstance(value, int | float):
            other = value + 1
        else:
            other = "flann" if value != "flann" else "bruteforce"
        changed = dataclasses.replace(base, **{f.name: other})
        assert cache.path_for("a", "b", changed) != cache.path_for("a", "b", base), f.name


@pytest.mark.parametrize(("n", "mode"), [(3, "exhaustive"), (50, "exhaustive"), (51, "sequential")])
def test_auto_pair_mode(n, mode):
    from src.features.matcher import resolve_pair_mode

    cfg = MatchConfig(pair_mode="auto", exhaustive_max_images=50)
    assert resolve_pair_mode(n, cfg) == mode
    expected = MatchConfig(pair_mode=mode, sequential_window=cfg.sequential_window)
    assert select_pairs(n, cfg) == select_pairs(n, expected)


def test_isolated_images(sift_features):
    from src.features.matcher import isolated_images

    extractor = SIFTExtractor(max_features=3000)
    features = [*sift_features[:3], extractor.detect(make_natural_texture(99))]
    _, stats = match_pairs(
        features, select_pairs(4, MatchConfig()), DescriptorMatcher(MatchConfig())
    )
    assert isolated_images(stats, 4, min_matches=30) == [3]

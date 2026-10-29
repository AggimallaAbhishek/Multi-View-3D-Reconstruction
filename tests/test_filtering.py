import numpy as np
import pytest

from src.config import FilteringConfig
from src.pointcloud.filtering import FILTER_STEPS, filter_cloud

OFF = FilteringConfig(statistical_nb_neighbors=0, radius_nb_points=0)


def _blob(n=600, seed=0):
    return np.random.default_rng(seed).normal(0.0, 1.0, (n, 3))


def _steps(result):
    return {s.name: s.removed for s in result.steps}


def test_point_filters_remove_exactly_the_planted_points():
    X = _blob()
    errors = np.full(len(X), 0.3)
    tracks = np.full(len(X), 4)
    X[0] = np.nan
    errors[1:4] = 3.0  # too high
    tracks[4:9] = 2  # seen twice only
    config = FilteringConfig(
        max_reprojection_error_px=1.0, min_track_length=3,
        statistical_nb_neighbors=0, radius_nb_points=0,
    )  # fmt: skip
    result = filter_cloud(X, config, errors=errors, track_lengths=tracks)
    expected = np.ones(len(X), bool)
    expected[:9] = False
    np.testing.assert_array_equal(result.keep, expected)
    assert _steps(result) == {"non_finite": 1, "reprojection": 3, "track_length": 5,
                              "statistical": 0, "radius": 0}  # fmt: skip


def test_steps_run_in_order_and_chain_their_counts():
    X = np.vstack([_blob(), [[40.0, 0, 0], [0, -35.0, 10]]])
    result = filter_cloud(X, FilteringConfig())
    assert [s.name for s in result.steps] == list(FILTER_STEPS)
    assert result.steps[0].before == len(X)
    for a, b in zip(result.steps, result.steps[1:], strict=False):
        assert b.before == a.after  # each step sees only the survivors of the last
    assert result.steps[-1].after == result.keep.sum() <= len(X)  # never adds points
    assert result.summary()["kept"] == int(result.keep.sum())


def test_statistical_filter_removes_far_outliers():
    X = np.vstack([_blob(), [[40.0, 0, 0], [0, -35.0, 10], [12.0, 12.0, 12.0]]])
    config = FilteringConfig(radius_nb_points=0)
    result = filter_cloud(X, config)
    assert not result.keep[-3:].any()
    assert result.keep[:-3].mean() > 0.95  # the blob itself is mostly kept


def test_radius_filter_removes_isolated_points_whatever_the_scale():
    X = np.vstack([_blob(), [[5.0, 5.0, 5.0], [-6.0, 4.0, 0.0]]])
    config = FilteringConfig(statistical_nb_neighbors=0)
    result = filter_cloud(X, config)
    assert not result.keep[-2:].any()
    assert result.keep[:-2].mean() > 0.95
    # The radius is relative to the point spacing: a reconstruction known only up to
    # scale is filtered the same at any scale.
    np.testing.assert_array_equal(filter_cloud(1e3 * X, config).keep, result.keep)
    np.testing.assert_array_equal(filter_cloud(1e-3 * X, config).keep, result.keep)


@pytest.mark.parametrize("n", [0, 1, 5])
def test_tiny_clouds_pass_through(n):
    X = _blob(n)
    result = filter_cloud(X, FilteringConfig())
    assert result.keep.shape == (n,) and result.keep.all()
    assert all(s.removed == 0 for s in result.steps)


def test_disabled_filters_keep_everything():
    X = np.vstack([_blob(), [[40.0, 0, 0]]])
    result = filter_cloud(X, OFF)
    assert result.keep.all()


def test_errors_and_track_lengths_are_optional_but_must_match():
    X = _blob(50)
    assert filter_cloud(X, OFF).keep.all()
    with pytest.raises(ValueError, match="errors"):
        filter_cloud(X, OFF, errors=np.zeros(3))

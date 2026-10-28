import pytest

from tests.synthetic import SyntheticScene, make_scene


@pytest.fixture
def scene() -> SyntheticScene:
    """Noise-free five-camera scene."""
    return make_scene(n_points=200, n_cameras=5, noise_px=0.0, seed=0)


@pytest.fixture
def noisy_scene() -> SyntheticScene:
    """Same geometry as ``scene`` with 1 px Gaussian pixel noise."""
    return make_scene(n_points=200, n_cameras=5, noise_px=1.0, seed=0)

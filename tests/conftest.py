import numpy as np
import numpy.typing as npt
import pytest

MASK_THRESHOLD = 0.5


@pytest.fixture
def image_random() -> npt.NDArray:
    """Random image as NumPy array.

    Returns
    -------
    npt.NDArray
        Random 2 dimensional array

    """
    rng = np.random.default_rng(seed=1000)
    return rng.random((1024, 1024))


@pytest.fixture
def mask_random(image_random: np.ndarray) -> npt.NDArray:
    """Random mask as NumPy array.

    Returns
    -------
    npt.NDArray
        Random 2 dimensional boolean array

    """
    return image_random > MASK_THRESHOLD

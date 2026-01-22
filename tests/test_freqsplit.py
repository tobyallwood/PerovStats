import pytest
import numpy as np

from perovstats.freqsplit import extend_image, create_frequency_mask, frequency_split

def test_extend_image(image_random: np.ndarray) -> None:
    """Test extending an image."""
    extended_image, extent = extend_image(image=image_random)

    rows, cols = image_random.shape
    assert isinstance(extent, dict)
    assert extent["top"] == rows // 2
    assert extent["bottom"] == rows // 2
    assert extent["left"] == cols // 2
    assert extent["right"] == cols // 2

    assert extended_image.shape == (
        rows + extent["top"] + extent["bottom"],
        cols + extent["left"] + extent["right"],
    )


def test_extend_image_not_implemented_error(image_random: np.ndarray) -> None:
    """Test NotImplementedError is raised if method != cv2.BORDER_REFLECT."""
    with pytest.raises(NotImplementedError):
        extend_image(image=image_random, method=0)


@pytest.mark.parametrize(
    ("shape", "cutoff", "width"),
    [
        (
            (512, 512),
            0.5,
            0,
        ),
        (
            (256, 512),
            0.5,
            0,
        ),
        (
            (512, 512),
            0.5,
            0.1,
        ),
    ],
)
def test_create_frequency_mask(shape: tuple, cutoff: float, width: float) -> None:
    """Test creating a frequency mask."""
    x = create_frequency_mask(shape, cutoff=cutoff, edge_width=width)

    assert x.shape == shape



def test_frequency_split():
    image = np.array(
        [
            [0, 1, 0.5],
            [0.6, 0, 0.3],
            [0.1, 0.9, 0.5]
        ]
    )
    cutoff = 0.01
    edge_width = 0.03

    expected_high_pass = np.array(
        [
            [-0.27252962, 0.72747038, 0.22747038],
            [0.32747038, -0.27252962, 0.02747038],
            [-0.17252962, 0.62747038, 0.22747038]
        ]
    )
    expected_low_pass = np.array(
        [
            [0.27252962, 0.27252962, 0.27252962],
            [0.27252962, 0.27252962, 0.27252962],
            [0.27252962, 0.27252962, 0.27252962]
        ]
    )

    high_pass, low_pass = frequency_split(image, cutoff, edge_width)

    print(high_pass)

    assert np.allclose(high_pass, expected_high_pass)
    assert np.allclose(low_pass, expected_low_pass)

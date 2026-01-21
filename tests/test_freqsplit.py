import pytest
import numpy as np

from perovstats.freqsplit import extend_image

@pytest.mark.parametrize(
        ("image","expected"),
        [
            pytest.param(
                np.array(
                    [
                        [1, 0, 1],
                        [1, 0, 1],
                        [1, 1, 1]
                    ]
                ),
                np.array(
                    [
                        [1, 1, 0, 1, 1],
                        [1, 1, 0, 1, 1],
                        [1, 1, 0, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                    ]
                )
            ),
            pytest.param(
                np.array(
                    [
                        [1, 1, 0, 1, 1],
                        [1, 1, 0, 1, 1],
                        [1, 1, 0, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1]
                    ]
                ),
                np.array(
                    [
                        [1, 1, 1, 1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1]
                    ]
                )
            ),
            pytest.param(
                np.array(
                    [
                        [1, 1],
                        [1, 0]
                    ]
                ),
                np.array(
                    [
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 0, 0],
                        [1, 1, 0, 0]
                    ]
                )
            ),
            pytest.param(
                np.array([[1]]),
                np.array([[1]])
            ),
        ]

)
def test_extend_image(image, expected):
    extended_image, _ = extend_image(image)
    assert np.array_equal(extended_image, expected)


def test_create_frequency_mask():
    pass


def test_frequency_split():
    pass

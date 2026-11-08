"""Mipmap generation tests."""

from __future__ import annotations

import numpy as np
import pytest

from tga2tpc.mipmaps import (
    generate_mipmaps,
    mip_level_count,
    writer_mip_levels,
)


def test_mip_level_count():
    assert mip_level_count(1, 1) == 1
    assert mip_level_count(2, 2) == 2
    assert mip_level_count(128, 128) == 8
    assert mip_level_count(3, 5) == 4
    assert mip_level_count(128, 720) == 11


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (1, 1, [(1, 1)]),
        (2, 2, [(2, 2), (1, 1)]),
        (4, 4, [(4, 4), (2, 2), (1, 1)]),
        (8, 8, [(8, 8), (4, 4), (2, 2), (1, 1)]),
        (128, 128, [(128, 128), (64, 64), (32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]),
        (128, 720, [(128, 720), (64, 360), (32, 180), (16, 90), (8, 45), (4, 22), (2, 11), (1, 5), (1, 2), (1, 1)]),
        (5, 3, [(5, 3), (2, 1), (1, 1)]),
    ],
)
def test_writer_mip_levels(width, height, expected):
    assert writer_mip_levels(width, height) == expected


@pytest.mark.parametrize(
    ("width", "height"),
    [(1, 1), (2, 1), (3, 2), (5, 3), (6, 4), (7, 5), (16, 16), (128, 720), (720, 128)],
)
def test_generated_levels_cover_writer_levels(width, height):
    image = np.zeros((height, width, 4), dtype=np.uint8)
    generated = generate_mipmaps(image)
    written = writer_mip_levels(width, height)
    assert len(generated) >= len(written)
    for expected, actual in zip(written, generated):
        assert actual.shape == (expected[1], expected[0], 4)


def test_bilinear_2x2_average():
    image = np.zeros((2, 2, 4), dtype=np.uint8)
    image[0, 0] = (0, 4, 8, 12)
    image[0, 1] = (4, 8, 12, 16)
    image[1, 0] = (8, 12, 16, 20)
    image[1, 1] = (12, 16, 20, 24)
    levels = generate_mipmaps(image, bicubic=False)
    assert len(levels) == 2
    np.testing.assert_array_equal(
        levels[1], np.array([[[6, 10, 14, 18]]], dtype=np.uint8)
    )


def test_bicubic_final_2x2_to_1x1_is_finite_average():
    # The original JS produced NaNs for this case; a 2x2 image must reduce to
    # the average of its four texels with the cubic kernel at x=0.5.
    image = np.zeros((2, 2, 4), dtype=np.uint8)
    image[0, 0] = (0, 4, 8, 12)
    image[0, 1] = (4, 8, 12, 16)
    image[1, 0] = (8, 12, 16, 20)
    image[1, 1] = (12, 16, 20, 24)
    levels = generate_mipmaps(image, bicubic=True)
    assert levels[1].shape == (1, 1, 4)
    assert np.issubdtype(levels[1].dtype, np.integer)
    np.testing.assert_array_equal(
        levels[1], np.array([[[6, 10, 14, 18]]], dtype=np.uint8)
    )


def test_bicubic_chain_for_4x4_is_finite_and_sized():
    rng = np.random.default_rng(1234)
    image = rng.integers(0, 256, size=(4, 4, 4), dtype=np.uint8)
    levels = generate_mipmaps(image, bicubic=True)
    assert [level.shape for level in levels] == [
        (4, 4, 4),
        (2, 2, 4),
        (1, 1, 4),
    ]
    assert all(np.all(level >= 0) and np.all(level <= 255) for level in levels)


def test_weighted_odd_downsample_area_average():
    # A 3x2 image is reduced by the weighted path to 1x1; the result is the
    # rounded area average of all six texels.
    image = np.empty((2, 3, 4), dtype=np.uint8)
    for y in range(2):
        for x in range(3):
            i = y * 3 + x
            image[y, x] = (i, i + 10, i * 2, i + 20)
    levels = generate_mipmaps(image, bicubic=True)
    assert levels[1].shape == (1, 1, 4)
    np.testing.assert_array_equal(
        levels[1], np.array([[[3, 13, 5, 23]]], dtype=np.uint8)
    )


def test_constant_image_stays_constant():
    for bicubic in (False, True):
        image = np.full((9, 7, 4), (23, 45, 67, 89), dtype=np.uint8)
        levels = generate_mipmaps(image, bicubic=bicubic)
        for level in levels:
            np.testing.assert_array_equal(
                level,
                np.full(level.shape, (23, 45, 67, 89), dtype=np.uint8),
            )


def test_generate_mipmaps_accepts_non_uint8_and_checks_shape():
    image = np.zeros((2, 2, 4), dtype=np.int16)
    levels = generate_mipmaps(image)
    assert levels[0].dtype == np.uint8
    with pytest.raises(ValueError):
        generate_mipmaps(np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        generate_mipmaps(np.zeros((2, 2), dtype=np.uint8))

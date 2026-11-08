"""DXT encoder tests, including independent block decoding and PSNR checks."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tga2tpc.dxt import (
    PROFILE_ITERATIONS,
    encode_dxt1,
    encode_dxt3,
    encode_dxt5,
)

from .helpers import decode_dxt, psnr, solid_rgba


def _expected_size(width: int, height: int, kind: str) -> int:
    blocks = ((width + 3) // 4) * ((height + 3) // 4)
    return blocks * (8 if kind == "dxt1" else 16)


@pytest.mark.parametrize(
    ("width", "height"),
    [(1, 1), (4, 4), (5, 3), (8, 8), (17, 9), (64, 32)],
)
def test_encoder_output_sizes(width, height):
    image = solid_rgba(width, height, (12, 34, 56, 78))
    assert len(encode_dxt1(image)) == _expected_size(width, height, "dxt1")
    assert len(encode_dxt3(image)) == _expected_size(width, height, "dxt3")
    assert len(encode_dxt5(image)) == _expected_size(width, height, "dxt5")


def test_dxt1_solid_colour_roundtrip():
    image = solid_rgba(4, 4, (255, 0, 0, 255))
    decoded = decode_dxt(encode_dxt1(image), 4, 4, "dxt1")
    np.testing.assert_array_equal(
        decoded[:, :, :3],
        np.full((4, 4, 3), (255, 0, 0), dtype=np.uint8),
    )
    np.testing.assert_array_equal(
        decoded[:, :, 3], np.full((4, 4), 255, dtype=np.uint8)
    )


def test_dxt1_quantisation_error_is_small():
    image = solid_rgba(8, 8, (17, 200, 99, 255))
    decoded = decode_dxt(encode_dxt1(image), 8, 8, "dxt1")
    assert np.abs(decoded[:, :, :3].astype(int) - image[:, :, :3].astype(int)).max() <= 2


def test_dxt1_always_uses_four_colour_mode():
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, size=(16, 16, 4), dtype=np.uint8)
    data = encode_dxt1(image)
    for block in range(0, len(data), 8):
        c0, c1 = struct.unpack_from("<HH", data, block)
        assert c0 > c1


def _gradient_image(width: int, height: int) -> np.ndarray:
    image = np.empty((height, width, 4), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            image[y, x, 0] = (x * 255) // max(1, width - 1)
            image[y, x, 1] = (y * 255) // max(1, height - 1)
            image[y, x, 2] = ((x + y) * 255) // max(1, width + height - 2)
            image[y, x, 3] = ((x * 7 + y * 3) * 255) // max(
                1, (width - 1) * 7 + (height - 1) * 3
            )
    return image


def test_dxt1_gradient_quality():
    image = _gradient_image(48, 48)
    decoded = decode_dxt(encode_dxt1(image), 48, 48, "dxt1")
    assert psnr(image[:, :, :3], decoded[:, :, :3]) > 35.0


def test_dxt5_gradient_colour_and_alpha_quality():
    image = _gradient_image(48, 48)
    decoded = decode_dxt(encode_dxt5(image), 48, 48, "dxt5")
    assert psnr(image[:, :, :3], decoded[:, :, :3]) > 35.0
    assert psnr(image[:, :, 3], decoded[:, :, 3]) > 38.0


def test_dxt3_gradient_alpha_quality():
    image = _gradient_image(48, 48)
    decoded = decode_dxt(encode_dxt3(image), 48, 48, "dxt3")
    assert psnr(image[:, :, :3], decoded[:, :, :3]) > 35.0
    assert psnr(image[:, :, 3], decoded[:, :, 3]) > 32.0


def test_dxt3_alpha_nibbles_are_exact_for_multiples_of_17():
    image = np.empty((4, 4, 4), dtype=np.uint8)
    image[:, :, :3] = 128
    image[:, :, 3] = np.arange(16, dtype=np.uint8).reshape(4, 4) * 17
    decoded = decode_dxt(encode_dxt3(image), 4, 4, "dxt3")
    np.testing.assert_array_equal(decoded[:, :, 3], image[:, :, 3])


def test_dxt5_solid_alpha_extremes():
    for alpha in (0, 255):
        image = solid_rgba(4, 4, (30, 40, 50, alpha))
        decoded = decode_dxt(encode_dxt5(image), 4, 4, "dxt5")
        np.testing.assert_array_equal(
            decoded[:, :, 3], np.full((4, 4), alpha, dtype=np.uint8)
        )


def _alpha_weighted_rmse(original: np.ndarray, decoded: np.ndarray) -> float:
    alpha = original[:, :, 3].astype(np.float64) / 255.0
    error = decoded[:, :, :3].astype(np.float64) - original[:, :, :3].astype(
        np.float64
    )
    weighted = np.einsum("ijk,ijk->", error * error, alpha[..., None])
    return float(np.sqrt(weighted / max(alpha.sum(), 1e-9) / 3.0))


def _sparse_colour_image() -> np.ndarray:
    rng = np.random.default_rng(123)
    image = np.zeros((16, 16, 4), dtype=np.uint8)
    visible = rng.random((16, 16)) < 0.25
    image[:, :, :3] = np.where(
        visible[..., None],
        rng.integers(0, 256, (16, 16, 3), dtype=np.uint8),
        0,
    )
    image[:, :, 3] = np.where(visible, 255, 0)
    return image


@pytest.mark.parametrize("kind,encoder", [("dxt5", encode_dxt5), ("dxt3", encode_dxt3)])
def test_alpha_weighting_preserves_sparse_visible_colour(kind, encoder):
    image = _sparse_colour_image()
    weighted = decode_dxt(encoder(image), 16, 16, kind)
    unweighted = decode_dxt(
        encoder(image, alpha_weighted=False), 16, 16, kind
    )
    visible = image[:, :, 3] > 0
    assert _alpha_weighted_rmse(image, weighted) < _alpha_weighted_rmse(
        image, unweighted
    )
    assert psnr(
        image[:, :, :3][visible], weighted[:, :, :3][visible]
    ) > psnr(image[:, :, :3][visible], unweighted[:, :, :3][visible])


@pytest.mark.parametrize("kind,encoder", [("dxt5", encode_dxt5), ("dxt3", encode_dxt3)])
def test_alpha_weighting_does_not_change_opaque_images(kind, encoder):
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (16, 16, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    assert encoder(image) == encoder(image, alpha_weighted=False)


def test_dxt5_alpha_gradient_benefits_from_refinement():
    image = np.empty((16, 16, 4), dtype=np.uint8)
    image[:, :, :3] = 100
    image[:, :, 3] = np.arange(256, dtype=np.uint8).reshape(16, 16)
    fast = decode_dxt(
        encode_dxt5(image, "super_fast"), 16, 16, "dxt5"
    )[:, :, 3]
    ultra = decode_dxt(encode_dxt5(image, "ultra"), 16, 16, "dxt5")[
        :, :, 3
    ]
    assert psnr(image[:, :, 3], ultra) >= psnr(image[:, :, 3], fast)


def test_all_profiles_are_deterministic_and_sized():
    rng = np.random.default_rng(99)
    image = rng.integers(0, 256, size=(9, 13, 4), dtype=np.uint8)
    for profile in PROFILE_ITERATIONS:
        first = encode_dxt5(image, profile)
        second = encode_dxt5(image, profile)
        assert first == second
        assert len(first) == _expected_size(13, 9, "dxt5")


def test_invalid_arguments():
    image = np.zeros((4, 4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        encode_dxt1(image, "does-not-exist")
    with pytest.raises(ValueError):
        encode_dxt1(np.zeros((4, 4, 3), dtype=np.uint8))

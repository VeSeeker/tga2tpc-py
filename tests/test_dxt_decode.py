"""Tests for the xoreos-compatible DXT decoders."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tga2tpc.dxt_decode import (
    DXTDecodeError,
    decode_dxt,
    decode_dxt1,
    decode_dxt3,
    decode_dxt5,
)


def _colour_block(c0: int, c1: int, indices: int) -> bytes:
    return struct.pack("<HHI", c0, c1, indices)


def test_rgb565_expansion_truncates_like_xoreos():
    image = decode_dxt1(_colour_block(0xFFFF, 0x0000, 0), 4, 4)
    np.testing.assert_array_equal(image[0, 0], [248, 252, 248, 255])


def test_dxt1_three_colour_mode_makes_index_three_transparent():
    image = decode_dxt1(_colour_block(0x0000, 0xFFFF, 0xFFFFFFFF), 4, 4)
    np.testing.assert_array_equal(image[0, 0], [0, 0, 0, 0])


def test_dxt3_alpha_nibbles_are_shifted_not_scaled():
    block = struct.pack("<8sHHI", b"\xff" * 8, 0xFFFF, 0xFFFF, 0)
    image = decode_dxt3(block, 4, 4)
    np.testing.assert_array_equal(image[0, 0], [248, 252, 248, 240])


def test_dxt5_eight_alpha_mode_uses_reversed_row_bits():
    # alpha_0 > alpha_1 selects the 8-entry interpolation mode.
    a0, a1 = 20, 10
    bits = (2 << (3 * ((3 - 0) * 4 + 0))) | (3 << (3 * ((3 - 3) * 4 + 0)))
    block = bytes((a0, a1)) + bits.to_bytes(6, "little")
    block += _colour_block(0xFFFF, 0xFFFF, 0)
    image = decode_dxt5(block, 4, 4)

    # The block's first row lands in the last output row because of xoreos'
    # block placement, and its last row lands in the first output row.
    assert image[0, 0, 3] == 17
    assert image[3, 0, 3] == 19


def test_dxt5_six_alpha_mode_special_values():
    a0, a1 = 10, 20
    bits = (6 << (3 * ((3 - 0) * 4 + 0))) | (7 << (3 * ((3 - 3) * 4 + 0)))
    block = bytes((a0, a1)) + bits.to_bytes(6, "little")
    block += _colour_block(0xFFFF, 0xFFFF, 0)
    image = decode_dxt5(block, 4, 4)
    assert image[0, 0, 3] == 255
    assert image[3, 0, 3] == 0


@pytest.mark.parametrize(
    ("kind", "size"),
    [("dxt1", 8), ("dxt3", 16), ("dxt5", 16)],
)
def test_truncated_data_raises(kind, size):
    with pytest.raises(DXTDecodeError):
        decode_dxt(b"\x00" * (size - 1), 4, 4, kind)


def test_unknown_kind_raises():
    with pytest.raises(DXTDecodeError):
        decode_dxt(b"\x00" * 8, 4, 4, "dxt2")


def test_odd_dimensions_are_placed_with_xoreos_partial_block_rule():
    # A 5x3 image uses two block columns but only the first three rows of each
    # decoded block.  This mirrors xoreos' MIN(width, 4)/MIN(height, 4) rule.
    block0 = _colour_block(0xFFFF, 0x0000, 0)
    block1 = _colour_block(0x0000, 0xFFFF, 0)
    image = decode_dxt1(block0 + block1, 5, 3)
    assert image.shape == (3, 5, 4)
    np.testing.assert_array_equal(image[0, 0], [248, 252, 248, 255])
    np.testing.assert_array_equal(image[0, 4], [0, 0, 0, 255])

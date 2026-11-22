"""
NumPy DXT1/DXT3/DXT5 decoders.

This module is the decoding counterpart of :mod:`tga2tpc.dxt`.  It reproduces
the pixel arithmetic used by xoreos-tools' ``Images::decompressDXT*`` routines
so that textures converted by :mod:`tga2tpc.tex2tga` match the output of
``xoreostex2tga`` byte for byte.

The returned arrays are RGBA ``uint8`` images in the internal orientation used
by xoreos-tools' TGA dumper.  Callers should not assume that this is the usual
top-left canvas orientation; the TGA writer in :mod:`tga2tpc.tga` writes the
rows as they are.
"""

from __future__ import annotations

import numpy as np

_BLOCK_BYTES = {"dxt1": 8, "dxt3": 16, "dxt5": 16}

# xoreos-tools uses the float literals 0.333333f and 0.666666f in
# interpolate32().  Keeping their float32-rounded values is important for
# byte-identical output.
_WEIGHT_ONE_THIRD = np.float64(np.float32(0.333333))
_WEIGHT_TWO_THIRDS = np.float64(np.float32(0.666666))


class DXTDecodeError(ValueError):
    """Raised when compressed data cannot be decoded."""


def _expand_rgb565(packed: np.ndarray) -> np.ndarray:
    """Expand RGB565 integers exactly like xoreos' convert565To8888()."""

    r = ((packed >> 11) & 0x1F).astype(np.int32) << 3
    g = ((packed >> 5) & 0x3F).astype(np.int32) << 2
    b = (packed & 0x1F).astype(np.int32) << 3
    return np.stack((r, g, b), axis=1)


def _interpolate(
    weight: float, colour0: np.ndarray, colour1: np.ndarray
) -> np.ndarray:
    """xoreos' interpolate32() colour blend, including its truncation."""

    value = (1.0 - weight) * colour0.astype(np.float64)
    value += weight * colour1.astype(np.float64)
    return np.floor(value).astype(np.uint8)


def _colour_palette(
    c0: np.ndarray, c1: np.ndarray, kind: str
) -> tuple[np.ndarray, np.ndarray]:
    """Build the four-entry colour palette and transparency mask per block."""

    e0 = _expand_rgb565(c0)
    e1 = _expand_rgb565(c1)

    palette = np.empty((c0.shape[0], 4, 3), dtype=np.uint8)
    palette[:, 0] = e0.astype(np.uint8)
    palette[:, 1] = e1.astype(np.uint8)
    palette[:, 2] = _interpolate(_WEIGHT_ONE_THIRD, e0, e1)
    palette[:, 3] = _interpolate(_WEIGHT_TWO_THIRDS, e0, e1)

    transparent = np.zeros(c0.shape[0], dtype=bool)
    if kind == "dxt1":
        # DXT1 has a three-colour mode when c0 <= c1.  The fourth entry is
        # fully transparent black.  DXT3 and DXT5 always use four colours.
        three_colour = c0 <= c1
        if np.any(three_colour):
            palette[three_colour, 2] = _interpolate(
                0.5, e0[three_colour], e1[three_colour]
            )
            palette[three_colour, 3] = 0
            transparent[three_colour] = True
    return palette, transparent


def _colour_indices(blocks: np.ndarray, offset: int) -> np.ndarray:
    """Extract the 16 two-bit colour indices.

    xoreos reads the 32-bit colour-index word with ``readUint32BE``.  Reading
    it big-endian reverses the four 4-pixel rows, which is then cancelled by
    the vertical flip in its block placement loop.
    """

    raw = blocks[:, offset : offset + 4].astype(np.uint32)
    bits = (
        (raw[:, 0] << 24)
        | (raw[:, 1] << 16)
        | (raw[:, 2] << 8)
        | raw[:, 3]
    )
    indices = np.empty((blocks.shape[0], 16), dtype=np.uint8)
    for pixel in range(16):
        indices[:, pixel] = (bits >> (2 * pixel)) & 0x3
    return indices


def _alpha_dxt3(blocks: np.ndarray) -> np.ndarray:
    """Decode the explicit 4-bit alpha nibbles of DXT3.

    xoreos shifts each nibble left by four bits instead of scaling it to
    ../../255, so the decoded alpha values are 0, 16, 32, ... 240.
    """

    raw = blocks[:, 0:8].astype(np.uint64)
    bits = (
        raw[:, 0]
        | (raw[:, 1] << 8)
        | (raw[:, 2] << 16)
        | (raw[:, 3] << 24)
        | (raw[:, 4] << 32)
        | (raw[:, 5] << 40)
        | (raw[:, 6] << 48)
        | (raw[:, 7] << 56)
    )
    alpha = np.empty((blocks.shape[0], 16), dtype=np.uint8)
    for pixel in range(16):
        nibble = ((bits >> (4 * pixel)) & 0xF).astype(np.uint8)
        alpha[:, pixel] = nibble << 4
    return alpha


def _alpha_dxt5_palette(a0: np.ndarray, a1: np.ndarray) -> np.ndarray:
    """Build the eight-entry DXT5 alpha palette exactly as xoreos does."""

    palette = np.empty((a0.shape[0], 8), dtype=np.uint8)
    palette[:, 0] = a0
    palette[:, 1] = a1

    eight_mode = a0 > a1
    for index in range(1, 7):
        value = ((7 - index) * a0 + index * a1 + 3) // 7
        palette[:, 1 + index] = np.where(eight_mode, value, 0)
    for index in range(1, 5):
        value = ((5 - index) * a0 + index * a1 + 2) // 5
        palette[:, 1 + index] = np.where(
            eight_mode, palette[:, 1 + index], value
        )
    palette[:, 6] = np.where(eight_mode, palette[:, 6], 0)
    palette[:, 7] = np.where(eight_mode, palette[:, 7], 255)
    return palette


def _alpha_dxt5(blocks: np.ndarray) -> np.ndarray:
    """Decode DXT5 alpha, including xoreos' reversed row bit order."""

    a0 = blocks[:, 0].astype(np.int32)
    a1 = blocks[:, 1].astype(np.int32)
    palette = _alpha_dxt5_palette(a0, a1)

    raw = blocks[:, 2:8].astype(np.uint64)
    bits = (
        raw[:, 0]
        | (raw[:, 1] << 8)
        | (raw[:, 2] << 16)
        | (raw[:, 3] << 24)
        | (raw[:, 4] << 32)
        | (raw[:, 5] << 40)
    )

    alpha = np.empty((blocks.shape[0], 16), dtype=np.uint8)
    block_index = np.arange(blocks.shape[0])
    for y in range(4):
        for x in range(4):
            pixel = y * 4 + x
            # xoreos indexes the alpha bits with (3 - y), not y.
            bit_row = 3 - y
            index = (bits >> (3 * (bit_row * 4 + x))) & 0x7
            alpha[:, pixel] = palette[block_index, index]
    return alpha


def _place_blocks(
    source: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Place decoded 4x4 blocks using xoreos' coordinate mapping."""

    block_w = (width + 3) // 4
    block_h = (height + 3) // 4
    source = source.reshape(block_h, block_w, 4, 4, 4)

    out = np.empty((height, width, 4), dtype=np.uint8)
    # xoreos uses MIN(width, 4) / MIN(height, 4), not the remaining size of
    # the final partial block.
    valid_block_h = min(height, 4)
    valid_block_w = min(width, 4)
    rows = np.arange(block_h)
    cols = np.arange(block_w)

    for y in range(valid_block_h):
        dest_y = 4 * rows + valid_block_h - 1 - y
        row_mask = (dest_y >= 0) & (dest_y < height)
        if not np.any(row_mask):
            continue
        for x in range(valid_block_w):
            dest_x = 4 * cols + x
            col_mask = dest_x < width
            if not np.any(col_mask):
                continue
            out[np.ix_(dest_y[row_mask], dest_x[col_mask])] = source[
                row_mask
            ][:, col_mask, y, x]
    return out


def decode_dxt(
    data: bytes | bytearray | memoryview,
    width: int,
    height: int,
    kind: str,
) -> np.ndarray:
    """Decode DXT1/DXT3/DXT5 bytes into an internal RGBA image."""

    if width <= 0 or height <= 0:
        raise DXTDecodeError("invalid compressed image dimensions")
    if kind not in _BLOCK_BYTES:
        raise DXTDecodeError(f"unknown DXT kind: {kind}")

    block_bytes = _BLOCK_BYTES[kind]
    block_w = (width + 3) // 4
    block_h = (height + 3) // 4
    block_count = block_w * block_h
    needed = block_count * block_bytes
    view = memoryview(data)
    if len(view) < needed:
        raise DXTDecodeError(
            f"truncated {kind.upper()} data: need {needed} bytes, "
            f"got {len(view)}"
        )

    blocks = np.frombuffer(
        view[:needed], dtype=np.uint8, count=needed
    ).reshape(block_count, block_bytes)

    colour_offset = 0 if kind == "dxt1" else 8
    c0 = (
        blocks[:, colour_offset].astype(np.uint16)
        | (blocks[:, colour_offset + 1].astype(np.uint16) << 8)
    )
    c1 = (
        blocks[:, colour_offset + 2].astype(np.uint16)
        | (blocks[:, colour_offset + 3].astype(np.uint16) << 8)
    )
    palette, transparent = _colour_palette(c0, c1, kind)
    indices = _colour_indices(blocks, colour_offset + 4)

    block_index = np.arange(block_count)
    rgba = np.empty((block_count, 16, 4), dtype=np.uint8)
    for pixel in range(16):
        rgba[:, pixel, :3] = palette[block_index, indices[:, pixel]]

    if kind == "dxt1":
        alpha = np.full((block_count, 16), 255, dtype=np.uint8)
        if np.any(transparent):
            for pixel in range(16):
                alpha[:, pixel] = np.where(
                    transparent & (indices[:, pixel] == 3), 0, 255
                )
    elif kind == "dxt3":
        alpha = _alpha_dxt3(blocks)
    else:
        alpha = _alpha_dxt5(blocks)
    rgba[:, :, 3] = alpha

    return _place_blocks(rgba, width, height)


def decode_dxt1(
    data: bytes | bytearray | memoryview, width: int, height: int
) -> np.ndarray:
    """Decode DXT1/BC1 data into an internal RGBA image."""

    return decode_dxt(data, width, height, "dxt1")


def decode_dxt3(
    data: bytes | bytearray | memoryview, width: int, height: int
) -> np.ndarray:
    """Decode DXT3/BC2 data into an internal RGBA image."""

    return decode_dxt(data, width, height, "dxt3")


def decode_dxt5(
    data: bytes | bytearray | memoryview, width: int, height: int
) -> np.ndarray:
    """Decode DXT5/BC3 data into an internal RGBA image."""

    return decode_dxt(data, width, height, "dxt5")

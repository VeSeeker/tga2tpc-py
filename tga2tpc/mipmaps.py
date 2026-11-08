"""
Mipmap generation compatible with tga2tpc's JavaScript implementation.

The original tool generates each detail level from the previous one.  For even-sized parents it either averages 2x2 texels (bilinear) or samples a 4x4 neighbourhood with the Catmull-Rom-like cubic used by tga2tpc (bicubic). When a parent has an odd dimension it uses a weighted area average, because a plain 2x2 sample would not cover the parent evenly.

All returned arrays are RGBA, ``uint8`` and in top-left orientation.
"""

from __future__ import annotations

import math

import numpy as np


def mip_level_count(width: int, height: int) -> int:
    """Number of mip levels tga2tpc generates for a layer."""

    maximum = max(int(width), int(height), 1)
    return int(math.ceil(math.log2(maximum))) + 1


def writer_mip_levels(width: int, height: int) -> list[tuple[int, int]]:
    """Dimensions of the levels the TPC writer actually emits.

    This mirrors the JavaScript writer loop: dimensions are repeatedly
    halved with integer truncation, clamping to one when the other axis has
    already reached zero.
    """

    levels: list[tuple[int, int]] = []
    w = max(int(width), 1)
    h = max(int(height), 1)
    while w >= 1 or h >= 1:
        levels.append((max(w, 1), max(h, 1)))
        w //= 2
        h //= 2
    return levels


def generate_mipmaps(rgba: np.ndarray, bicubic: bool = True) -> list[np.ndarray]:
    """Generate all detail levels for one layer.

    The input must be ``(height, width, 4)`` uint8 RGBA in top-left
    orientation.  The first returned array is the input itself.
    """

    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("mipmap input must be an (h, w, 4) RGBA array")
    if rgba.dtype != np.uint8:
        rgba = rgba.astype(np.uint8, copy=False)

    height, width = rgba.shape[:2]
    total = mip_level_count(width, height)
    levels: list[np.ndarray] = [np.ascontiguousarray(rgba)]
    for _ in range(1, total):
        levels.append(_downsample(levels[-1], bicubic))
    return levels


def _downsample(parent: np.ndarray, bicubic: bool) -> np.ndarray:
    ph, pw = parent.shape[:2]
    th = max(ph // 2, 1)
    tw = max(pw // 2, 1)

    if ph % 2 == 0 and pw % 2 == 0:
        if bicubic:
            return _bicubic_downsample(parent)
        return _bilinear_2x2(parent)

    return _weighted_downsample(parent, th, tw)


def _bilinear_2x2(parent: np.ndarray) -> np.ndarray:
    # parent dimensions are even, so these slices have identical shapes.
    p = parent.astype(np.int32)
    total = (
        p[0::2, 0::2]
        + p[0::2, 1::2]
        + p[1::2, 0::2]
        + p[1::2, 1::2]
    )
    # Math.round(total * 0.25), with total a non-negative integer.
    return ((total + 2) // 4).astype(np.uint8)


def _axis_indices(parent_size: int, out_size: int) -> np.ndarray:
    """Four source indices for each destination position (bicubic path)."""

    base = (np.arange(out_size, dtype=np.int64) * parent_size) // out_size
    indices = np.stack((base - 1, base, base + 1, base + 2))
    if out_size == 1:
        # The JavaScript code changes the first point, and its "last point"
        # branch is an else-if that never runs for a single output pixel.
        indices[0] = base
    else:
        indices[0, 0] = base[0]
        indices[3, -1] = base[-1] + 1
    np.clip(indices, 0, parent_size - 1, out=indices)
    return indices


def _cubic(values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
    """tga2tpc's cubic interpolation evaluated at x=0.5."""

    p0, p1, p2, p3 = values
    inner = 3.0 * (p1 - p2) + p3 - p0
    middle = 2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + 0.5 * inner
    return p1 + 0.25 * ((p2 - p0) + 0.5 * middle)


def _bicubic_downsample(parent: np.ndarray) -> np.ndarray:
    ph, pw = parent.shape[:2]
    th = max(ph // 2, 1)
    tw = max(pw // 2, 1)
    src = parent.astype(np.float64)

    y_idx = _axis_indices(ph, th)
    # Vertical interpolation first, matching the order in the JS source.
    temp = _cubic((src[y_idx[0]], src[y_idx[1]], src[y_idx[2]], src[y_idx[3]]))

    x_idx = _axis_indices(pw, tw)
    result = _cubic(
        (temp[:, x_idx[0]], temp[:, x_idx[1]], temp[:, x_idx[2]], temp[:, x_idx[3]])
    )
    result = np.clip(result, 0.0, 255.0)
    # Math.round for the non-negative values involved here.
    return np.floor(result + 0.5).astype(np.uint8)


def _weighted_downsample(
    parent: np.ndarray, out_height: int, out_width: int
) -> np.ndarray:
    """Exact port of the JavaScript weighted/area downsampler."""

    ph, pw = parent.shape[:2]

    # Precompute the x factors.  The JavaScript code multiplies factors in
    # the order x1, y1, x2, y2, so keep the two x factors separate.
    x_entries: list[list[tuple[int, float | None, float | None]]] = []
    for x in range(out_width):
        pxmin = (x / out_width) * pw
        pxmax = ((x + 1) / out_width) * pw
        entry: list[tuple[int, float | None, float | None]] = []
        for px in range(math.floor(pxmin), math.floor(pxmax) + 1):
            factor1 = (px + 1) - pxmin if px < pxmin else None
            factor2 = (
                pxmax - px
                if px == math.floor(pxmax) and pxmax >= px
                else None
            )
            entry.append((px, factor1, factor2))
        x_entries.append(entry)

    result = np.empty((out_height, out_width, 4), dtype=np.uint8)

    for y in range(out_height):
        pymin = (y / out_height) * ph
        pymax = ((y + 1) / out_height) * ph
        y_entries: list[tuple[int, float | None, float | None]] = []
        for py in range(math.floor(pymin), math.floor(pymax) + 1):
            factor1 = (py + 1) - pymin if py < pymin else None
            factor2 = (
                pymax - py
                if py == math.floor(pymax) and pymax >= py
                else None
            )
            y_entries.append((py, factor1, factor2))

        for x in range(out_width):
            total = 0.0
            samples: list[tuple[int, int, float]] = []
            for py, y_factor1, y_factor2 in y_entries:
                for px, x_factor1, x_factor2 in x_entries[x]:
                    weight = 1.0
                    if x_factor1 is not None:
                        weight *= x_factor1
                    if y_factor1 is not None:
                        weight *= y_factor1
                    if x_factor2 is not None:
                        weight *= x_factor2
                    if y_factor2 is not None:
                        weight *= y_factor2
                    total += weight
                    if (
                        weight > 0.0
                        and 0 <= py < ph
                        and 0 <= px < pw
                    ):
                        samples.append((py, px, weight))

            if total == 0.0:
                result[y, x] = 0
                continue
            for channel in range(4):
                datum = 0.0
                for py, px, weight in samples:
                    datum = datum + float(parent[py, px, channel]) * weight
                result[y, x, channel] = int(math.floor(datum / total + 0.5))
    return result

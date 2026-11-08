"""
NumPy DXT1/DXT3/DXT5 encoders.

The original application used two different compressors: libsquish for the ``super_fast`` profile and a JavaScript port of AMD's Compressonator for the other profiles.  This module uses a single vectorised encoder with a principal-axis start and iteratively refined least-squares endpoints.  The profile controls how many refinement passes are made.

Blocks are consumed in ordinary row-major order.  Feeding the encoder a bottom-up image (the TPC convention) therefore also produces bottom-up blocks.

For DXT3/DXT5 the colour fit is weighted by alpha by default.  Transparent texels carry no visible colour (and are frequently left black in source art), so fitting them on equal terms pulls block endpoints away from the texels that are actually seen.  Alpha weighting minimises the error that survives alpha compositing instead; fully opaque blocks are unaffected.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

# Refinement passes per quality profile.
PROFILE_ITERATIONS = {
    "super_fast": 0,
    "fast": 1,
    "normal": 2,
    "slow": 4,
    "ultra": 8,
}

DEFAULT_PROFILE = "normal"

# RGB565 palette interpolation coefficients for the 4-colour mode.
_INDEX_COEFFS = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [2.0 / 3.0, 1.0 / 3.0],
        [1.0 / 3.0, 2.0 / 3.0],
    ],
    dtype=np.float32,
)

_CHUNK_BLOCKS = 16384


def _iter_block_chunks(blocks: np.ndarray) -> Iterator[np.ndarray]:
    for start in range(0, blocks.shape[0], _CHUNK_BLOCKS):
        yield blocks[start : start + _CHUNK_BLOCKS]


def _to_blocks(rgba: np.ndarray) -> np.ndarray:
    """Return an ``(nblocks, 16, 4)`` view/copy of an RGBA image."""

    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("DXT input must be an (h, w, 4) RGBA array")
    height, width = rgba.shape[:2]
    block_h = (height + 3) // 4
    block_w = (width + 3) // 4
    pad_h = block_h * 4 - height
    pad_w = block_w * 4 - width
    if pad_h or pad_w:
        rgba = np.pad(rgba, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    blocks = rgba.reshape(block_h, 4, block_w, 4, 4)
    blocks = blocks.transpose(0, 2, 1, 3, 4).reshape(block_h * block_w, 16, 4)
    return blocks


def _quantise_rgb565(colours: np.ndarray) -> np.ndarray:
    """Quantise float RGB colours to packed RGB565 integers."""

    c = np.clip(colours, 0.0, 255.0)
    r = np.floor(c[..., 0] * (31.0 / 255.0) + 0.5).astype(np.int32)
    g = np.floor(c[..., 1] * (63.0 / 255.0) + 0.5).astype(np.int32)
    b = np.floor(c[..., 2] * (31.0 / 255.0) + 0.5).astype(np.int32)
    r = np.clip(r, 0, 31)
    g = np.clip(g, 0, 63)
    b = np.clip(b, 0, 31)
    return (r << 11) | (g << 5) | b


def _expand_rgb565(packed: np.ndarray) -> np.ndarray:
    r5 = (packed >> 11) & 0x1F
    g6 = (packed >> 5) & 0x3F
    b5 = packed & 0x1F
    out = np.empty(packed.shape + (3,), dtype=np.float32)
    out[..., 0] = ((r5 << 3) | (r5 >> 2)).astype(np.float32)
    out[..., 1] = ((g6 << 2) | (g6 >> 4)).astype(np.float32)
    out[..., 2] = ((b5 << 3) | (b5 >> 2)).astype(np.float32)
    return out


def _sort_endpoints(
    c0: np.ndarray, c1: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Ensure c0 > c1 (the 4-colour DXT mode) after quantisation."""

    swap = c0 < c1
    if np.any(swap):
        c0 = c0.copy()
        c1 = c1.copy()
        c0[swap], c1[swap] = c1[swap], c0[swap]
    equal = c0 == c1
    if np.any(equal):
        c0 = c0.copy()
        c1 = c1.copy()
        decrement = equal & (c1 > 0)
        increment = equal & (c1 == 0)
        c1[decrement] -= 1
        c0[increment] += 1
    return c0, c1


def _build_palette(c0: np.ndarray, c1: np.ndarray) -> np.ndarray:
    a = _expand_rgb565(c0)
    b = _expand_rgb565(c1)
    palette = np.empty((c0.shape[0], 4, 3), dtype=np.float32)
    palette[:, 0] = a
    palette[:, 1] = b
    palette[:, 2] = (2.0 * a + b) / 3.0
    palette[:, 3] = (a + 2.0 * b) / 3.0
    return palette


def _nearest_indices(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    diff = rgb[:, :, None, :] - palette[:, None, :, :]
    dist = np.einsum("npkc,npkc->npk", diff, diff)
    return np.argmin(dist, axis=2).astype(np.uint8)


def _colour_indices(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    return _nearest_indices(rgb, palette)


def _ls_endpoints(
    rgb: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted least-squares endpoints for fixed palette indices."""

    coeff = _INDEX_COEFFS[indices]  # (N, 16, 2)
    a = coeff[:, :, 0]
    b = coeff[:, :, 1]
    if weights is None:
        saa = np.einsum("np,np->n", a, a)
        sab = np.einsum("np,np->n", a, b)
        sbb = np.einsum("np,np->n", b, b)
        ta = np.einsum("np,npc->nc", a, rgb)
        tb = np.einsum("np,npc->nc", b, rgb)
    else:
        saa = np.einsum("np,np,np->n", a, a, weights)
        sab = np.einsum("np,np,np->n", a, b, weights)
        sbb = np.einsum("np,np,np->n", b, b, weights)
        ta = np.einsum("np,npc,np->nc", a, rgb, weights)
        tb = np.einsum("np,npc,np->nc", b, rgb, weights)
    det = saa * sbb - sab * sab
    valid = np.abs(det) > 1e-6
    safe_det = np.where(valid, det, 1.0)
    c0 = (sbb[:, None] * ta - sab[:, None] * tb) / safe_det[:, None]
    c1 = (saa[:, None] * tb - sab[:, None] * ta) / safe_det[:, None]
    return c0, c1, valid


def _encode_colour_blocks(
    blocks: np.ndarray,
    iterations: int,
    alpha_weighted: bool = False,
) -> np.ndarray:
    """Encode BC1 colour blocks; returns ``(N, 8)`` bytes."""

    n = blocks.shape[0]
    rgb = blocks[:, :, :3].astype(np.float32)

    weights = None
    weight_sum = None
    if alpha_weighted:
        weights = blocks[:, :, 3].astype(np.float32) / 255.0
        # Fully transparent blocks have no colour to preserve; fall back to
        # an ordinary unweighted fit so their endpoints stay stable.
        weight_sum = weights.sum(axis=1)
        empty = weight_sum < 1e-9
        if np.any(empty):
            weights = weights.copy()
            weights[empty] = 1.0
            weight_sum = weights.sum(axis=1)

    if weights is None:
        mean = rgb.mean(axis=1)
    else:
        # Both values are set together in the alpha_weighted branch above.
        assert weight_sum is not None
        mean = np.einsum("npc,np->nc", rgb, weights) / weight_sum[:, None]
    centred = rgb - mean[:, None, :]

    # Principal axis via a few power iterations, started from the channel
    # with the largest variance.  This is enough to obtain stable endpoints
    # and is considerably cheaper than a batched eigen-decomposition.
    if weights is None:
        cov = np.einsum("npi,npj->nij", centred, centred)
    else:
        cov = np.einsum("npi,npj,np->nij", centred, centred, weights)
    diagonal = np.diagonal(cov, axis1=1, axis2=2)
    start = np.argmax(diagonal, axis=1)
    axis = np.eye(3, dtype=np.float32)[start]
    for _ in range(4):
        axis = np.einsum("nij,nj->ni", cov, axis)
        norm = np.linalg.norm(axis, axis=1, keepdims=True)
        axis /= np.maximum(norm, 1e-12)

    projection = np.einsum("npi,ni->np", centred, axis)
    if weights is None:
        tmin = projection.min(axis=1)
        tmax = projection.max(axis=1)
    else:
        visible = weights > 1e-6
        tmin = np.where(visible, projection, np.inf).min(axis=1)
        tmax = np.where(visible, projection, -np.inf).max(axis=1)
        bad = ~np.isfinite(tmin) | ~np.isfinite(tmax)
        if np.any(bad):
            tmin[bad] = projection[bad].min(axis=1)
            tmax[bad] = projection[bad].max(axis=1)
    c0 = mean + tmin[:, None] * axis
    c1 = mean + tmax[:, None] * axis
    # Give degenerate (solid colour) blocks a stable, non-zero separation.
    p0 = _quantise_rgb565(c0)
    p1 = _quantise_rgb565(c1)
    p0, p1 = _sort_endpoints(p0, p1)

    indices = _colour_indices(rgb, _build_palette(p0, p1))
    for _ in range(max(0, iterations)):
        new_c0, new_c1, valid = _ls_endpoints(rgb, indices, weights)
        if not np.all(valid):
            new_c0 = np.where(valid[:, None], new_c0, c0)
            new_c1 = np.where(valid[:, None], new_c1, c1)
        new_c0 = np.clip(new_c0, 0.0, 255.0)
        new_c1 = np.clip(new_c1, 0.0, 255.0)
        q0 = _quantise_rgb565(new_c0)
        q1 = _quantise_rgb565(new_c1)
        q0, q1 = _sort_endpoints(q0, q1)
        palette = _build_palette(q0, q1)
        indices = _colour_indices(rgb, palette)
        p0, p1 = q0, q1
        c0, c1 = new_c0, new_c1

    if iterations <= 0:
        palette = _build_palette(p0, p1)
        indices = _colour_indices(rgb, palette)

    out = np.empty((n, 8), dtype=np.uint8)
    out[:, 0] = p0 & 0xFF
    out[:, 1] = (p0 >> 8) & 0xFF
    out[:, 2] = p1 & 0xFF
    out[:, 3] = (p1 >> 8) & 0xFF
    bits = np.zeros(n, dtype=np.uint64)
    for i in range(16):
        bits |= indices[:, i].astype(np.uint64) << (2 * i)
    out[:, 4] = bits & 0xFF
    out[:, 5] = (bits >> 8) & 0xFF
    out[:, 6] = (bits >> 16) & 0xFF
    out[:, 7] = (bits >> 24) & 0xFF
    return out, p0, p1, indices


def _alpha_palette8(a0: np.ndarray, a1: np.ndarray) -> np.ndarray:
    palette = np.empty((a0.shape[0], 8), dtype=np.int32)
    palette[:, 0] = a0
    palette[:, 1] = a1
    for i in range(1, 7):
        palette[:, 1 + i] = ((7 - i) * a0 + i * a1) // 7
    return palette


def _alpha_palette6(a0: np.ndarray, a1: np.ndarray) -> np.ndarray:
    palette = np.empty((a0.shape[0], 8), dtype=np.int32)
    palette[:, 0] = a0
    palette[:, 1] = a1
    for i in range(1, 5):
        palette[:, 1 + i] = ((5 - i) * a0 + i * a1) // 5
    palette[:, 6] = 0
    palette[:, 7] = 255
    return palette


def _alpha_error(
    alpha: np.ndarray, palette: np.ndarray, indices: np.ndarray
) -> np.ndarray:
    chosen = np.take_along_axis(palette, indices, axis=1)
    return ((alpha - chosen) ** 2).sum(axis=1)


def _alpha_indices(alpha: np.ndarray, palette: np.ndarray) -> np.ndarray:
    return np.abs(alpha[:, :, None] - palette[:, None, :]).argmin(axis=2)


# Coefficients of palette levels as a function of (a0, a1).
_ALPHA_COEFF8 = np.array(
    [[1.0, 0.0], [0.0, 1.0]]
    + [[(7 - i) / 7.0, i / 7.0] for i in range(1, 7)],
    dtype=np.float32,
)
_ALPHA_COEFF6 = np.array(
    [[1.0, 0.0], [0.0, 1.0]]
    + [[(5 - i) / 5.0, i / 5.0] for i in range(1, 5)]
    + [[0.0, 0.0], [0.0, 0.0]],
    dtype=np.float32,
)


def _refine_alpha(
    a0: np.ndarray,
    a1: np.ndarray,
    alpha: np.ndarray,
    mode: int,
    iterations: int,
):
    """Iteratively least-squares-fit alpha endpoints for a fixed mode."""

    palette_fn = _alpha_palette8 if mode == 8 else _alpha_palette6
    coeff_table = _ALPHA_COEFF8 if mode == 8 else _ALPHA_COEFF6
    palette = palette_fn(a0, a1)
    indices = _alpha_indices(alpha, palette)
    error = _alpha_error(alpha, palette, indices)
    best = (a0.copy(), a1.copy(), palette.copy(), indices.copy(), error.copy())

    af = alpha.astype(np.float32)
    for _ in range(max(0, iterations)):
        coeff = coeff_table[indices]  # (N, 16, 2)
        if mode == 6:
            valid = indices < 6
            coeff = coeff * valid[:, :, None]
            target = af * valid
        else:
            target = af
        ca = coeff[:, :, 0]
        cb = coeff[:, :, 1]
        saa = np.einsum("np,np->n", ca, ca)
        sab = np.einsum("np,np->n", ca, cb)
        sbb = np.einsum("np,np->n", cb, cb)
        ta = np.einsum("np,np->n", ca, target)
        tb = np.einsum("np,np->n", cb, target)
        det = saa * sbb - sab * sab
        valid_det = np.abs(det) > 1e-6
        safe = np.where(valid_det, det, 1.0)
        na0 = (sbb * ta - sab * tb) / safe
        na1 = (saa * tb - sab * ta) / safe
        na0 = np.where(valid_det, na0, a0)
        na1 = np.where(valid_det, na1, a1)
        na0 = np.clip(np.floor(na0 + 0.5), 0, 255).astype(np.int32)
        na1 = np.clip(np.floor(na1 + 0.5), 0, 255).astype(np.int32)
        if mode == 8:
            swap = na0 < na1
            na0, na1 = np.where(swap, na1, na0), np.where(swap, na0, na1)
            equal = na0 == na1
            na1 = np.where(equal & (na0 > 0), na0 - 1, na1)
            na0 = np.where(equal & (na0 == 0), 1, na0)
        else:
            swap = na0 > na1
            na0, na1 = np.where(swap, na1, na0), np.where(swap, na0, na1)

        palette = palette_fn(na0, na1)
        indices = _alpha_indices(alpha, palette)
        error = _alpha_error(alpha, palette, indices)
        better = error < best[4]
        if np.any(better):
            best[0][better] = na0[better]
            best[1][better] = na1[better]
            best[2][better] = palette[better]
            best[3][better] = indices[better]
            best[4][better] = error[better]
        a0, a1 = na0, na1
    return best


def _encode_alpha8(blocks: np.ndarray, iterations: int = 1) -> np.ndarray:
    """BC3 (DXT5) 8-byte alpha blocks, choosing the better of both modes."""

    alpha = blocks[:, :, 3].astype(np.int32)
    n = alpha.shape[0]

    # Mode 8: a0 > a1, six interpolated values between them.
    a0 = alpha.max(axis=1)
    a1 = alpha.min(axis=1)
    equal = a0 == a1
    a0 = np.where(equal & (a0 < 255), a0 + 1, a0)
    a1 = np.where(equal & (a0 == 255), 254, a1)
    a0, a1, palette8, idx8, err8 = _refine_alpha(
        a0, a1, alpha, 8, iterations
    )

    # Mode 6: a0 <= a1, four interpolated values plus transparent/opaque.
    b0 = alpha.min(axis=1)
    b1 = alpha.max(axis=1)
    b0, b1, palette6, idx6, err6 = _refine_alpha(
        b0, b1, alpha, 6, iterations
    )

    use6 = err6 < err8
    out_a0 = np.where(use6, b0, a0)
    out_a1 = np.where(use6, b1, a1)
    out_idx = np.where(use6[:, None], idx6, idx8)

    out = np.empty((n, 8), dtype=np.uint8)
    out[:, 0] = out_a0
    out[:, 1] = out_a1
    bits = np.zeros(n, dtype=np.uint64)
    idx64 = out_idx.astype(np.uint64)
    for i in range(16):
        bits |= idx64[:, i] << (3 * i)
    for i in range(6):
        out[:, 2 + i] = (bits >> (8 * i)) & 0xFF
    return out


def _encode_alpha4(blocks: np.ndarray) -> np.ndarray:
    """DXT3 4-bit alpha blocks (8 bytes each)."""

    alpha = blocks[:, :, 3].astype(np.int32)
    nibbles = ((alpha * 15 + 127) // 255).astype(np.uint64)
    bits = np.zeros(alpha.shape[0], dtype=np.uint64)
    for i in range(16):
        bits |= nibbles[:, i] << (4 * i)
    out = np.empty((alpha.shape[0], 8), dtype=np.uint8)
    for i in range(8):
        out[:, i] = (bits >> (8 * i)) & 0xFF
    return out


def _encode(
    rgba: np.ndarray,
    profile: str,
    kind: str,
    alpha_weighted: bool = True,
) -> bytes:
    if profile not in PROFILE_ITERATIONS:
        raise ValueError(
            f"unknown compression profile {profile!r}; "
            f"choose from {', '.join(PROFILE_ITERATIONS)}"
        )
    iterations = PROFILE_ITERATIONS[profile]
    alpha_iterations = {
        "super_fast": 0,
        "fast": 1,
        "normal": 2,
        "slow": 3,
        "ultra": 4,
    }[profile]
    blocks = _to_blocks(rgba)
    weight_colour = alpha_weighted and kind != "dxt1"
    chunks: list[bytes] = []
    for chunk in _iter_block_chunks(blocks):
        colour, _p0, _p1, _indices = _encode_colour_blocks(
            chunk, iterations, alpha_weighted=weight_colour
        )
        if kind == "dxt1":
            chunks.append(colour.tobytes())
        elif kind == "dxt3":
            alpha = _encode_alpha4(chunk)
            combined = np.concatenate((alpha, colour), axis=1)
            chunks.append(combined.tobytes())
        elif kind == "dxt5":
            alpha = _encode_alpha8(chunk, alpha_iterations)
            combined = np.concatenate((alpha, colour), axis=1)
            chunks.append(combined.tobytes())
        else:
            raise ValueError(f"unknown DXT kind: {kind}")
    return b"".join(chunks)


def encode_dxt1(rgba: np.ndarray, profile: str = DEFAULT_PROFILE) -> bytes:
    """Encode an RGBA image to DXT1/BC1 bytes."""

    return _encode(rgba, profile, "dxt1", alpha_weighted=False)


def encode_dxt3(
    rgba: np.ndarray,
    profile: str = DEFAULT_PROFILE,
    alpha_weighted: bool = True,
) -> bytes:
    """Encode an RGBA image to DXT3/BC2 bytes.

    ``alpha_weighted`` fits colour endpoints to the visible (alpha-scaled)
    texels; pass ``False`` for the original unweighted behaviour.
    """

    return _encode(rgba, profile, "dxt3", alpha_weighted=alpha_weighted)


def encode_dxt5(
    rgba: np.ndarray,
    profile: str = DEFAULT_PROFILE,
    alpha_weighted: bool = True,
) -> bytes:
    """Encode an RGBA image to DXT5/BC3 bytes.

    ``alpha_weighted`` fits colour endpoints to the visible (alpha-scaled)
    texels; pass ``False`` for the original unweighted behaviour.
    """

    return _encode(rgba, profile, "dxt5", alpha_weighted=alpha_weighted)

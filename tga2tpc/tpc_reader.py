"""
BioWare TPC texture reading for :mod:`tga2tpc.tex2tga`.

The layout handling follows xoreos-tools' ``Images::TPC`` loader, including
mip-map chains, greyscale expansion, Xbox-style swizzled BGRA data, animated
texture sheets and cube-map side fixups.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dxt_decode import decode_dxt1, decode_dxt5
from .tpc import get_data_size
from .txi import decode_txi_bytes

_ENCODING_GRAY = 0x01
_ENCODING_RGB = 0x02
_ENCODING_RGBA = 0x04
_ENCODING_SWIZZLED_BGRA = 0x0C

_CUBE_ROTATIONS = (3, 1, 0, 2, 2, 0)


class TPCError(ValueError):
    """Raised when a TPC file cannot be decoded."""


@dataclass
class TPCMip:
    width: int
    height: int
    pixels: np.ndarray


@dataclass
class TPCImage:
    """A decoded TPC texture.

    ``layers`` contains one list of :class:`TPCMip` objects per texture layer.
    Pixels are RGBA ``uint8`` arrays in xoreos-tools' internal orientation.
    """

    width: int
    height: int
    layers: list[list[TPCMip]]
    encoding: int
    format: str
    alpha_blending: float = 1.0
    txi: bytes = b""
    is_animated: bool = False
    is_cube_map: bool = False
    header_width: int = 0
    header_height: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    @property
    def mip_count(self) -> int:
        return len(self.layers[0]) if self.layers else 0

    @property
    def base_layers(self) -> list[np.ndarray]:
        return [layer[0].pixels for layer in self.layers]


def _data_size(format_raw: str, width: int, height: int) -> int:
    # xoreos treats swizzled BGRA as a plain 32-bit BGRA format when it
    # computes mip sizes.  ``get_data_size`` only knows the TPC writer's
    # format names, so handle this alias here.
    if format_raw == "bgra8":
        return width * height * 4
    return get_data_size(format_raw, width, height)


def _pixel_format(encoding: int, uncompressed: bool) -> str:
    if uncompressed:
        if encoding in (_ENCODING_GRAY, _ENCODING_RGB):
            return "rgb8"
        if encoding == _ENCODING_RGBA:
            return "rgba8"
        if encoding == _ENCODING_SWIZZLED_BGRA:
            return "bgra8"
    else:
        if encoding == _ENCODING_RGB:
            return "dxt1"
        if encoding == _ENCODING_RGBA:
            return "dxt5"
    raise TPCError(f"unknown TPC encoding: {encoding}")


def _min_data_size(encoding: int, uncompressed: bool) -> int:
    if uncompressed:
        if encoding == _ENCODING_GRAY:
            return 1
        if encoding == _ENCODING_RGB:
            return 3
        if encoding in (_ENCODING_RGBA, _ENCODING_SWIZZLED_BGRA):
            return 4
    else:
        if encoding == _ENCODING_RGB:
            return 8
        if encoding == _ENCODING_RGBA:
            return 16
    raise TPCError(f"unknown TPC encoding: {encoding}")


def _log2(value: int) -> int:
    if value <= 0:
        return 0
    return int(math.log2(value))


def _deswizzle(
    raw: bytes, width: int, height: int, bpp: int = 4
) -> np.ndarray:
    """Undo the Xbox swizzle used by encoding 0x0C and TXB textures."""

    count = width * height
    source = np.frombuffer(raw, dtype=np.uint8, count=count * bpp).reshape(
        -1, bpp
    )

    x = np.arange(width, dtype=np.int64)[None, :].repeat(height, axis=0)
    y = np.arange(height, dtype=np.int64)[:, None].repeat(width, axis=1)
    width_bits = _log2(width)
    height_bits = _log2(height)
    offset = np.zeros((height, width), dtype=np.int64)
    shift = 0
    while width_bits or height_bits:
        if width_bits:
            offset |= (x & 1) << shift
            x >>= 1
            shift += 1
            width_bits -= 1
        if height_bits:
            offset |= (y & 1) << shift
            y >>= 1
            shift += 1
            height_bits -= 1
    return source[offset]


def _decode_raw(
    raw: bytes, width: int, height: int, encoding: int
) -> np.ndarray:
    count = width * height
    if encoding == _ENCODING_GRAY:
        values = np.frombuffer(raw, dtype=np.uint8, count=count)
        rgba = np.empty((count, 4), dtype=np.uint8)
        rgba[:, 0] = values
        rgba[:, 1] = values
        rgba[:, 2] = values
        rgba[:, 3] = 255
        return rgba.reshape(height, width, 4)

    if encoding == _ENCODING_RGB:
        values = np.frombuffer(raw, dtype=np.uint8, count=count * 3).reshape(
            count, 3
        )
        rgba = np.empty((count, 4), dtype=np.uint8)
        rgba[:, :3] = values
        rgba[:, 3] = 255
        return rgba.reshape(height, width, 4)

    if encoding == _ENCODING_RGBA:
        return np.frombuffer(
            raw, dtype=np.uint8, count=count * 4
        ).reshape(height, width, 4).copy()

    if encoding == _ENCODING_SWIZZLED_BGRA:
        width_pot = (width & (width - 1)) == 0
        if width_pot:
            bgra = _deswizzle(raw, width, height)
        else:
            bgra = np.frombuffer(
                raw, dtype=np.uint8, count=count * 4
            ).reshape(height, width, 4)
        rgba = np.empty_like(bgra)
        rgba[:, :, 0] = bgra[:, :, 2]
        rgba[:, :, 1] = bgra[:, :, 1]
        rgba[:, :, 2] = bgra[:, :, 0]
        rgba[:, :, 3] = bgra[:, :, 3]
        return rgba

    raise TPCError(f"unknown uncompressed TPC encoding: {encoding}")


def _decode_level(
    raw: bytes,
    width: int,
    height: int,
    encoding: int,
    format_raw: str,
    uncompressed: bool,
) -> np.ndarray:
    if not uncompressed:
        if format_raw == "dxt1":
            return decode_dxt1(raw, width, height)
        if format_raw == "dxt5":
            return decode_dxt5(raw, width, height)
    return _decode_raw(raw, width, height, encoding)


def _txi_animation(txi: bytes) -> tuple[bool, int, int, int]:
    """Return (animated, numx, numy, fps) from embedded TXI data."""

    if not txi:
        return False, 0, 0, 0
    text = decode_txi_bytes(txi)

    def search_int(keyword: str) -> int:
        match = re.search(
            rf"^\s*{keyword}\s+(\d+)", text, re.IGNORECASE | re.MULTILINE
        )
        return int(match.group(1)) if match else 0

    def search_float(keyword: str) -> float:
        match = re.search(
            rf"^\s*{keyword}\s+([-+0-9.eE]+)",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        if not match:
            return 0.0
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0

    procedure = re.search(
        r"^\s*proceduretype\s+(\S+)", text, re.IGNORECASE | re.MULTILINE
    )
    is_cycle = bool(procedure and procedure.group(1).lower() == "cycle")
    numx = search_int("numx")
    numy = search_int("numy")
    fps = search_float("fps")
    return is_cycle and numx > 0 and numy > 0 and fps != 0.0, numx, numy, fps


def decode_tpc(data: bytes) -> TPCImage:
    """Decode TPC bytes into a :class:`TPCImage`."""

    if len(data) < 128:
        raise TPCError("not enough data to contain a TPC header")

    (
        data_size,
        alpha_blending,
        header_width,
        header_height,
        encoding,
        header_mip_count,
    ) = struct.unpack_from("<IfHHBB", data, 0)

    if header_width >= 0x8000 or header_height >= 0x8000:
        raise TPCError(
            f"unsupported image dimensions ({header_width}x{header_height})"
        )
    if header_width == 0 or header_height == 0:
        raise TPCError("invalid TPC image dimensions")
    if header_mip_count == 0:
        raise TPCError("TPC contains no mip levels")

    uncompressed = data_size == 0
    width = header_width
    height = header_height
    layer_count = 1
    is_cube_map = False

    if uncompressed:
        if encoding == _ENCODING_GRAY:
            data_size = width * height
        elif encoding == _ENCODING_RGB:
            data_size = width * height * 3
        elif encoding in (_ENCODING_RGBA, _ENCODING_SWIZZLED_BGRA):
            data_size = width * height * 4
        else:
            raise TPCError(f"unknown TPC encoding: {encoding}")
    else:
        # A cube map is encoded as six stacked square sides: height == 6*width.
        if height != 0 and width != 0 and height // width == 6:
            height //= 6
            layer_count = 6
            is_cube_map = True

    min_data_size = _min_data_size(encoding, uncompressed)
    format_raw = _pixel_format(encoding, uncompressed)

    complete_data_size = data_size
    level_width, level_height = width, height
    for _ in range(1, header_mip_count):
        level_width >>= 1
        level_height >>= 1
        level_width = max(level_width, 1)
        level_height = max(level_height, 1)
        complete_data_size += _data_size(
            format_raw, level_width, level_height
        )
    complete_data_size *= layer_count

    txi_offset = 128 + complete_data_size
    if txi_offset > len(data):
        raise TPCError("TPC mip data extends beyond the end of the file")
    txi = data[txi_offset:]

    is_animated, numx, numy, _fps = _txi_animation(txi)
    mip_count = header_mip_count
    if is_animated:
        layer_count = numx * numy
        width //= numx
        height //= numy
        data_size //= layer_count
        if width < 1 or height < 1:
            raise TPCError("animated TPC frame dimensions are invalid")
        # xoreos recomputes the mip count for the per-frame dimensions.
        mip_count = 0
        level_width, level_height = width, height
        while level_width > 0 and level_height > 0:
            level_width //= 2
            level_height //= 2
            mip_count += 1
        if mip_count == 0:
            raise TPCError("animated TPC has no mip levels")

    if not uncompressed:
        if encoding == _ENCODING_RGB:
            if data_size != (width * height) // 2 and not is_animated:
                raise TPCError(
                    "invalid data size for a DXT1 TPC texture of "
                    f"{width}x{height}"
                )
        elif encoding == _ENCODING_RGBA:
            if data_size != width * height and not is_animated:
                raise TPCError(
                    "invalid data size for a DXT5 TPC texture of "
                    f"{width}x{height}"
                )
        else:
            raise TPCError(f"unknown compressed TPC encoding: {encoding}")

    # xoreos seeks back to the start of the pixel data and reads the levels
    # sequentially, using the header's dataSize for the base level.
    position = 128
    full_data_size = len(data) - 128
    flat_levels: list[TPCMip] = []
    layer_lengths: list[int] = []

    for _layer in range(layer_count):
        level_width, level_height = width, height
        layer_size = (
            _data_size(format_raw, level_width, level_height)
            if is_animated
            else data_size
        )
        count_before = len(flat_levels)

        for _ in range(mip_count):
            mip_width = max(level_width, 1)
            mip_height = max(level_height, 1)
            mip_size = max(layer_size, min_data_size)
            mip_data_size = _data_size(
                format_raw, mip_width, mip_height
            )
            if full_data_size < mip_size or mip_size < mip_data_size:
                break
            if position + mip_size > len(data):
                raise TPCError("truncated TPC pixel data")

            raw = data[position : position + mip_size]
            position += mip_size
            full_data_size -= mip_size

            pixels = _decode_level(
                raw,
                mip_width,
                mip_height,
                encoding,
                format_raw,
                uncompressed,
            )
            flat_levels.append(
                TPCMip(width=mip_width, height=mip_height, pixels=pixels)
            )

            level_width >>= 1
            level_height >>= 1
            layer_size = _data_size(
                format_raw, level_width, level_height
            )
            if level_width < 1 and level_height < 1:
                break

        layer_lengths.append(len(flat_levels) - count_before)

    if not flat_levels:
        raise TPCError("TPC contains no readable mip levels")
    if any(length != layer_lengths[0] for length in layer_lengths):
        raise TPCError("TPC layers contain different numbers of mip levels")

    per_layer = layer_lengths[0]
    layers = [
        flat_levels[index * per_layer : (index + 1) * per_layer]
        for index in range(layer_count)
    ]

    if is_cube_map:
        _fixup_cube_map(layers)

    return TPCImage(
        width=layers[0][0].width,
        height=layers[0][0].height,
        layers=layers,
        encoding=encoding,
        format=format_raw,
        alpha_blending=float(alpha_blending),
        txi=txi,
        is_animated=is_animated,
        is_cube_map=is_cube_map,
        header_width=header_width,
        header_height=header_height,
    )


def _fixup_cube_map(layers: list[list[TPCMip]]) -> None:
    """Rotate and reorder cube-map sides exactly like xoreos-tools."""

    mip_count = len(layers[0])
    for layer in layers:
        if len(layer) != mip_count:
            raise TPCError("cube map layer mip count mismatch")

    for mip_index in range(mip_count):
        reference = layers[0][mip_index]
        for layer in layers:
            current = layer[mip_index]
            if (
                current.width != reference.width
                or current.height != reference.height
            ):
                raise TPCError("cube map layer dimensions mismatch")

        for layer_index, steps in enumerate(_CUBE_ROTATIONS):
            mip = layers[layer_index][mip_index]
            if steps:
                mip.pixels = np.ascontiguousarray(
                    np.rot90(mip.pixels, -steps, axes=(0, 1))
                )

        layers[0][mip_index].pixels, layers[1][mip_index].pixels = (
            layers[1][mip_index].pixels,
            layers[0][mip_index].pixels,
        )


def load_tpc(path: str | Path) -> TPCImage:
    """Load and decode a TPC file from disk."""

    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise TPCError(f"unable to read {path}: {exc}") from exc
    try:
        return decode_tpc(data)
    except TPCError as exc:
        raise TPCError(f"{path}: {exc}") from exc

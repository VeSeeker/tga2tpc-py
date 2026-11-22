"""
DDS texture decoding for :mod:`tga2tpc.tex2tga`.

Both the standard Microsoft DirectDraw Surface format and BioWare's simpler
DDS variant (used by Neverwinter Nights) are supported.  The parser mirrors
xoreos-tools' ``Images::DDS`` loader, including its handling of uncompressed
pixel formats and mip-map chains.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dxt_decode import decode_dxt1, decode_dxt3, decode_dxt5
from .tpc import get_data_size

_DDS_MAGIC = b"DDS "
_HEADER_FLAGS_HAS_MIPMAPS = 0x00020000
_PIXEL_FLAGS_HAS_ALPHA = 0x00000001
_PIXEL_FLAGS_HAS_FOURCC = 0x00000004
_PIXEL_FLAGS_IS_INDEXED = 0x00000020
_PIXEL_FLAGS_IS_RGB = 0x00000040


class DDSError(ValueError):
    """Raised when a DDS file cannot be decoded."""


@dataclass
class DDSImage:
    """A decoded DDS texture.

    ``pixels`` and ``mipmaps`` are RGBA ``uint8`` arrays in xoreos-tools'
    internal orientation.  DDS files contain a single layer.
    """

    width: int
    height: int
    pixels: np.ndarray
    mipmaps: list[np.ndarray]
    format: str
    is_bioware: bool = False

    @property
    def layer_count(self) -> int:
        return 1

    @property
    def mip_count(self) -> int:
        return len(self.mipmaps)

    @property
    def base_layers(self) -> list[np.ndarray]:
        return [self.pixels]

    @property
    def txi(self) -> bytes:
        return b""


def _unpack_standard_header(data: bytes) -> tuple[int, int, int, str, bool]:
    if len(data) < 128:
        raise DDSError("not enough data to contain a DDS header")
    if struct.unpack_from("<I", data, 4)[0] != 124:
        raise DDSError("DDS header size is not 124")

    flags = struct.unpack_from("<I", data, 8)[0]
    height, width = struct.unpack_from("<II", data, 12)
    if width >= 0x8000 or height >= 0x8000:
        raise DDSError(f"unsupported image dimensions ({width}x{height})")

    mip_count = struct.unpack_from("<I", data, 28)[0]
    if not (flags & _HEADER_FLAGS_HAS_MIPMAPS):
        mip_count = 1
    if mip_count < 1:
        raise DDSError("DDS contains no mip levels")

    pixel_flags, fourcc = struct.unpack_from("<I4s", data, 80)
    bit_count, r_mask, g_mask, b_mask, a_mask = struct.unpack_from(
        "<IIIII", data, 88
    )

    is_4444 = False
    if (pixel_flags & _PIXEL_FLAGS_HAS_FOURCC) and fourcc == b"DXT1":
        format_raw = "dxt1"
    elif (pixel_flags & _PIXEL_FLAGS_HAS_FOURCC) and fourcc == b"DXT3":
        format_raw = "dxt3"
    elif (pixel_flags & _PIXEL_FLAGS_HAS_FOURCC) and fourcc == b"DXT5":
        format_raw = "dxt5"
    elif (
        (pixel_flags & _PIXEL_FLAGS_IS_RGB)
        and (pixel_flags & _PIXEL_FLAGS_HAS_ALPHA)
        and bit_count == 32
        and (r_mask, g_mask, b_mask, a_mask)
        == (0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000)
    ):
        format_raw = "bgra8"
    elif (
        (pixel_flags & _PIXEL_FLAGS_IS_RGB)
        and not (pixel_flags & _PIXEL_FLAGS_HAS_ALPHA)
        and bit_count == 24
        and (r_mask, g_mask, b_mask)
        == (0x00FF0000, 0x0000FF00, 0x000000FF)
    ):
        format_raw = "bgr8"
    elif (
        (pixel_flags & _PIXEL_FLAGS_IS_RGB)
        and (pixel_flags & _PIXEL_FLAGS_HAS_ALPHA)
        and bit_count == 16
        and (r_mask, g_mask, b_mask, a_mask)
        == (0x00007C00, 0x000003E0, 0x0000001F, 0x00008000)
    ):
        format_raw = "a1r5g5b5"
    elif (
        (pixel_flags & _PIXEL_FLAGS_IS_RGB)
        and not (pixel_flags & _PIXEL_FLAGS_HAS_ALPHA)
        and bit_count == 16
        and (r_mask, g_mask, b_mask)
        == (0x0000F800, 0x000007E0, 0x0000001F)
    ):
        format_raw = "r5g6b5"
    elif (
        (pixel_flags & _PIXEL_FLAGS_IS_RGB)
        and (pixel_flags & _PIXEL_FLAGS_HAS_ALPHA)
        and bit_count == 16
        and (r_mask, g_mask, b_mask, a_mask)
        == (0x00000F00, 0x000000F0, 0x0000000F, 0x0000F000)
    ):
        format_raw = "4444"
        is_4444 = True
    elif pixel_flags & _PIXEL_FLAGS_IS_INDEXED:
        raise DDSError("indexed/paletted DDS files are not supported")
    else:
        raise DDSError(
            "unknown DDS pixel format "
            f"(flags={pixel_flags:#x}, fourCC={fourcc!r}, "
            f"bits={bit_count}, masks={r_mask:#x},{g_mask:#x},"
            f"{b_mask:#x},{a_mask:#x})"
        )

    return width, height, mip_count, format_raw, is_4444


def _read_raw_mip(
    data: bytes, offset: int, width: int, height: int, format_raw: str
) -> tuple[np.ndarray, int]:
    """Decode one uncompressed mip level; returns (RGBA, bytes consumed)."""

    count = width * height
    if format_raw in ("bgra8", "bgr8"):
        bpp = 4 if format_raw == "bgra8" else 3
        size = count * bpp
        if offset + size > len(data):
            raise DDSError("truncated DDS pixel data")
        raw = np.frombuffer(data, dtype=np.uint8, count=size, offset=offset)
        raw = raw.reshape(height, width, bpp)
        rgba = np.empty((height, width, 4), dtype=np.uint8)
        rgba[:, :, 0] = raw[:, :, 2]
        rgba[:, :, 1] = raw[:, :, 1]
        rgba[:, :, 2] = raw[:, :, 0]
        rgba[:, :, 3] = raw[:, :, 3] if bpp == 4 else 255
        return rgba, size

    if format_raw in ("a1r5g5b5", "r5g6b5"):
        size = count * 2
        if offset + size > len(data):
            raise DDSError("truncated DDS pixel data")
        values = np.frombuffer(data, dtype="<u2", count=count, offset=offset)
        rgba = np.empty((count, 4), dtype=np.uint8)
        if format_raw == "a1r5g5b5":
            rgba[:, 0] = ((values >> 10) & 0x1F).astype(np.uint8)
            rgba[:, 1] = ((values >> 5) & 0x1F).astype(np.uint8)
            rgba[:, 2] = (values & 0x1F).astype(np.uint8)
            rgba[:, 3] = np.where(values & 0x8000, 255, 0).astype(np.uint8)
        else:
            rgba[:, 0] = ((values >> 11) & 0x1F).astype(np.uint8)
            rgba[:, 1] = ((values >> 5) & 0x3F).astype(np.uint8)
            rgba[:, 2] = (values & 0x1F).astype(np.uint8)
            rgba[:, 3] = 255
        return rgba.reshape(height, width, 4), size

    if format_raw == "4444":
        # xoreos reads 16-bit 4:4:4:4 data but allocates four bytes per pixel.
        size = count * 2
        if offset + size > len(data):
            raise DDSError("truncated DDS pixel data")
        values = np.frombuffer(data, dtype="<u2", count=count, offset=offset)
        rgba = np.empty((count, 4), dtype=np.uint8)
        rgba[:, 0] = ((values >> 8) & 0xF).astype(np.uint8) << 4
        rgba[:, 1] = ((values >> 4) & 0xF).astype(np.uint8) << 4
        rgba[:, 2] = (values & 0xF).astype(np.uint8) << 4
        rgba[:, 3] = ((values >> 12) & 0xF).astype(np.uint8) << 4
        return rgba.reshape(height, width, 4), size

    raise DDSError(f"unsupported uncompressed DDS format: {format_raw}")


def _decode_mip(
    data: bytes,
    offset: int,
    width: int,
    height: int,
    format_raw: str,
    is_4444: bool,
) -> tuple[np.ndarray, int]:
    if format_raw == "dxt1":
        size = get_data_size("dxt1", width, height)
        if offset + size > len(data):
            raise DDSError("truncated DDS pixel data")
        return decode_dxt1(data[offset : offset + size], width, height), size
    if format_raw == "dxt3":
        size = get_data_size("dxt3", width, height)
        if offset + size > len(data):
            raise DDSError("truncated DDS pixel data")
        return decode_dxt3(data[offset : offset + size], width, height), size
    if format_raw == "dxt5":
        size = get_data_size("dxt5", width, height)
        if offset + size > len(data):
            raise DDSError("truncated DDS pixel data")
        return decode_dxt5(data[offset : offset + size], width, height), size
    return _read_raw_mip(data, offset, width, height, format_raw)


def _decode_standard(data: bytes) -> DDSImage:
    width, height, mip_count, format_raw, is_4444 = _unpack_standard_header(
        data
    )
    levels: list[np.ndarray] = []
    offset = 128
    level_width, level_height = width, height

    for _ in range(mip_count):
        level_width = max(level_width, 1)
        level_height = max(level_height, 1)
        pixels, consumed = _decode_mip(
            data,
            offset,
            level_width,
            level_height,
            format_raw,
            is_4444,
        )
        levels.append(pixels)
        offset += consumed
        level_width >>= 1
        level_height >>= 1

    return DDSImage(
        width=width,
        height=height,
        pixels=levels[0],
        mipmaps=levels,
        format=format_raw,
    )


def _decode_bioware(data: bytes) -> DDSImage:
    if len(data) < 20:
        raise DDSError("not enough data to contain a BioWare DDS header")

    width, height, bytes_per_pixel, data_size = struct.unpack_from(
        "<IIII", data, 0
    )
    if width >= 0x8000 or height >= 0x8000:
        raise DDSError(f"unsupported image dimensions ({width}x{height})")
    if width == 0 or height == 0:
        raise DDSError("invalid DDS image dimensions")
    if width & (width - 1) or height & (height - 1):
        raise DDSError("BioWare DDS width and height must be powers of two")

    if bytes_per_pixel == 3:
        format_raw = "dxt1"
        expected = (width * height) // 2
    elif bytes_per_pixel == 4:
        format_raw = "dxt5"
        expected = width * height
    else:
        raise DDSError(
            f"unsupported BioWare DDS bytes-per-pixel value: {bytes_per_pixel}"
        )
    if data_size != expected:
        raise DDSError(
            f"invalid BioWare DDS data size ({width}x{height}x"
            f"{bytes_per_pixel}: {data_size})"
        )

    # Header: width, height, bytes-per-pixel, data size and a float.
    offset = 20
    remaining = len(data) - offset
    levels: list[np.ndarray] = []
    level_width, level_height = width, height

    while True:
        level_width = max(level_width, 1)
        level_height = max(level_height, 1)
        size = get_data_size(format_raw, level_width, level_height)
        if remaining < size:
            break
        pixels, consumed = _decode_mip(
            data,
            offset,
            level_width,
            level_height,
            format_raw,
            False,
        )
        levels.append(pixels)
        offset += consumed
        remaining -= consumed
        level_width >>= 1
        level_height >>= 1
        if level_width < 1 or level_height < 1:
            break

    if not levels:
        raise DDSError("BioWare DDS contains no mip levels")

    return DDSImage(
        width=width,
        height=height,
        pixels=levels[0],
        mipmaps=levels,
        format=format_raw,
        is_bioware=True,
    )


def decode_dds(data: bytes) -> DDSImage:
    """Decode DDS bytes into a :class:`DDSImage`."""

    if data[:4] == _DDS_MAGIC:
        return _decode_standard(data)
    return _decode_bioware(data)


def load_dds(path: str | Path) -> DDSImage:
    """Load and decode a DDS file from disk."""

    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise DDSError(f"unable to read {path}: {exc}") from exc
    try:
        return decode_dds(data)
    except DDSError as exc:
        raise DDSError(f"{path}: {exc}") from exc

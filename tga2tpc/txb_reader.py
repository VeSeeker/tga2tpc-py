"""
Xbox TXB texture reading for :mod:`tga2tpc.tex2tga`.

TXB is another BioWare texture container used by the Xbox versions of the
Odyssey engine games.  The loader mirrors xoreos-tools' ``Images::TXB``:
uncompressed BGRA and greyscale data is de-swizzled when the width is a power
of two, and DXT1/DXT5 data is decoded with the shared S3TC routines.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .dxt_decode import decode_dxt1, decode_dxt5
from .tpc_reader import _deswizzle
from .tpc import get_data_size

_ENCODING_BGRA = 0x04
_ENCODING_GRAY = 0x09
_ENCODING_DXT1 = 0x0A
_ENCODING_DXT5 = 0x0C

_FORMATS = {
    _ENCODING_BGRA: "bgra8",
    _ENCODING_GRAY: "rgb8",
    _ENCODING_DXT1: "dxt1",
    _ENCODING_DXT5: "dxt5",
}


class TXBError(ValueError):
    """Raised when a TXB file cannot be decoded."""


@dataclass
class TXBImage:
    """A decoded TXB texture (a single layer with a mip chain)."""

    width: int
    height: int
    pixels: np.ndarray
    mipmaps: list[np.ndarray]
    format: str
    txi: bytes = b""

    @property
    def layer_count(self) -> int:
        return 1

    @property
    def mip_count(self) -> int:
        return len(self.mipmaps)

    @property
    def base_layers(self) -> list[np.ndarray]:
        return [self.pixels]


def _mip_size(
    encoding: int, format_raw: str, width: int, height: int
) -> int:
    if encoding == _ENCODING_GRAY:
        return width * height
    if encoding == _ENCODING_BGRA:
        return width * height * 4
    return get_data_size(format_raw, width, height)


def _decode_mip(
    raw: bytes,
    width: int,
    height: int,
    encoding: int,
    format_raw: str,
) -> np.ndarray:
    if encoding == _ENCODING_DXT1:
        return decode_dxt1(raw, width, height)
    if encoding == _ENCODING_DXT5:
        return decode_dxt5(raw, width, height)

    width_pot = (width & (width - 1)) == 0

    if encoding == _ENCODING_BGRA:
        if width_pot:
            bgra = _deswizzle(raw, width, height, 4)
        else:
            bgra = np.frombuffer(
                raw, dtype=np.uint8, count=width * height * 4
            ).reshape(height, width, 4)
        rgba = np.empty_like(bgra)
        rgba[:, :, 0] = bgra[:, :, 2]
        rgba[:, :, 1] = bgra[:, :, 1]
        rgba[:, :, 2] = bgra[:, :, 0]
        rgba[:, :, 3] = bgra[:, :, 3]
        return rgba

    # Greyscale data is expanded to BGR before xoreos de-swizzles it.  Doing
    # the de-swizzle on the single grey channel first is equivalent and avoids
    # materialising the intermediate triple-byte image.
    if width_pot:
        grey = _deswizzle(raw, width, height, 1)[:, :, 0]
    else:
        grey = np.frombuffer(raw, dtype=np.uint8, count=width * height).reshape(
            height, width
        )
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[:, :, 0] = grey
    rgba[:, :, 1] = grey
    rgba[:, :, 2] = grey
    rgba[:, :, 3] = 255
    return rgba


def decode_txb(data: bytes) -> TXBImage:
    """Decode TXB bytes into a :class:`TXBImage`."""

    if len(data) < 128:
        raise TXBError("not enough data to contain a TXB header")

    data_size, _unknown_float, width, height, encoding, mip_count = (
        struct.unpack_from("<IfHHBB", data, 0)
    )

    if width >= 0x8000 or height >= 0x8000:
        raise TXBError(f"unsupported image dimensions ({width}x{height})")
    if width == 0 or height == 0:
        raise TXBError("invalid TXB image dimensions")
    if encoding not in _FORMATS:
        raise TXBError(f"unknown TXB encoding: {encoding:#04x}")
    if mip_count == 0:
        raise TXBError("TXB contains no mip levels")

    format_raw = _FORMATS[encoding]
    full_image_size = _mip_size(encoding, format_raw, width, height)
    if data_size < full_image_size:
        raise TXBError("TXB image would not fit into its pixel data")

    # xoreos clamps each dimension at one pixel while walking the mip chain.
    sizes: list[tuple[int, int, int]] = []
    mip_width, mip_height = width, height
    for _ in range(mip_count):
        sizes.append(
            (
                mip_width,
                mip_height,
                _mip_size(encoding, format_raw, mip_width, mip_height),
            )
        )
        if mip_width > 1:
            mip_width >>= 1
        if mip_height > 1:
            mip_height >>= 1

    offset = 128
    levels: list[np.ndarray] = []
    for mip_width, mip_height, size in sizes:
        if offset + size > len(data):
            raise TXBError("truncated TXB pixel data")
        raw = data[offset : offset + size]
        offset += size
        levels.append(
            _decode_mip(raw, mip_width, mip_height, encoding, format_raw)
        )

    if not levels:
        raise TXBError("TXB contains no readable mip levels")

    txi_offset = 128 + data_size
    txi = data[txi_offset:] if txi_offset <= len(data) else b""

    return TXBImage(
        width=width,
        height=height,
        pixels=levels[0],
        mipmaps=levels,
        format=format_raw,
        txi=txi,
    )


def load_txb(path: str | Path) -> TXBImage:
    """Load and decode a TXB file from disk."""

    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise TXBError(f"unable to read {path}: {exc}") from exc
    try:
        return decode_txb(data)
    except TXBError as exc:
        raise TXBError(f"{path}: {exc}") from exc

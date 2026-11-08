"""
BioWare TPC container writing.

TPC layout (all values little-endian):

==========  =========================================================
offset      contents
==========  =========================================================
0           uint32 data size / compressed-payload indicator
4           float32 alpha blending
8           uint16 width
10          uint16 height
12          uint8 encoding (1=greyscale, 2=RGB/DXT1, 4=RGBA/DXT3/5;
            most TPC consumers treat encoding 4 as DXT5)
13          uint8 mip count
14..127     114 reserved bytes
128...      mip data (layers are stored consecutively), then TXI text
==========  =========================================================
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dxt import encode_dxt1, encode_dxt3, encode_dxt5
from .mipmaps import writer_mip_levels

ENCODING_GRAY = 1
ENCODING_RGB = 2
ENCODING_RGBA = 4

FORMAT_R8 = "r8"
FORMAT_RGB8 = "rgb8"
FORMAT_RGBA8 = "rgba8"
FORMAT_DXT1 = "dxt1"
FORMAT_DXT3 = "dxt3"
FORMAT_DXT5 = "dxt5"

COMPRESSED_FORMATS = frozenset({FORMAT_DXT1, FORMAT_DXT3, FORMAT_DXT5})

FORMAT_ENCODINGS = {
    FORMAT_R8: ENCODING_GRAY,
    FORMAT_RGB8: ENCODING_RGB,
    FORMAT_RGBA8: ENCODING_RGBA,
    FORMAT_DXT1: ENCODING_RGB,
    FORMAT_DXT3: ENCODING_RGBA,
    FORMAT_DXT5: ENCODING_RGBA,
}

FORMAT_NAMES = {
    FORMAT_R8: "Raw",
    FORMAT_RGB8: "Raw",
    FORMAT_RGBA8: "Raw",
    FORMAT_DXT1: "DXT1",
    FORMAT_DXT3: "DXT3",
    FORMAT_DXT5: "DXT5",
}


def is_compressed(format_raw: str) -> bool:
    return format_raw in COMPRESSED_FORMATS


def get_data_size(format_raw: str, width: int, height: int) -> int:
    """Bytes required for one mip level in the given format."""

    if format_raw == FORMAT_RGB8:
        return width * height * 3
    if format_raw == FORMAT_R8:
        return width * height
    if format_raw == FORMAT_RGBA8:
        return width * height * 4
    if format_raw == FORMAT_DXT1:
        return max(8, ((width + 3) // 4) * ((height + 3) // 4) * 8)
    if format_raw in (FORMAT_DXT3, FORMAT_DXT5):
        return max(16, ((width + 3) // 4) * ((height + 3) // 4) * 16)
    return 0


def header_mip_count(width: int, height: int) -> int:
    """The (truncated) mip count the JavaScript writer puts in the header."""

    maximum = max(int(width), int(height), 1)
    return min(255, int(math.log2(maximum) + 1))


def compute_data_size(
    format_raw: str,
    layer_width: int,
    layer_height: int,
    layer_count: int,
    animated: bool,
    cube: bool,
) -> int:
    """Compute the header ``dataSize`` field using the original rules.

    Plain and cube textures are represented by the size of one base mip;
    animated textures by the sum of all mips of one layer times the layer
    count.  Uncompressed data uses zero.
    """

    if not is_compressed(format_raw):
        return 0
    if animated:
        one_layer = sum(
            get_data_size(format_raw, w, h)
            for w, h in writer_mip_levels(layer_width, layer_height)
        )
        return one_layer * layer_count
    if cube:
        return get_data_size(format_raw, layer_width, layer_width)
    return get_data_size(format_raw, layer_width, layer_height)


@dataclass
class TPCLayer:
    width: int
    height: int
    mipmaps: list[np.ndarray]


@dataclass
class TPCTexture:
    width: int
    height: int
    format_raw: str
    encoding: int
    mip_count: int
    data_size: int
    layers: list[TPCLayer]
    alpha_blending: float = 1.0
    txi: bytes = b""
    animated: bool = False
    cube: bool = False
    source: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def format_name(self) -> str:
        return FORMAT_NAMES.get(self.format_raw, self.format_raw)


def _orient_for_tpc(
    mipmap: np.ndarray, flip_x: bool, flip_y: bool
) -> np.ndarray:
    """Apply the user's flips and the mandatory canvas-to-OpenGL flip."""

    result = mipmap
    if flip_x:
        result = result[:, ::-1]
    if not flip_y:
        result = result[::-1]
    return np.ascontiguousarray(result)


def _encode_level(
    pixels: np.ndarray,
    format_raw: str,
    profile: str,
    alpha_weighted: bool = True,
) -> bytes:
    if format_raw == FORMAT_R8:
        return np.ascontiguousarray(pixels[:, :, 0]).tobytes()
    if format_raw == FORMAT_RGB8:
        return np.ascontiguousarray(pixels[:, :, :3]).tobytes()
    if format_raw == FORMAT_RGBA8:
        return np.ascontiguousarray(pixels).tobytes()
    if format_raw == FORMAT_DXT1:
        return encode_dxt1(pixels, profile)
    if format_raw == FORMAT_DXT3:
        return encode_dxt3(pixels, profile, alpha_weighted=alpha_weighted)
    if format_raw == FORMAT_DXT5:
        return encode_dxt5(pixels, profile, alpha_weighted=alpha_weighted)
    raise ValueError(f"unsupported output format: {format_raw}")


def build_header(texture: TPCTexture) -> bytes:
    header = bytearray(128)
    struct.pack_into(
        "<IfHHBB",
        header,
        0,
        int(texture.data_size),
        float(texture.alpha_blending),
        int(texture.width),
        int(texture.height),
        int(texture.encoding),
        int(texture.mip_count),
    )
    return bytes(header)


def build_tpc_bytes(
    texture: TPCTexture,
    flip_x: bool = False,
    flip_y: bool = False,
    profile: str = "normal",
    alpha_weighted: bool = True,
) -> bytes:
    """Serialise a prepared texture to TPC bytes."""

    out = bytearray(build_header(texture))
    gray = texture.format_raw == FORMAT_R8

    for layer in texture.layers:
        levels = writer_mip_levels(layer.width, layer.height)
        if gray:
            levels = levels[:1]
        if len(layer.mipmaps) < len(levels):
            raise ValueError(
                "internal error: not enough generated mip levels "
                f"({len(layer.mipmaps)} < {len(levels)})"
            )
        for index, (width, height) in enumerate(levels):
            mipmap = layer.mipmaps[index]
            if mipmap.shape[0] != height or mipmap.shape[1] != width:
                raise ValueError(
                    "internal error: mip level dimensions do not match "
                    f"(expected {width}x{height}, got "
                    f"{mipmap.shape[1]}x{mipmap.shape[0]})"
                )
            pixels = _orient_for_tpc(mipmap, flip_x, flip_y)
            out.extend(
                _encode_level(
                    pixels,
                    texture.format_raw,
                    profile,
                    alpha_weighted=alpha_weighted,
                )
            )

    out.extend(texture.txi)
    return bytes(out)


def write_tpc(
    texture: TPCTexture,
    path: str | Path,
    flip_x: bool = False,
    flip_y: bool = False,
    profile: str = "normal",
    alpha_weighted: bool = True,
) -> int:
    """Write a prepared texture and return the number of bytes written."""

    data = build_tpc_bytes(
        texture,
        flip_x=flip_x,
        flip_y=flip_y,
        profile=profile,
        alpha_weighted=alpha_weighted,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)

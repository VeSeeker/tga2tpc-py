"""TPC header, sizing and serialisation tests."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tga2tpc.mipmaps import generate_mipmaps
from tga2tpc.tpc import (
    ENCODING_GRAY,
    ENCODING_RGBA,
    FORMAT_DXT1,
    FORMAT_DXT3,
    FORMAT_DXT5,
    FORMAT_ENCODINGS,
    FORMAT_R8,
    FORMAT_RGB8,
    FORMAT_RGBA8,
    TPCLayer,
    TPCTexture,
    build_header,
    build_tpc_bytes,
    compute_data_size,
    get_data_size,
    header_mip_count,
    is_compressed,
    write_tpc,
)


def _texture(
    format_raw: str,
    layers: list[TPCLayer],
    width: int,
    height: int,
    *,
    data_size: int = 0,
    mip_count: int | None = None,
    alpha_blending: float = 1.0,
    txi: bytes = b"",
) -> TPCTexture:
    return TPCTexture(
        width=width,
        height=height,
        format_raw=format_raw,
        encoding=FORMAT_ENCODINGS[format_raw],
        mip_count=mip_count
        if mip_count is not None
        else header_mip_count(width, height),
        data_size=data_size,
        layers=layers,
        alpha_blending=alpha_blending,
        txi=txi,
    )


def _rgba_layer(path_colours: list[list[tuple[int, int, int, int]]]):
    image = np.array(path_colours, dtype=np.uint8)
    return image, generate_mipmaps(image)


def test_build_header_layout():
    texture = _texture(
        FORMAT_RGBA8,
        [],
        5,
        3,
        data_size=1234,
        mip_count=7,
        alpha_blending=0.25,
    )
    header = build_header(texture)
    assert len(header) == 128
    data_size, alpha, width, height, encoding, mips = struct.unpack_from(
        "<IfHHBB", header
    )
    assert data_size == 1234
    assert alpha == pytest.approx(0.25)
    assert (width, height) == (5, 3)
    assert encoding == ENCODING_RGBA
    assert mips == 7
    assert header[14:] == bytes(114)


@pytest.mark.parametrize(
    ("format_raw", "width", "height", "expected"),
    [
        (FORMAT_R8, 7, 5, 35),
        (FORMAT_RGB8, 7, 5, 105),
        (FORMAT_RGBA8, 7, 5, 140),
        (FORMAT_DXT1, 1, 1, 8),
        (FORMAT_DXT1, 4, 4, 8),
        (FORMAT_DXT1, 5, 3, 16),
        (FORMAT_DXT1, 8, 8, 32),
        (FORMAT_DXT3, 1, 1, 16),
        (FORMAT_DXT3, 8, 8, 64),
        (FORMAT_DXT5, 5, 3, 32),
    ],
)
def test_get_data_size(format_raw, width, height, expected):
    assert get_data_size(format_raw, width, height) == expected


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [(1, 1, 1), (2, 2, 2), (128, 128, 8), (128, 720, 10), (720, 128, 10)],
)
def test_header_mip_count(width, height, expected):
    assert header_mip_count(width, height) == expected


def test_is_compressed():
    assert is_compressed(FORMAT_DXT1)
    assert is_compressed(FORMAT_DXT3)
    assert is_compressed(FORMAT_DXT5)
    assert not is_compressed(FORMAT_R8)
    assert not is_compressed(FORMAT_RGB8)
    assert not is_compressed(FORMAT_RGBA8)


def test_compute_data_size_rules():
    assert compute_data_size(FORMAT_RGBA8, 16, 16, 1, False, False) == 0
    assert compute_data_size(FORMAT_RGB8, 16, 16, 1, False, False) == 0
    assert compute_data_size(FORMAT_R8, 16, 16, 1, False, False) == 0

    assert compute_data_size(FORMAT_DXT1, 8, 8, 1, False, False) == 32
    assert compute_data_size(FORMAT_DXT5, 8, 8, 6, False, True) == 64
    # Animated: every mip of one layer, multiplied by the layer count.
    assert compute_data_size(FORMAT_DXT5, 16, 16, 4, True, False) == 1472
    assert compute_data_size(FORMAT_DXT1, 16, 16, 4, True, False) == 736


def test_raw_rgba_serialisation_and_orientation():
    image = np.array(
        [
            [(1, 2, 3, 4), (5, 6, 7, 8)],
            [(9, 10, 11, 12), (13, 14, 15, 16)],
        ],
        dtype=np.uint8,
    )
    mips = generate_mipmaps(image)
    texture = _texture(
        FORMAT_RGBA8, [TPCLayer(2, 2, mips)], 2, 2, mip_count=2
    )
    data = build_tpc_bytes(texture)
    assert data[:2] == b"\x00\x00"
    expected = image[::-1].tobytes() + mips[1][::-1].tobytes()
    assert data[128:] == expected


def test_raw_rgb_serialisation_drops_alpha():
    image = np.array(
        [
            [(1, 2, 3, 200), (4, 5, 6, 201)],
            [(7, 8, 9, 202), (10, 11, 12, 203)],
        ],
        dtype=np.uint8,
    )
    mips = generate_mipmaps(image)
    texture = _texture(
        FORMAT_RGB8, [TPCLayer(2, 2, mips)], 2, 2, mip_count=2
    )
    data = build_tpc_bytes(texture)
    assert data[128:] == image[::-1, :, :3].tobytes() + mips[1][::-1, :, :3].tobytes()


def test_gray_serialisation_writes_only_base_mip():
    image = np.array(
        [[(10, 20, 30, 40)], [(50, 60, 70, 80)]], dtype=np.uint8
    )
    mips = generate_mipmaps(image)
    texture = _texture(
        FORMAT_R8,
        [TPCLayer(1, 2, mips)],
        1,
        2,
        mip_count=1,
        data_size=0,
    )
    data = build_tpc_bytes(texture)
    assert len(data) == 128 + 2
    assert data[128:] == bytes((50, 10))
    assert struct.unpack_from("<I", data)[0] == 0
    assert data[12] == ENCODING_GRAY


def _flip_texture() -> tuple[TPCTexture, np.ndarray]:
    image = np.array(
        [
            [(1, 2, 3, 4), (5, 6, 7, 8)],
            [(9, 10, 11, 12), (13, 14, 15, 16)],
        ],
        dtype=np.uint8,
    )
    mips = generate_mipmaps(image)
    texture = _texture(
        FORMAT_RGBA8, [TPCLayer(2, 2, mips)], 2, 2, mip_count=2
    )
    return texture, image


@pytest.mark.parametrize(
    ("flip_x", "flip_y", "transform"),
    [
        (False, False, lambda a: a[::-1]),
        (False, True, lambda a: a),
        (True, False, lambda a: a[::-1, ::-1]),
        (True, True, lambda a: a[:, ::-1]),
    ],
)
def test_flip_combinations(flip_x, flip_y, transform):
    texture, image = _flip_texture()
    data = build_tpc_bytes(texture, flip_x=flip_x, flip_y=flip_y)
    first_level = np.frombuffer(data[128 : 128 + 16], dtype=np.uint8)
    np.testing.assert_array_equal(
        first_level.reshape(2, 2, 4), transform(image)
    )


@pytest.mark.parametrize(
    ("format_raw", "kind", "block_size"),
    [
        (FORMAT_DXT1, "dxt1", 8),
        (FORMAT_DXT3, "dxt3", 16),
        (FORMAT_DXT5, "dxt5", 16),
    ],
)
def test_dxt_serialisation_lengths(format_raw, kind, block_size):
    image = np.array(
        [[[10 * i, 20 * i, 30 * i, 40 * i] for i in range(4)] for _ in range(4)],
        dtype=np.uint8,
    )
    mips = generate_mipmaps(image)
    data_size = get_data_size(format_raw, 4, 4)
    texture = _texture(
        format_raw,
        [TPCLayer(4, 4, mips)],
        4,
        4,
        data_size=data_size,
        mip_count=header_mip_count(4, 4),
    )
    data = build_tpc_bytes(texture)
    expected = (
        128 + data_size + get_data_size(format_raw, 2, 2) + get_data_size(format_raw, 1, 1)
    )
    assert len(data) == expected
    header = struct.unpack_from("<IfHHBB", data)
    assert header[0] == data_size
    assert header[4] == FORMAT_ENCODINGS[format_raw]
    first_block = decode_first_colour_block(data[128:], kind)
    assert len(first_block) == block_size


def decode_first_colour_block(data: bytes, kind: str) -> bytes:
    # Only used to assert the payload starts with a complete block.
    return data[: 8 if kind == "dxt1" else 16]


def test_txi_is_appended_after_pixels():
    image = np.zeros((2, 2, 4), dtype=np.uint8)
    mips = generate_mipmaps(image)
    txi = b"cube 1\r\n"
    texture = _texture(
        FORMAT_RGBA8,
        [TPCLayer(2, 2, mips)],
        2,
        2,
        mip_count=2,
        txi=txi,
    )
    data = build_tpc_bytes(texture)
    assert data.endswith(txi)
    assert len(data) == 128 + 16 + 4 + len(txi)


def test_alpha_blending_is_written_to_header():
    texture = _texture(FORMAT_RGBA8, [], 1, 1, alpha_blending=0.625)
    alpha = struct.unpack_from("<f", build_header(texture), 4)[0]
    assert alpha == pytest.approx(0.625)


def test_too_few_mipmaps_raises():
    image = np.zeros((4, 4, 4), dtype=np.uint8)
    texture = _texture(
        FORMAT_RGBA8, [TPCLayer(4, 4, [image])], 4, 4, mip_count=3
    )
    with pytest.raises(ValueError, match="not enough generated mip levels"):
        build_tpc_bytes(texture)


def test_write_tpc_creates_parent_and_returns_size(tmp_path):
    image = np.zeros((2, 2, 4), dtype=np.uint8)
    mips = generate_mipmaps(image)
    texture = _texture(
        FORMAT_RGBA8, [TPCLayer(2, 2, mips)], 2, 2, mip_count=2
    )
    path = tmp_path / "nested" / "texture.tpc"
    size = write_tpc(texture, path)
    assert path.exists()
    assert size == len(path.read_bytes())
    assert size == 128 + 16 + 4


def test_format_names():
    for format_raw, name in [
        (FORMAT_R8, "Raw"),
        (FORMAT_RGB8, "Raw"),
        (FORMAT_RGBA8, "Raw"),
        (FORMAT_DXT1, "DXT1"),
        (FORMAT_DXT3, "DXT3"),
        (FORMAT_DXT5, "DXT5"),
    ]:
        texture = _texture(format_raw, [], 1, 1)
        assert texture.format_name == name

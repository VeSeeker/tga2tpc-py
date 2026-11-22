"""Tests for the xoreostex2tga-compatible texture converter."""

from __future__ import annotations

import math
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tga2tpc import __version__
from tga2tpc.dxt_decode import decode_dxt1, decode_dxt5
from tga2tpc.tga import decode_tga, encode_tga
from tga2tpc.tex2tga import (
    TextureConversionError,
    convert_file,
    decode_texture_bytes,
    detect_texture_type,
)
from tga2tpc.tex2tga_cli import main
from tga2tpc.tpc_reader import decode_tpc
from tga2tpc.txb_reader import decode_txb

from .helpers import make_rgb16_tga, make_truecolour_tga, psnr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = PROJECT_ROOT.parent / "exampleTextures"
REFERENCE = os.environ.get("XOREOSTEX2TGA")

requires_reference = pytest.mark.skipif(
    not REFERENCE, reason="XOREOSTEX2TGA is not set"
)
requires_examples = pytest.mark.skipif(
    not (EXAMPLES / "P_VisasHd01.dds").exists(),
    reason="exampleTextures are not available",
)


# ---------------------------------------------------------------------------
# Synthetic texture builders
# ---------------------------------------------------------------------------


def _dds_header(
    width: int,
    height: int,
    pixel_flags: int,
    *,
    fourcc: bytes = b"\0\0\0\0",
    bit_count: int = 0,
    masks: tuple[int, int, int, int] = (0, 0, 0, 0),
    mip_count: int = 1,
) -> bytes:
    header = bytearray(128)
    header[0:4] = b"DDS "
    struct.pack_into("<I", header, 4, 124)
    flags = 0x1 | 0x2 | 0x4 | 0x1000
    if mip_count > 1:
        flags |= 0x00020000
    struct.pack_into("<I", header, 8, flags)
    struct.pack_into("<II", header, 12, height, width)
    struct.pack_into("<I", header, 28, mip_count)
    struct.pack_into(
        "<II4sIIIII",
        header,
        76,
        32,
        pixel_flags,
        fourcc,
        bit_count,
        *masks,
    )
    return bytes(header)


def _standard_dxt5_dds(width: int, height: int, body: bytes) -> bytes:
    return (
        _dds_header(width, height, 0x4, fourcc=b"DXT5")
        + body
    )


def _bioware_dds(width: int, height: int, body: bytes) -> bytes:
    header = struct.pack(
        "<IIIIf",
        width,
        height,
        4,
        width * height,
        1.0,
    )
    return header + body


def _tpc_header(
    data_size: int,
    width: int,
    height: int,
    encoding: int,
    mip_count: int = 1,
) -> bytes:
    header = bytearray(128)
    struct.pack_into(
        "<IfHHBB",
        header,
        0,
        data_size,
        1.0,
        width,
        height,
        encoding,
        mip_count,
    )
    return bytes(header)


def _txb_header(
    data_size: int,
    width: int,
    height: int,
    encoding: int,
    mip_count: int = 1,
) -> bytes:
    header = bytearray(128)
    struct.pack_into(
        "<IfHHBB",
        header,
        0,
        data_size,
        1.0,
        width,
        height,
        encoding,
        mip_count,
    )
    header[14:16] = b"\x01\x01"
    return bytes(header)


def _swizzle(data: np.ndarray, width: int, height: int, bpp: int) -> bytes:
    """Inverse of the Xbox swizzle used by TPC/TXB."""

    x = np.arange(width, dtype=np.int64)[None, :].repeat(height, axis=0)
    y = np.arange(height, dtype=np.int64)[:, None].repeat(width, axis=1)
    width_bits = int(math.log2(width))
    height_bits = int(math.log2(height))
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
    out = np.empty((width * height, bpp), dtype=np.uint8)
    out[offset.reshape(-1)] = data.reshape(-1, bpp)
    return out.tobytes()


# ---------------------------------------------------------------------------
# TGA writing
# ---------------------------------------------------------------------------


def test_encode_tga_header_and_channel_order():
    pixels = np.array(
        [[[1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.uint8
    )
    data = encode_tga(pixels)
    assert len(data) == 18 + 2 * 4
    assert data[:18] == struct.pack(
        "<BBBHHBHHHHBB", 0, 0, 2, 0, 0, 0, 0, 0, 2, 1, 32, 0
    )
    assert data[18:] == bytes((3, 2, 1, 4, 7, 6, 5, 8))


def test_tga_input_round_trips_when_already_bottom_left(tmp_path):
    source = tmp_path / "input.tga"
    make_truecolour_tga(source, 7, 5, 32, origin="BL")
    output = tmp_path / "output.tga"
    convert_file(source, output)
    assert output.read_bytes() == source.read_bytes()


def test_tga_input_honours_upper_left_origin(tmp_path):
    source = tmp_path / "input.tga"
    make_truecolour_tga(source, 7, 5, 32, origin="UL")
    output = tmp_path / "output.tga"
    convert_file(source, output)

    original = source.read_bytes()
    converted = output.read_bytes()
    assert converted[:18] == struct.pack(
        "<BBBHHBHHHHBB", 0, 0, 2, 0, 0, 0, 0, 0, 7, 5, 32, 0
    )
    rows = np.frombuffer(original[18:], dtype=np.uint8).reshape(5, 7, 4)
    expected = rows[::-1].tobytes()
    assert converted[18:] == expected


def test_tga_xoreos_compat_16_bit_expansion(tmp_path):
    source = tmp_path / "rgb16.tga"
    make_rgb16_tga(source, 4, 3, attribute_bits=1, origin="BL")
    image = decode_tga(source.read_bytes(), xoreos_compat=True)
    raw = source.read_bytes()[18:]
    values = np.frombuffer(raw, dtype="<u2").reshape(3, 4)
    r = ((values >> 10) & 0x1F) << 3
    g = ((values >> 5) & 0x1F) << 3
    b = (values & 0x1F) << 3
    a = np.where(values & 0x8000, 255, 0).astype(np.uint8)
    # xoreos stores the 16-bit channels in R, G, B byte order even though its
    # pixel format is labelled B8G8R8A8.  The compatibility decoder returns
    # internal RGBA, so the output writer restores that order.
    expected = np.stack((b, g, r, a), axis=2)
    np.testing.assert_array_equal(image.pixels, expected)


# ---------------------------------------------------------------------------
# DDS reading
# ---------------------------------------------------------------------------


def test_standard_dxt5_dds_decodes_and_writes(tmp_path):
    block = struct.pack("<HHI", 0xFFFF, 0x0000, 0)
    block = b"\xff" * 8 + block
    dds = _standard_dxt5_dds(4, 4, block)
    output = tmp_path / "out.tga"
    result = convert_file_bytes(dds, output, "dds")
    assert result.format == "dxt5"
    assert result.width == result.height == 4
    expected = decode_dxt5(block, 4, 4)
    assert output.read_bytes()[18:] == encode_tga(expected)[18:]


def test_standard_raw_bgra_dds(tmp_path):
    bgra = np.array(
        [[[1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.uint8
    )
    dds = (
        _dds_header(
            2,
            1,
            0x1 | 0x40,
            bit_count=32,
            masks=(0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000),
        )
        + bgra.tobytes()
    )
    output = tmp_path / "out.tga"
    convert_file_bytes(dds, output, "dds")
    assert output.read_bytes()[18:] == bgra.tobytes()


def test_bioware_dxt5_dds(tmp_path):
    block = struct.pack("<HHI", 0xFFFF, 0x0000, 0)
    block = b"\xff" * 8 + block
    dds = _bioware_dds(4, 4, block)
    output = tmp_path / "out.tga"
    result = convert_file_bytes(dds, output, "dds")
    assert result.format == "dxt5"
    assert output.read_bytes()[18:] == encode_tga(decode_dxt5(block, 4, 4))[18:]


# ---------------------------------------------------------------------------
# TPC reading
# ---------------------------------------------------------------------------


def test_raw_rgba_tpc(tmp_path):
    pixels = np.array(
        [
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            [[9, 10, 11, 12], [13, 14, 15, 16]],
        ],
        dtype=np.uint8,
    )
    data = _tpc_header(0, 2, 2, 0x04) + pixels.tobytes()
    image = decode_tpc(data)
    np.testing.assert_array_equal(image.layers[0][0].pixels, pixels)

    output = tmp_path / "out.tga"
    convert_file_bytes(data, output, "tpc")
    assert output.read_bytes()[18:] == pixels[:, :, [2, 1, 0, 3]].tobytes()


def test_swizzled_bgra_tpc(tmp_path):
    expected = np.array(
        [
            [[1, 2, 3, 4], [5, 6, 7, 8]],
            [[9, 10, 11, 12], [13, 14, 15, 16]],
        ],
        dtype=np.uint8,
    )
    bgra = expected[:, :, [2, 1, 0, 3]]
    body = _swizzle(bgra, 2, 2, 4)
    data = _tpc_header(0, 2, 2, 0x0C) + body
    image = decode_tpc(data)
    np.testing.assert_array_equal(image.layers[0][0].pixels, expected)


def test_dxt5_tpc(tmp_path):
    block = struct.pack("<HHI", 0xFFFF, 0x0000, 0)
    block = b"\xff" * 8 + block
    data = _tpc_header(16, 4, 4, 0x04) + block
    image = decode_tpc(data)
    np.testing.assert_array_equal(
        image.layers[0][0].pixels, decode_dxt5(block, 4, 4)
    )


def test_cube_map_tpc_has_six_layers(tmp_path):
    sides = [
        struct.pack("<HHI", 0xFFFF, 0x0000, 0),
        struct.pack("<HHI", 0x0000, 0xFFFF, 0),
        struct.pack("<HHI", 0xF800, 0x001F, 0),
        struct.pack("<HHI", 0x07E0, 0x001F, 0),
        struct.pack("<HHI", 0x001F, 0xF800, 0),
        struct.pack("<HHI", 0xFFFF, 0xF800, 0),
    ]
    data = _tpc_header(8, 4, 24, 0x02) + b"".join(sides) + b"cube 1\r\n"
    image = decode_tpc(data)
    assert image.is_cube_map
    assert image.layer_count == 6
    assert all(layer[0].width == 4 for layer in image.layers)
    assert all(layer[0].height == 4 for layer in image.layers)

    output = tmp_path / "out.tga"
    result = convert_file_bytes(data, output, "tpc")
    assert result.layers == 6
    assert result.width == 4
    assert result.height == 24


# ---------------------------------------------------------------------------
# TXB reading
# ---------------------------------------------------------------------------


def test_txb_dxt1(tmp_path):
    block = struct.pack("<HHI", 0xFFFF, 0x0000, 0)
    data = _txb_header(8, 4, 4, 0x0A) + block
    image = decode_txb(data)
    np.testing.assert_array_equal(image.pixels, decode_dxt1(block, 4, 4))

    output = tmp_path / "out.tga"
    convert_file_bytes(data, output, "txb")
    assert output.read_bytes()[18:] == encode_tga(
        decode_dxt1(block, 4, 4)
    )[18:]


def test_txb_bgra_non_power_of_two(tmp_path):
    bgra = np.array(
        [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]], dtype=np.uint8
    )
    data = _txb_header(12, 3, 1, 0x04) + bgra.tobytes()
    image = decode_txb(data)
    np.testing.assert_array_equal(image.pixels, bgra[:, :, [2, 1, 0, 3]])

    output = tmp_path / "out.tga"
    convert_file_bytes(data, output, "txb")
    assert output.read_bytes()[18:] == bgra.tobytes()


def test_txb_swizzled_greyscale():
    grey = np.arange(16, dtype=np.uint8).reshape(4, 4)
    body = _swizzle(grey[:, :, None], 4, 4, 1)
    data = _txb_header(16, 4, 4, 0x09) + body
    image = decode_txb(data)
    np.testing.assert_array_equal(image.pixels[:, :, 0], grey)
    np.testing.assert_array_equal(image.pixels[:, :, 1], grey)
    np.testing.assert_array_equal(image.pixels[:, :, 2], grey)
    np.testing.assert_array_equal(image.pixels[:, :, 3], 255)


# ---------------------------------------------------------------------------
# Detection and CLI
# ---------------------------------------------------------------------------


def test_detect_texture_types():
    assert detect_texture_type(b"DDS " + b"\0" * 124) == "dds"
    assert detect_texture_type(b"\0" * 128, "texture.tpc") == "tpc"
    assert detect_texture_type(b"\0" * 128, "texture.txb") == "txb"
    assert detect_texture_type(b"\0" * 128, "texture.tga") == "tga"
    with pytest.raises(TextureConversionError):
        detect_texture_type(b"not a texture", "texture.bin")


def test_cli_converts_one_file(tmp_path, capsys):
    block = struct.pack("<HHI", 0xFFFF, 0x0000, 0)
    block = b"\xff" * 8 + block
    source = tmp_path / "input.dds"
    source.write_bytes(_standard_dxt5_dds(4, 4, block))
    output = tmp_path / "output.tga"
    assert main([str(source), str(output)]) == 0
    assert output.exists()


def test_cli_version_subprocess():
    result = subprocess.run(
        [sys.executable, "-m", "tga2tpc.tex2tga", "--version"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert __version__ in result.stdout


def test_cli_converts_directory(tmp_path):
    block = struct.pack("<HHI", 0xFFFF, 0x0000, 0)
    block = b"\xff" * 8 + block
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "a.dds").write_bytes(_standard_dxt5_dds(4, 4, block))
    (input_root / "b.dds").write_bytes(_standard_dxt5_dds(4, 4, block))
    output_root = tmp_path / "output"
    assert main(["-i", str(input_root), "-o", str(output_root), "-q"]) == 0
    assert sorted(path.name for path in output_root.glob("*.tga")) == [
        "a.tga",
        "b.tga",
    ]


# ---------------------------------------------------------------------------
# Golden examples and optional xoreos reference comparison
# ---------------------------------------------------------------------------


@requires_examples
def test_example_dds_matches_reference_tga(tmp_path):
    output = tmp_path / "visas.tga"
    result = convert_file(EXAMPLES / "P_VisasHd01.dds", output)
    assert result.format == "dxt5"
    assert result.width == result.height == 2048
    assert output.read_bytes() == (EXAMPLES / "P_VisasHd01.tga").read_bytes()


@requires_examples
def test_example_tpc_is_close_to_reference_tga(tmp_path):
    output = tmp_path / "visas_tpc.tga"
    result = convert_file(EXAMPLES / "P_VisasHd01.tpc", output)
    assert result.kind == "tpc"
    assert result.mip_count == 12
    assert result.txi_path is not None
    assert b"envmaptexture" in Path(result.txi_path).read_bytes()

    converted = np.frombuffer(output.read_bytes()[18:], dtype=np.uint8)
    reference = np.frombuffer(
        (EXAMPLES / "P_VisasHd01.tga").read_bytes()[18:], dtype=np.uint8
    )
    assert psnr(converted, reference) > 40.0


@requires_reference
@pytest.mark.parametrize("kind", ["dds", "tga", "tpc", "txb"])
def test_matches_xoreos_reference_binary(tmp_path, kind):
    block = struct.pack("<HHI", 0xFFFF, 0x0000, 0)
    block = b"\xff" * 8 + block

    if kind == "dds":
        source = tmp_path / "input.dds"
        source.write_bytes(_standard_dxt5_dds(4, 4, block))
    elif kind == "tpc":
        source = tmp_path / "input.tpc"
        source.write_bytes(_tpc_header(16, 4, 4, 0x04) + block)
    elif kind == "txb":
        source = tmp_path / "input.txb"
        source.write_bytes(_txb_header(16, 4, 4, 0x0C) + block)
    else:
        source = tmp_path / "input.tga"
        make_truecolour_tga(source, 7, 5, 32, origin="UL")

    expected = tmp_path / "expected.tga"
    subprocess.run(
        [REFERENCE, str(source), str(expected)],
        check=True,
        capture_output=True,
    )
    actual = tmp_path / "actual.tga"
    convert_file(source, actual)
    assert actual.read_bytes() == expected.read_bytes()


# ---------------------------------------------------------------------------
# Helper used by several tests
# ---------------------------------------------------------------------------


def convert_file_bytes(data: bytes, output: Path, kind: str):
    texture = decode_texture_bytes(data, kind)
    from tga2tpc.tex2tga import convert_decoded_texture

    return convert_decoded_texture(texture, output)

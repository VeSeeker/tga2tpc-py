"""High-level conversion rules, TXI handling, animation and cube maps."""

from __future__ import annotations

import struct

import numpy as np
import pytest

from tga2tpc.converter import (
    ConversionError,
    ConversionOptions,
    convert_file,
    normalise_compression,
    normalise_profile,
    prepare_texture,
)
from tga2tpc.tga import load_tga
from tga2tpc.tpc import (
    ENCODING_GRAY,
    ENCODING_RGB,
    ENCODING_RGBA,
    FORMAT_DXT1,
    FORMAT_DXT3,
    FORMAT_DXT5,
    FORMAT_R8,
    FORMAT_RGB8,
    FORMAT_RGBA8,
    get_data_size,
    header_mip_count,
)

from .helpers import make_grey_tga, make_truecolour_tga


def _prepare(
    tmp_path,
    width: int,
    height: int,
    depth: int,
    txi: str = "",
    *,
    name: str = "texture.tga",
    **options,
):
    case = make_truecolour_tga(tmp_path / name, width, height, depth)
    tga = load_tga(case.path)
    return prepare_texture(tga, txi, ConversionOptions(**options)), case


def _prepare_grey(
    tmp_path,
    width: int,
    height: int,
    depth: int,
    txi: str = "",
    **options,
):
    case = make_grey_tga(tmp_path / "grey.tga", width, height, depth)
    tga = load_tga(case.path)
    return prepare_texture(tga, txi, ConversionOptions(**options)), case


def _header(data: bytes):
    return struct.unpack_from("<IfHHBB", data)


def test_normalise_compression_aliases():
    assert normalise_compression("auto") == "auto"
    for value in ("none", "raw", "rgba8", "uncompressed", "NONE"):
        assert normalise_compression(value) == "none"
    for value in ("dxt1", "DXT1", " dxt3 ", "dxt5"):
        assert normalise_compression(value) == value.strip().lower()
    for value in ("gray", "grey", "grayscale", "greyscale"):
        assert normalise_compression(value) == "gray"


def test_normalise_compression_rejects_unknown():
    with pytest.raises(ConversionError):
        normalise_compression("bc7")


def test_normalise_profile():
    assert normalise_profile("normal") == "normal"
    assert normalise_profile("super-fast") == "super_fast"
    assert normalise_profile("ULTRA") == "ultra"
    with pytest.raises(ConversionError):
        normalise_profile("best")


def test_options_normalised():
    options = ConversionOptions(
        compression="RGBA8", profile="super-fast"
    ).normalised()
    assert options.compression == "none"
    assert options.profile == "super_fast"


@pytest.mark.parametrize(
    ("depth", "format_raw", "encoding"),
    [
        (8, FORMAT_R8, ENCODING_GRAY),
        (16, FORMAT_R8, ENCODING_GRAY),
        (24, FORMAT_DXT1, ENCODING_RGB),
        (32, FORMAT_DXT5, ENCODING_RGBA),
    ],
)
def test_auto_format_by_depth(tmp_path, depth, format_raw, encoding):
    if depth < 24:
        texture, _ = _prepare_grey(tmp_path, 8, 8, depth)
    else:
        texture, _ = _prepare(tmp_path, 8, 8, depth)
    assert texture.format_raw == format_raw
    assert texture.encoding == encoding
    if format_raw == FORMAT_R8:
        assert texture.data_size == 0
        assert texture.mip_count == 1
        assert len(texture.layers[0].mipmaps) == 1
    else:
        assert texture.data_size == get_data_size(format_raw, 8, 8)
        assert texture.mip_count == header_mip_count(8, 8)


def test_bumpmap_uncompressed_rule(tmp_path):
    txi = "isbumpmap 1\ncompresstexture 0\n"
    texture24, _ = _prepare(tmp_path, 8, 8, 24, txi)
    assert texture24.format_raw == FORMAT_RGB8
    assert texture24.data_size == 0

    texture32, _ = _prepare(
        tmp_path, 8, 8, 32, txi, name="texture32.tga"
    )
    assert texture32.format_raw == FORMAT_RGBA8
    assert texture32.data_size == 0


def test_bumpmap_with_proceduretype_is_compressed(tmp_path):
    txi = (
        "isbumpmap 1\n"
        "compresstexture 0\n"
        "proceduretype cycle\n"
        "numx 2\n"
        "numy 2\n"
    )
    texture, _ = _prepare(tmp_path, 32, 32, 24, txi)
    assert texture.format_raw == FORMAT_DXT1
    assert texture.animated


@pytest.mark.parametrize(
    ("depth", "compression", "format_raw"),
    [
        (24, "none", FORMAT_RGB8),
        (32, "none", FORMAT_RGBA8),
        (8, "none", FORMAT_R8),
        (24, "gray", FORMAT_R8),
        (32, "gray", FORMAT_R8),
        (8, "dxt5", FORMAT_DXT5),
        (24, "dxt3", FORMAT_DXT3),
        (32, "dxt1", FORMAT_DXT1),
    ],
)
def test_explicit_compression_and_depth_rules(
    tmp_path, depth, compression, format_raw
):
    if depth < 24:
        texture, _ = _prepare_grey(
            tmp_path, 8, 8, depth, compression=compression
        )
    else:
        texture, _ = _prepare(
            tmp_path, 8, 8, depth, compression=compression
        )
    assert texture.format_raw == format_raw


def test_alpha_blending_and_txi_embedding(tmp_path):
    txi = "# alphablending 0.5\nsome_directive 3\n"
    texture, _ = _prepare(tmp_path, 8, 8, 32, txi)
    assert texture.alpha_blending == pytest.approx(0.5)
    assert b"some_directive 3\r\n" in texture.txi
    assert b"alphablending" not in texture.txi


def test_alpha_blending_override(tmp_path):
    txi = "# alphablending 0.5\ncube 1\n"
    texture, _ = _prepare(
        tmp_path, 8, 48, 32, txi, alpha_blending=0.25
    )
    assert texture.alpha_blending == pytest.approx(0.25)
    assert texture.cube


def test_ignore_alpha_blending(tmp_path):
    txi = "# alphablending 0.5\n"
    texture, _ = _prepare(
        tmp_path, 8, 8, 32, txi, parse_alpha_blending=False
    )
    assert texture.alpha_blending == 1.0
    assert b"# alphablending 0.5\r\n" in texture.txi


def test_embed_txi_false(tmp_path):
    texture, _ = _prepare(
        tmp_path, 8, 8, 32, "some_directive 1\n", embed_txi=False
    )
    assert texture.txi == b""


def test_non_finite_alpha_blending_rejected(tmp_path):
    with pytest.raises(ConversionError, match="finite"):
        _prepare(
            tmp_path, 8, 8, 32, alpha_blending=float("nan")
        )


def test_sibling_txi_autoload(tmp_path):
    case = make_truecolour_tga(tmp_path / "texture.tga", 8, 8, 24)
    (tmp_path / "texture.txi").write_text(
        "# alphablending 0.75\nsome_directive 1\n", encoding="utf-8"
    )
    output = tmp_path / "texture.tpc"
    result = convert_file(
        case.path, output, ConversionOptions(compression="none")
    )
    header = _header(output.read_bytes())
    assert header[1] == pytest.approx(0.75)
    assert result.txi
    assert b"some_directive 1\r\n" in output.read_bytes()
    assert b"alphablending" not in output.read_bytes()


def test_sibling_txi_disabled(tmp_path):
    case = make_truecolour_tga(tmp_path / "texture.tga", 8, 8, 24)
    (tmp_path / "texture.txi").write_text(
        "# alphablending 0.75\n", encoding="utf-8"
    )
    output = tmp_path / "texture.tpc"
    convert_file(
        case.path,
        output,
        ConversionOptions(compression="none", autoload_txi=False),
    )
    assert _header(output.read_bytes())[1] == pytest.approx(1.0)


def test_explicit_txi_text_wins_over_sibling(tmp_path):
    case = make_truecolour_tga(tmp_path / "texture.tga", 8, 8, 24)
    (tmp_path / "texture.txi").write_text(
        "# alphablending 0.75\n", encoding="utf-8"
    )
    output = tmp_path / "texture.tpc"
    convert_file(
        case.path,
        output,
        ConversionOptions(
            compression="none", txi_text="# alphablending 0.125\n"
        ),
    )
    assert _header(output.read_bytes())[1] == pytest.approx(0.125)


def test_explicit_txi_file(tmp_path):
    case = make_truecolour_tga(tmp_path / "texture.tga", 8, 8, 24)
    txi_path = tmp_path / "shared.txi"
    txi_path.write_text("# alphablending 0.25\n", encoding="utf-8")
    output = tmp_path / "texture.tpc"
    convert_file(
        case.path,
        output,
        ConversionOptions(compression="none", txi_file=txi_path),
    )
    assert _header(output.read_bytes())[1] == pytest.approx(0.25)


def test_animated_layout_and_data_size(tmp_path):
    txi = "proceduretype cycle\nnumx 2\nnumy 2\n"
    texture, _ = _prepare(
        tmp_path,
        32,
        32,
        32,
        txi,
        compression="dxt5",
        embed_txi=False,
    )
    assert texture.animated
    assert not texture.cube
    assert texture.width == 32
    assert texture.height == 32
    assert len(texture.layers) == 4
    assert all(layer.width == 16 and layer.height == 16 for layer in texture.layers)
    assert texture.mip_count == 1
    assert texture.data_size == 1472

    output = tmp_path / "animated.tpc"
    from tga2tpc.tpc import write_tpc

    size = write_tpc(texture, output)
    assert size == 128 + 1472
    assert _header(output.read_bytes())[0] == 1472


def test_animated_default_frame_size(tmp_path):
    txi = (
        "proceduretype cycle\n"
        "numx 2\n"
        "numy 2\n"
        "defaultwidth 8\n"
        "defaultheight 8\n"
    )
    texture, _ = _prepare(
        tmp_path, 32, 32, 32, txi, compression="dxt5", embed_txi=False
    )
    assert len(texture.layers) == 4
    assert all(layer.width == 8 and layer.height == 8 for layer in texture.layers)
    assert texture.data_size == 448


def test_animated_requires_numx_numy(tmp_path):
    texture, _ = _prepare(tmp_path, 32, 32, 32, "")
    assert not texture.animated
    with pytest.raises(ConversionError, match="numx/numy"):
        _prepare(
            tmp_path,
            32,
            32,
            32,
            "proceduretype cycle\n",
            compression="dxt5",
        )


def test_animated_uncompressed_rejected(tmp_path):
    txi = "proceduretype cycle\nnumx 2\nnumy 2\n"
    with pytest.raises(ConversionError, match="uncompressed animated"):
        _prepare(tmp_path, 32, 32, 32, txi, compression="none")


def test_animated_invalid_numx(tmp_path):
    txi = "proceduretype cycle\nnumx 0\nnumy 4\n"
    with pytest.raises(ConversionError, match="positive"):
        _prepare(tmp_path, 32, 32, 32, txi, compression="dxt5")


def test_cube_layout_and_data_size(tmp_path):
    texture, _ = _prepare(
        tmp_path,
        8,
        48,
        32,
        "cube 1\n",
        compression="dxt5",
        embed_txi=False,
    )
    assert texture.cube
    assert not texture.animated
    assert texture.width == 8
    assert texture.height == 48
    assert len(texture.layers) == 6
    assert all(layer.width == 8 and layer.height == 8 for layer in texture.layers)
    assert texture.mip_count == header_mip_count(8, 8)
    assert texture.data_size == get_data_size(FORMAT_DXT5, 8, 8)

    output = tmp_path / "cube.tpc"
    from tga2tpc.tpc import write_tpc

    size = write_tpc(texture, output)
    assert size == 128 + 6 * (
        get_data_size(FORMAT_DXT5, 8, 8)
        + get_data_size(FORMAT_DXT5, 4, 4)
        + get_data_size(FORMAT_DXT5, 2, 2)
        + get_data_size(FORMAT_DXT5, 1, 1)
    )


def test_cube_requires_six_square_layers(tmp_path):
    with pytest.raises(ConversionError, match="six square layers"):
        _prepare(tmp_path, 8, 40, 32, "cube 1\n", compression="dxt5")


def test_flips_are_applied_when_writing(tmp_path):
    case = make_truecolour_tga(tmp_path / "texture.tga", 3, 2, 32)
    output = tmp_path / "texture.tpc"
    convert_file(
        case.path,
        output,
        ConversionOptions(compression="none", embed_txi=False),
    )
    # Six mip levels are not generated for a 3x2 image; only the base data
    # appears before the next level.  The first level is 3x2 RGBA.
    raw = np.frombuffer(output.read_bytes()[128 : 128 + 24], dtype=np.uint8)
    np.testing.assert_array_equal(
        raw.reshape(2, 3, 4), case.expected[::-1]
    )

    flipped_path = tmp_path / "flipped.tpc"
    convert_file(
        case.path,
        flipped_path,
        ConversionOptions(
            compression="none", embed_txi=False, flip_y=True
        ),
    )
    flipped = np.frombuffer(
        flipped_path.read_bytes()[128 : 128 + 24], dtype=np.uint8
    )
    np.testing.assert_array_equal(
        flipped.reshape(2, 3, 4), case.expected
    )


def test_overwrite_false_skips_existing_file(tmp_path):
    case = make_truecolour_tga(tmp_path / "texture.tga", 4, 4, 24)
    output = tmp_path / "texture.tpc"
    options = ConversionOptions(compression="none")
    convert_file(case.path, output, options)
    original = output.read_bytes()
    result = convert_file(case.path, output, ConversionOptions(
        compression="dxt5", overwrite=False
    ))
    assert result.skipped
    assert output.read_bytes() == original


def test_invalid_tga_is_wrapped(tmp_path):
    path = tmp_path / "bad.tga"
    path.write_bytes(b"not a tga")
    with pytest.raises(ConversionError):
        convert_file(path, tmp_path / "bad.tpc", ConversionOptions())


def test_convert_image_result_fields(tmp_path):
    case = make_truecolour_tga(tmp_path / "texture.tga", 8, 8, 32)
    output = tmp_path / "texture.tpc"
    result = convert_file(
        case.path,
        output,
        ConversionOptions(compression="dxt5", embed_txi=False),
    )
    assert result.format_name == "DXT5"
    assert (result.width, result.height, result.layers) == (8, 8, 1)
    assert result.data_size == get_data_size(FORMAT_DXT5, 8, 8)
    assert result.file_size == output.stat().st_size
    assert not result.skipped
    assert result.source_pixel_depth == 32

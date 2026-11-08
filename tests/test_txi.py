"""TXI parsing, stripping and serialisation tests."""

from __future__ import annotations

import pytest

from tga2tpc.txi import (
    decode_txi_bytes,
    parse_txi,
    read_txi_file,
    render_txi,
    strip_alpha_blending,
)


def test_parse_empty():
    info = parse_txi("")
    assert info.text == ""
    assert not info.is_animated
    assert not info.is_cube
    assert info.alpha_blending is None


def test_parse_animation_and_layout():
    text = (
        "proceduretype cycle\n"
        "numx 4\n"
        "numy 2\n"
        "defaultwidth 32\n"
        "defaultheight 16\n"
        "fps 8\n"
    )
    info = parse_txi(text)
    assert info.is_animated
    assert info.has_proceduretype
    assert info.numx == 4
    assert info.numy == 2
    assert info.default_width == 32
    assert info.default_height == 16


def test_parse_cube_and_case_insensitivity():
    info = parse_txi("CUBE 1\nProcedureType cycle\n")
    assert info.is_cube
    assert info.is_animated
    assert info.has_proceduretype


@pytest.mark.parametrize("value", ["1", "t", "T", "y", "Y"])
def test_parse_bumpmap_true(value):
    info = parse_txi(f"isbumpmap {value}\n")
    assert info.is_bumpmap


@pytest.mark.parametrize("value", ["0", "f", "F", "n", "N"])
def test_parse_compresstexture_false(value):
    info = parse_txi(f"compresstexture {value}\n")
    assert info.compress_texture_false


def test_parse_alpha_blending():
    info = parse_txi("# alphablending 0.375\n")
    assert info.alpha_blending == pytest.approx(0.375)


def test_parse_alpha_blending_nan_is_ignored():
    info = parse_txi("# alphablending nope\n")
    assert info.alpha_blending is None


def test_parse_non_animation_proceduretype():
    info = parse_txi("proceduretype something\n")
    assert not info.is_animated
    assert info.has_proceduretype


def test_strip_alpha_blending():
    text = "# alphablending 0.5\ncube 1\n# alphablending 0.25\n"
    stripped = strip_alpha_blending(text)
    assert "alphablending" not in stripped
    assert "cube 1" in stripped


def test_render_txi_normalises_line_endings():
    rendered = render_txi("cube 1\nmode 2")
    assert rendered == b"cube 1\r\nmode 2\r\n"


def test_render_txi_empty():
    assert render_txi("") == b""
    assert render_txi("   \n") == b""


def test_decode_txi_bytes_strips_bom():
    assert decode_txi_bytes(b"\xef\xbb\xbfcube 1\r\n") == "cube 1\r\n"


def test_read_txi_file(tmp_path):
    path = tmp_path / "example.txi"
    path.write_bytes(b"\xef\xbb\xbfnumx 2\r\n")
    assert read_txi_file(path) == "numx 2\r\n"


def test_render_txi_accepts_crlf_input():
    assert render_txi("a\r\nb\r\n") == b"a\r\nb\r\n"

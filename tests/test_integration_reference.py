"""Optional integration tests against reference TPCs from the JS tool.

Set ``TGA2TPC_REFERENCE_MANIFEST`` to a JSON file containing a list of cases::

    [
      {
        "tga": "path/to/source.tga",
        "reference": "path/to/reference.tpc",
        "compression": "none",
        "bicubic": false
      }
    ]

Relative paths are resolved against the manifest's directory.  These tests are
skipped when the variable is not set, so no game assets are required for the
normal test suite.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

from tga2tpc.converter import ConversionOptions, convert_file
from tga2tpc.mipmaps import writer_mip_levels
from tga2tpc.tpc import FORMAT_R8, FORMAT_RGB8, FORMAT_RGBA8, get_data_size

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.environ.get("TGA2TPC_REFERENCE_MANIFEST"),
        reason="TGA2TPC_REFERENCE_MANIFEST is not set",
    ),
]

OPTION_FIELDS = {
    "compression",
    "profile",
    "bicubic",
    "flip_x",
    "flip_y",
    "alpha_blending",
    "autoload_txi",
    "embed_txi",
    "parse_alpha_blending",
    "alpha_weighted",
    "overwrite",
}


def _load_cases() -> list[tuple[dict, Path]]:
    manifest_path = Path(os.environ["TGA2TPC_REFERENCE_MANIFEST"])
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    return entries, manifest_path.parent


def _resolve(path: str, base: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate


def _split_header(data: bytes):
    return struct.unpack_from("<IfHHBB", data)


_RAW_FORMATS = {1: FORMAT_R8, 2: FORMAT_RGB8, 4: FORMAT_RGBA8}


def _last_mip_range(header) -> tuple[int, int]:
    """Byte range of the final mip level in an uncompressed plain texture."""

    _, _, width, height, encoding, _ = header
    format_raw = _RAW_FORMATS[encoding]
    levels = writer_mip_levels(width, height)
    if format_raw == FORMAT_R8:
        levels = levels[:1]
    sizes = [get_data_size(format_raw, w, h) for w, h in levels]
    start = 128 + sum(sizes[:-1])
    return start, start + sizes[-1]


def test_reference_cases_are_compatible(tmp_path):
    entries, base = _load_cases()
    if not entries:
        pytest.skip("manifest contains no cases")

    for index, entry in enumerate(entries):
        tga_path = _resolve(entry["tga"], base)
        reference_path = _resolve(entry["reference"], base)
        if not tga_path.is_file():
            pytest.skip(f"missing source asset: {tga_path}")
        if not reference_path.is_file():
            pytest.skip(f"missing reference TPC: {reference_path}")

        options_data = {
            key: value for key, value in entry.items() if key in OPTION_FIELDS
        }
        options = ConversionOptions(**options_data)
        output_path = tmp_path / f"case_{index}_{tga_path.stem}.tpc"
        convert_file(tga_path, output_path, options)

        ours = output_path.read_bytes()
        reference = reference_path.read_bytes()
        our_header = _split_header(ours)
        ref_header = _split_header(reference)
        compression = options.normalised().compression

        # Alpha blending, dimensions, encoding and mip count must always
        # match.  Greyscale dataSize is intentionally corrected to zero.
        assert our_header[1:6] == ref_header[1:6], entry
        if ref_header[4] == 1:
            assert our_header[0] == 0
            assert ours[4:] == reference[4:], entry
        else:
            assert our_header[0] == ref_header[0], entry
            if ref_header[0] == 0:
                if options.normalised().bicubic:
                    # The original tool wrote NaNs into the final cubic
                    # 1x1 mip for some square textures; everything else must
                    # be byte-identical.
                    start, end = _last_mip_range(ref_header)
                    assert ours[:start] == reference[:start], entry
                    assert ours[end:] == reference[end:], entry
                else:
                    assert ours == reference, entry
            elif compression != "dxt3":
                # Compressed payload bytes differ by encoder, but the layout
                # (and therefore the full file size) matches.
                assert len(ours) == len(reference), entry

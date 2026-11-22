"""
High-level texture-to-TGA conversion.

This is the Python counterpart of xoreos-tools' ``xoreostex2tga`` command.  It
reads BioWare TPC textures, Xbox TXB textures, DirectDraw Surface (DDS)
textures and Targa textures, then writes a 32-bit uncompressed TGA using the
same layout and orientation as xoreos-tools.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dds import DDSImage, DDSError, decode_dds
from .tga import TGAImage, TGAError, write_tga
from .tpc_reader import TPCImage, TPCError, decode_tpc
from .txb_reader import TXBImage, TXBError, decode_txb

TEXTURE_TYPES = ("auto", "dds", "tpc", "txb", "tga")
_SUFFIX_TYPES = {
    ".dds": "dds",
    ".tpc": "tpc",
    ".txb": "txb",
    ".tga": "tga",
}


class TextureConversionError(RuntimeError):
    """A texture could not be converted to TGA."""


@dataclass(frozen=True)
class TextureOptions:
    type: str = "auto"
    flip: bool = False
    write_txi: bool = True
    overwrite: bool = True


@dataclass
class DecodedTexture:
    """A decoded texture ready to be written as a TGA."""

    kind: str
    width: int
    height: int
    layers: list[np.ndarray]
    mip_count: int
    format: str
    txi: bytes = b""
    is_animated: bool = False
    is_cube_map: bool = False
    source: object = None

    @property
    def layer_count(self) -> int:
        return len(self.layers)

    def stacked_pixels(self) -> np.ndarray:
        if not self.layers:
            raise TextureConversionError("texture contains no image layers")
        width = self.layers[0].shape[1]
        for layer in self.layers:
            if layer.shape[1] != width:
                raise TextureConversionError(
                    "TGA output cannot combine layers with different widths"
                )
        return np.concatenate(self.layers, axis=0)


@dataclass
class TextureConversionResult:
    input_path: str
    output_path: str
    kind: str
    format: str
    width: int
    height: int
    layers: int
    mip_count: int
    file_size: int
    elapsed: float
    txi_path: str | None = None
    txi_size: int = 0
    skipped: bool = False
    is_animated: bool = False
    is_cube_map: bool = False
    extra: dict = field(default_factory=dict)


def _normalise_type(value: str) -> str:
    key = str(value).strip().lower()
    if key not in TEXTURE_TYPES:
        choices = ", ".join(TEXTURE_TYPES)
        raise TextureConversionError(
            f"unknown texture type {value!r}; choose one of: {choices}"
        )
    return key


def detect_texture_type(
    data: bytes, path: str | Path | None = None
) -> str:
    """Detect a texture type from its contents, then from its file name."""

    if data[:4] == b"DDS ":
        return "dds"

    if path is not None:
        suffix = Path(path).suffix.lower()
        if suffix in _SUFFIX_TYPES:
            return _SUFFIX_TYPES[suffix]

    # Content fallbacks for streams without a useful file name.
    if len(data) >= 18:
        image_type = data[2]
        pixel_depth = data[16]
        width, height = np.frombuffer(data, dtype="<u2", count=2, offset=12)
        if (
            image_type in (1, 2, 3, 9, 10, 11)
            and pixel_depth in (8, 16, 24, 32)
            and width > 0
            and height > 0
        ):
            return "tga"

    if len(data) >= 128:
        encoding = data[12]
        width, height = np.frombuffer(data, dtype="<u2", count=2, offset=8)
        if encoding in (1, 2, 4, 12) and width > 0 and height > 0:
            return "tpc"

    raise TextureConversionError("failed to detect texture type")


def _decode_dds(data: bytes) -> DecodedTexture:
    try:
        image: DDSImage = decode_dds(data)
    except DDSError as exc:
        raise TextureConversionError(str(exc)) from exc
    return DecodedTexture(
        kind="dds",
        width=image.width,
        height=image.height,
        layers=[np.ascontiguousarray(image.pixels)],
        mip_count=image.mip_count,
        format=image.format,
        source=image,
    )


def _decode_tpc(data: bytes) -> DecodedTexture:
    try:
        image: TPCImage = decode_tpc(data)
    except TPCError as exc:
        raise TextureConversionError(str(exc)) from exc
    return DecodedTexture(
        kind="tpc",
        width=image.width,
        height=image.height,
        layers=[np.ascontiguousarray(layer) for layer in image.base_layers],
        mip_count=image.mip_count,
        format=image.format,
        txi=image.txi,
        is_animated=image.is_animated,
        is_cube_map=image.is_cube_map,
        source=image,
    )


def _decode_txb(data: bytes) -> DecodedTexture:
    try:
        image: TXBImage = decode_txb(data)
    except TXBError as exc:
        raise TextureConversionError(str(exc)) from exc
    return DecodedTexture(
        kind="txb",
        width=image.width,
        height=image.height,
        layers=[np.ascontiguousarray(image.pixels)],
        mip_count=image.mip_count,
        format=image.format,
        txi=image.txi,
        source=image,
    )


def _decode_tga(data: bytes) -> DecodedTexture:
    from .tga import decode_tga

    try:
        image: TGAImage = decode_tga(data, xoreos_compat=True)
    except TGAError as exc:
        raise TextureConversionError(str(exc)) from exc
    # ``xoreos_compat`` already returns xoreos' internal row order, which is
    # what the TGA writer expects.
    return DecodedTexture(
        kind="tga",
        width=image.width,
        height=image.height,
        layers=[np.ascontiguousarray(image.pixels)],
        mip_count=1,
        format="tga",
        source=image,
    )


def decode_texture_bytes(
    data: bytes, kind: str = "auto", path: str | Path | None = None
) -> DecodedTexture:
    """Decode texture bytes into a :class:`DecodedTexture`."""

    kind = _normalise_type(kind)
    if kind == "auto":
        kind = detect_texture_type(data, path)
    if kind == "dds":
        return _decode_dds(data)
    if kind == "tpc":
        return _decode_tpc(data)
    if kind == "txb":
        return _decode_txb(data)
    if kind == "tga":
        return _decode_tga(data)
    raise TextureConversionError(f"unsupported texture type: {kind}")


def decode_texture_file(
    path: str | Path, kind: str = "auto"
) -> DecodedTexture:
    """Load and decode a texture file."""

    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise TextureConversionError(f"unable to read {path}: {exc}") from exc
    try:
        return decode_texture_bytes(data, kind, path)
    except TextureConversionError as exc:
        raise TextureConversionError(f"{path}: {exc}") from exc


def convert_decoded_texture(
    texture: DecodedTexture,
    output_path: str | Path,
    options: TextureOptions | None = None,
    source: str = "",
) -> TextureConversionResult:
    """Write a decoded texture as a TGA, optionally with a TXI sidecar."""

    options = options or TextureOptions()
    start = time.perf_counter()
    output_path = Path(output_path)

    layers = list(texture.layers)
    if options.flip:
        layers = [np.ascontiguousarray(layer[::-1]) for layer in layers]
    if not layers:
        raise TextureConversionError("texture contains no image layers")

    width = layers[0].shape[1]
    for layer in layers:
        if layer.shape[1] != width:
            raise TextureConversionError(
                "TGA output cannot combine layers with different widths"
            )
    stacked = np.concatenate(layers, axis=0)

    try:
        size = write_tga(output_path, stacked)
    except TGAError as exc:
        raise TextureConversionError(str(exc)) from exc

    txi_path = None
    txi_size = 0
    if options.write_txi and texture.txi:
        txi_path = output_path.with_suffix(".txi")
        try:
            txi_path.parent.mkdir(parents=True, exist_ok=True)
            txi_path.write_bytes(texture.txi)
            txi_size = len(texture.txi)
        except OSError as exc:
            raise TextureConversionError(
                f"unable to write {txi_path}: {exc}"
            ) from exc

    return TextureConversionResult(
        input_path=source,
        output_path=str(output_path),
        kind=texture.kind,
        format=texture.format,
        width=width,
        height=sum(layer.shape[0] for layer in layers),
        layers=len(layers),
        mip_count=texture.mip_count,
        file_size=size,
        elapsed=time.perf_counter() - start,
        txi_path=str(txi_path) if txi_path else None,
        txi_size=txi_size,
        is_animated=texture.is_animated,
        is_cube_map=texture.is_cube_map,
    )


def convert_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    options: TextureOptions | None = None,
) -> TextureConversionResult:
    """Convert one texture file to TGA."""

    options = options or TextureOptions()
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".tga")
    output_path = Path(output_path)

    if output_path.resolve() == input_path.resolve():
        raise TextureConversionError("input and output paths are the same")
    if not options.overwrite and output_path.exists():
        stat = output_path.stat()
        return TextureConversionResult(
            input_path=str(input_path),
            output_path=str(output_path),
            kind="",
            format="",
            width=0,
            height=0,
            layers=0,
            mip_count=0,
            file_size=stat.st_size,
            elapsed=0.0,
            skipped=True,
        )

    texture = decode_texture_file(input_path, options.type)
    return convert_decoded_texture(
        texture, output_path, options, source=str(input_path)
    )


if __name__ == "__main__":  # pragma: no cover
    from .tex2tga_cli import main

    raise SystemExit(main())

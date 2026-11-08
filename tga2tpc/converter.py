"""High-level TGA -> TPC conversion pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .dxt import PROFILE_ITERATIONS
from .mipmaps import generate_mipmaps
from .tga import TGAImage, load_tga
from .tpc import (
    FORMAT_DXT1,
    FORMAT_DXT3,
    FORMAT_DXT5,
    FORMAT_R8,
    FORMAT_RGB8,
    FORMAT_RGBA8,
    FORMAT_ENCODINGS,
    TPCLayer,
    TPCTexture,
    compute_data_size,
    header_mip_count,
    is_compressed,
    write_tpc,
)
from .txi import (
    TXIInfo,
    parse_txi,
    read_txi_file,
    render_txi,
    strip_alpha_blending,
)


class ConversionError(RuntimeError):
    """A texture could not be converted."""


COMPRESSION_CHOICES = ("auto", "none", "dxt1", "dxt3", "dxt5", "gray")
_COMPRESSION_ALIASES = {
    "auto": "auto",
    "none": "none",
    "raw": "none",
    "rgba8": "none",
    "uncompressed": "none",
    "dxt1": "dxt1",
    "dxt3": "dxt3",
    "dxt5": "dxt5",
    "grey": "gray",
    "gray": "gray",
    "grayscale": "gray",
    "greyscale": "gray",
}
_COMPRESSION_TO_FORMAT = {
    "none": FORMAT_RGBA8,
    "dxt1": FORMAT_DXT1,
    "dxt3": FORMAT_DXT3,
    "dxt5": FORMAT_DXT5,
    "gray": FORMAT_R8,
}


def normalise_compression(value: str) -> str:
    key = str(value).strip().lower()
    if key not in _COMPRESSION_ALIASES:
        choices = ", ".join(COMPRESSION_CHOICES)
        raise ConversionError(
            f"unknown compression {value!r}; choose one of: {choices}"
        )
    return _COMPRESSION_ALIASES[key]


def normalise_profile(value: str) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key not in PROFILE_ITERATIONS:
        choices = ", ".join(PROFILE_ITERATIONS)
        raise ConversionError(
            f"unknown compression profile {value!r}; choose one of: {choices}"
        )
    return key


@dataclass(frozen=True)
class ConversionOptions:
    compression: str = "auto"
    profile: str = "normal"
    bicubic: bool = True
    flip_x: bool = False
    flip_y: bool = False
    alpha_blending: float | None = None
    autoload_txi: bool = True
    txi_file: Path | None = None
    txi_text: str | None = None
    embed_txi: bool = True
    parse_alpha_blending: bool = True
    alpha_weighted: bool = True
    overwrite: bool = True

    def normalised(self) -> "ConversionOptions":
        return ConversionOptions(
            compression=normalise_compression(self.compression),
            profile=normalise_profile(self.profile),
            bicubic=self.bicubic,
            flip_x=self.flip_x,
            flip_y=self.flip_y,
            alpha_blending=self.alpha_blending,
            autoload_txi=self.autoload_txi,
            txi_file=self.txi_file,
            txi_text=self.txi_text,
            embed_txi=self.embed_txi,
            parse_alpha_blending=self.parse_alpha_blending,
            alpha_weighted=self.alpha_weighted,
            overwrite=self.overwrite,
        )


@dataclass
class ConversionResult:
    input_path: str
    output_path: str
    format_name: str
    width: int
    height: int
    layers: int
    mip_count: int
    data_size: int
    file_size: int
    txi: bool
    elapsed: float
    source_pixel_depth: int = 0
    skipped: bool = False
    extra: dict = field(default_factory=dict)


def _sibling_txi(path: Path) -> Path | None:
    candidate = path.with_suffix(".txi")
    if candidate.exists():
        return candidate
    # Be forgiving about case on case-sensitive file systems.
    try:
        stem = path.stem.lower()
        for entry in path.parent.iterdir():
            if entry.is_file() and entry.stem.lower() == stem:
                if entry.suffix.lower() == ".txi":
                    return entry
    except OSError:
        pass
    return None


def _load_txi_text(path: Path, options: ConversionOptions) -> str:
    if options.txi_text is not None:
        return options.txi_text
    if options.txi_file is not None:
        try:
            return read_txi_file(options.txi_file)
        except OSError as exc:
            raise ConversionError(f"unable to read TXI file: {exc}") from exc
    if options.autoload_txi:
        sibling = _sibling_txi(path)
        if sibling is not None:
            try:
                return read_txi_file(sibling)
            except OSError as exc:
                raise ConversionError(f"unable to read {sibling}: {exc}") from exc
    return ""


def _resolve_auto_format(tga: TGAImage, info: TXIInfo) -> str:
    """Automatic compression rules used by tga2tpc."""

    if (
        info.is_bumpmap
        and info.compress_texture_false
        and not info.has_proceduretype
        and tga.pixel_depth >= 24
    ):
        return FORMAT_RGBA8
    if tga.pixel_depth < 24:
        return FORMAT_R8
    if tga.pixel_depth > 24:
        return FORMAT_DXT5
    return FORMAT_DXT1


def _apply_depth_rules(
    format_raw: str, pixel_depth: int, explicit_gray: bool
) -> str:
    if is_compressed(format_raw) or explicit_gray:
        return format_raw
    if pixel_depth == 24:
        return FORMAT_RGB8
    if pixel_depth == 8:
        return FORMAT_R8
    return format_raw


def _animated_layout(
    tga: TGAImage, info: TXIInfo
) -> tuple[int, int, int, list[tuple[int, int]]]:
    if info.numx is None or info.numy is None:
        raise ConversionError(
            "animated TXI is missing numx/numy directives"
        )
    numx, numy = info.numx, info.numy
    if numx <= 0 or numy <= 0:
        raise ConversionError("animated TXI numx/numy must be positive")

    # This mirrors the JavaScript fallbacks (including the truncation to int).
    default_width = info.default_width
    default_height = info.default_height
    if not default_width or default_width > tga.width / numx:
        default_width = tga.width / numx
    if not default_height or default_height > tga.height / numy:
        default_height = tga.height / numy
    layer_width = int(default_width)
    layer_height = int(default_height)
    if layer_width < 1 or layer_height < 1:
        raise ConversionError("animated TXI has an invalid frame size")

    layer_count = numx * numy
    positions: list[tuple[int, int]] = []
    y = 0
    while y < tga.height:
        x = 0
        while x < tga.width:
            positions.append((x, y))
            x += layer_width
        y += layer_height
    if len(positions) < layer_count:
        raise ConversionError(
            "animated TXI frame grid does not fit the image "
            f"({len(positions)} frames available, {layer_count} required)"
        )
    return layer_width, layer_height, layer_count, positions[:layer_count]


def _cube_positions(tga: TGAImage) -> list[tuple[int, int]]:
    if tga.height < tga.width * 6:
        raise ConversionError(
            "cube map TGA must contain six square layers stacked vertically"
        )
    return [(0, tga.width * index) for index in range(6)]


def _extract_layer(
    pixels: np.ndarray, x: int, y: int, width: int, height: int
) -> np.ndarray:
    layer = pixels[y : y + height, x : x + width]
    if layer.shape[0] != height or layer.shape[1] != width:
        raise ConversionError("TXI layer layout extends beyond the TGA image")
    return np.ascontiguousarray(layer)


def prepare_texture(
    tga: TGAImage, txi_text: str, options: ConversionOptions
) -> TPCTexture:
    """Turn a decoded TGA plus TXI text into a texture ready for writing."""

    options = options.normalised()
    info = parse_txi(txi_text)

    alpha_blending = 1.0
    if options.alpha_blending is not None:
        alpha_blending = float(options.alpha_blending)
    elif options.parse_alpha_blending and info.alpha_blending is not None:
        alpha_blending = info.alpha_blending
    if not np.isfinite(alpha_blending):
        raise ConversionError("alpha blending must be a finite number")
    if options.parse_alpha_blending and info.alpha_blending is not None:
        txi_text = strip_alpha_blending(txi_text)

    compression = options.compression
    explicit_gray = compression == "gray"

    if compression == "auto":
        format_raw = _resolve_auto_format(tga, info)
    else:
        format_raw = _COMPRESSION_TO_FORMAT[compression]
    format_raw = _apply_depth_rules(format_raw, tga.pixel_depth, explicit_gray)

    # Animated uncompressed textures are not implemented by the original
    # tool either; greyscale is likewise treated as non-compressed.
    if info.is_animated and not is_compressed(format_raw):
        raise ConversionError(
            "uncompressed animated textures are not supported "
            "(choose DXT1, DXT3 or DXT5)"
        )

    if info.is_animated:
        layer_width, layer_height, layer_count, positions = _animated_layout(
            tga, info
        )
        cube = False
        animated = True
        mip_count = 1
    elif info.is_cube:
        positions = _cube_positions(tga)
        layer_width = tga.width
        layer_height = tga.width
        layer_count = 6
        cube = True
        animated = False
        mip_count = (
            1
            if format_raw == FORMAT_R8
            else header_mip_count(layer_width, layer_height)
        )
    else:
        positions = [(0, 0)]
        layer_width = tga.width
        layer_height = tga.height
        layer_count = 1
        cube = False
        animated = False
        mip_count = (
            1
            if format_raw == FORMAT_R8
            else header_mip_count(layer_width, layer_height)
        )

    data_size = compute_data_size(
        format_raw,
        layer_width,
        layer_height,
        layer_count,
        animated,
        cube,
    )

    layers: list[TPCLayer] = []
    for x, y in positions:
        base = _extract_layer(tga.pixels, x, y, layer_width, layer_height)
        if format_raw == FORMAT_R8:
            # The writer only emits the base level for greyscale textures.
            mipmaps = [base]
        else:
            mipmaps = generate_mipmaps(base, bicubic=options.bicubic)
        layers.append(
            TPCLayer(width=layer_width, height=layer_height, mipmaps=mipmaps)
        )

    txi_bytes = b""
    if options.embed_txi and txi_text.strip():
        txi_bytes = render_txi(txi_text)

    return TPCTexture(
        width=tga.width,
        height=tga.height,
        format_raw=format_raw,
        encoding=FORMAT_ENCODINGS[format_raw],
        mip_count=mip_count,
        data_size=data_size,
        layers=layers,
        alpha_blending=alpha_blending,
        txi=txi_bytes,
        animated=animated,
        cube=cube,
    )


def convert_image(
    tga: TGAImage,
    txi_text: str,
    output_path: str | Path,
    options: ConversionOptions,
    source: str = "",
) -> ConversionResult:
    start = time.perf_counter()
    texture = prepare_texture(tga, txi_text, options)
    size = write_tpc(
        texture,
        output_path,
        flip_x=options.flip_x,
        flip_y=options.flip_y,
        profile=options.profile,
        alpha_weighted=options.alpha_weighted,
    )
    elapsed = time.perf_counter() - start
    return ConversionResult(
        input_path=source,
        output_path=str(output_path),
        format_name=texture.format_name,
        width=texture.width,
        height=texture.height,
        layers=len(texture.layers),
        mip_count=texture.mip_count,
        data_size=texture.data_size,
        file_size=size,
        txi=bool(texture.txi),
        elapsed=elapsed,
        source_pixel_depth=getattr(tga, "pixel_depth", 0),
    )


def convert_file(
    input_path: str | Path,
    output_path: str | Path,
    options: ConversionOptions,
) -> ConversionResult:
    """Convert one TGA file; raises :class:`ConversionError` on failure."""

    input_path = Path(input_path)
    output_path = Path(output_path)
    if not options.overwrite and output_path.exists():
        stat = output_path.stat()
        return ConversionResult(
            input_path=str(input_path),
            output_path=str(output_path),
            format_name="",
            width=0,
            height=0,
            layers=0,
            mip_count=0,
            data_size=0,
            file_size=stat.st_size,
            txi=False,
            elapsed=0.0,
            skipped=True,
        )
    try:
        tga = load_tga(input_path)
    except Exception as exc:
        raise ConversionError(str(exc)) from exc
    txi_text = _load_txi_text(input_path, options)
    try:
        return convert_image(
            tga, txi_text, output_path, options, source=str(input_path)
        )
    except ConversionError:
        raise
    except Exception as exc:
        raise ConversionError(f"{type(exc).__name__}: {exc}") from exc

"""
Minimal, dependency-free Targa (TGA) decoder.

The decoder returns RGBA pixels in the conventional top-left canvas orientation used by the rest of the converter.  It supports the TGA variants that the original tga2tpc loader supported (types 1/2/3/9/10/11, 8/16/24/32 bit pixels, palettes and RLE), and is a little more forgiving than the JavaScript loader where its behaviour was clearly accidental (8 bit uncompressed true-colour data is treated as greyscale, 16 bit colour is scaled correctly, and 16 bit greyscale is treated as grey+alpha).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


class TGAError(ValueError):
    """Raised when a TGA file cannot be decoded."""


# Header image types
_TYPE_INDEXED = 1
_TYPE_RGB = 2
_TYPE_GREY = 3
_TYPE_RLE_INDEXED = 9
_TYPE_RLE_RGB = 10
_TYPE_RLE_GREY = 11

_ORIGIN_BL = 0
_ORIGIN_BR = 1
_ORIGIN_UL = 2
_ORIGIN_UR = 3

_ORIGIN_NAMES = {
    _ORIGIN_BL: "BL",
    _ORIGIN_BR: "BR",
    _ORIGIN_UL: "UL",
    _ORIGIN_UR: "UR",
}


@dataclass(frozen=True)
class TGAImage:
    """A decoded TGA image.

    Attributes:
        pixels: ``(height, width, 4)`` uint8 RGBA array in top-left (canvas)
            orientation.
        width: Image width.
        height: Image height.
        pixel_depth: Pixel depth from the TGA header (8, 16, 24 or 32).
        image_type: TGA image type byte.
        origin: Two-letter origin code (``BL``, ``BR``, ``UL`` or ``UR``).
    """

    pixels: np.ndarray
    width: int
    height: int
    pixel_depth: int
    image_type: int
    origin: str


@dataclass(frozen=True)
class _Header:
    id_length: int
    colormap_type: int
    image_type: int
    colormap_index: int
    colormap_length: int
    colormap_size: int
    x_origin: int
    y_origin: int
    width: int
    height: int
    pixel_depth: int
    flags: int


def _read_header(data: bytes) -> _Header:
    if len(data) < 18:
        raise TGAError("not enough data to contain a TGA header")
    (
        id_length,
        colormap_type,
        image_type,
        colormap_index,
        colormap_length,
        colormap_size,
        x_origin,
        y_origin,
        width,
        height,
        pixel_depth,
        flags,
    ) = struct.unpack_from("<BBBHHBHHHHBB", data, 0)
    return _Header(
        id_length=id_length,
        colormap_type=colormap_type,
        image_type=image_type,
        colormap_index=colormap_index,
        colormap_length=colormap_length,
        colormap_size=colormap_size,
        x_origin=x_origin,
        y_origin=y_origin,
        width=width,
        height=height,
        pixel_depth=pixel_depth,
        flags=flags,
    )


def _check_header(header: _Header) -> None:
    image_type = header.image_type
    if image_type in (_TYPE_INDEXED, _TYPE_RLE_INDEXED):
        if header.colormap_type != 1:
            raise TGAError("indexed TGA has no colour map")
        if header.colormap_length > 256:
            raise TGAError("indexed TGA colour map has more than 256 entries")
        if header.colormap_size not in (15, 16, 24, 32):
            raise TGAError(
                f"unsupported indexed TGA colour map depth: {header.colormap_size}"
            )
    elif image_type in (
        _TYPE_RGB,
        _TYPE_GREY,
        _TYPE_RLE_RGB,
        _TYPE_RLE_GREY,
    ):
        if header.colormap_type:
            raise TGAError("true-colour TGA unexpectedly has a colour map")
    elif image_type == 0:
        raise TGAError("TGA contains no image data")
    else:
        raise TGAError(f"unsupported TGA image type: {image_type}")

    if header.width <= 0 or header.height <= 0:
        raise TGAError("invalid TGA image dimensions")
    if header.pixel_depth not in (8, 16, 24, 32):
        raise TGAError(f"unsupported TGA pixel depth: {header.pixel_depth}")


def _decode_rle(
    data: bytes, offset: int, pixel_count: int, pixel_size: int
) -> bytes:
    """Decode TGA RLE packets into a flat pixel byte string."""

    total = pixel_count * pixel_size
    out = bytearray(total)
    shift = 0
    length = len(data)
    while shift < total:
        if offset >= length:
            raise TGAError("truncated TGA RLE data")
        control = data[offset]
        offset += 1
        count = (control & 0x7F) + 1
        if control & 0x80:
            if offset + pixel_size > length:
                raise TGAError("truncated TGA RLE packet")
            pixel = data[offset : offset + pixel_size]
            offset += pixel_size
            if shift + count * pixel_size > total:
                raise TGAError("TGA RLE packet overruns image data")
            for i in range(count):
                start = shift + i * pixel_size
                out[start : start + pixel_size] = pixel
            shift += count * pixel_size
        else:
            count *= pixel_size
            if offset + count > length:
                raise TGAError("truncated TGA RLE raw packet")
            if shift + count > total:
                raise TGAError("TGA RLE raw packet overruns image data")
            out[shift : shift + count] = data[offset : offset + count]
            offset += count
            shift += count
    return bytes(out)


def _palette_rgba(raw: bytes, count: int, depth: int) -> np.ndarray:
    """Convert a BGR(A) palette to an RGBA lookup table."""

    entries = count * ((depth + 7) // 8)
    raw = raw[:entries]
    palette = np.zeros((max(count, 1), 4), dtype=np.uint8)
    palette[:, 3] = 255
    if depth == 15 or depth == 16:
        # 1 bit alpha + 5:5:5 colour.
        values = np.frombuffer(raw, dtype="<u2", count=min(count, len(raw) // 2))
        if values.size:
            r5 = (values >> 10) & 0x1F
            g5 = (values >> 5) & 0x1F
            b5 = values & 0x1F
            palette[: values.size, 0] = (r5 << 3) | (r5 >> 2)
            palette[: values.size, 1] = (g5 << 3) | (g5 >> 2)
            palette[: values.size, 2] = (b5 << 3) | (b5 >> 2)
            if depth == 16:
                palette[: values.size, 3] = np.where(values & 0x8000, 255, 255)
    elif depth == 24:
        values = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        n = min(count, values.shape[0])
        palette[:n, 0] = values[:n, 2]
        palette[:n, 1] = values[:n, 1]
        palette[:n, 2] = values[:n, 0]
    elif depth == 32:
        values = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 4)
        n = min(count, values.shape[0])
        palette[:n, 0] = values[:n, 2]
        palette[:n, 1] = values[:n, 1]
        palette[:n, 2] = values[:n, 0]
        palette[:n, 3] = values[:n, 3]
    return palette


def _expand_5bit(values: np.ndarray) -> np.ndarray:
    return ((values << 3) | (values >> 2)).astype(np.uint8)


def _decode_pixels(
    raw: bytes,
    width: int,
    height: int,
    pixel_depth: int,
    image_type: int,
    flags: int,
    palette: np.ndarray | None,
    *,
    xoreos_compat: bool = False,
) -> np.ndarray:
    """Convert flat file-order pixels to an RGBA array in file order."""

    count = width * height
    rgba = np.empty((count, 4), dtype=np.uint8)
    rgba[:, 3] = 255
    is_grey = image_type in (_TYPE_GREY, _TYPE_RLE_GREY)
    is_indexed = image_type in (_TYPE_INDEXED, _TYPE_RLE_INDEXED)

    if is_indexed:
        if pixel_depth != 8:
            raise TGAError("indexed TGA must use 8 bit indices")
        if palette is None:
            raise TGAError("indexed TGA has no colour map data")
        indices = np.frombuffer(raw, dtype=np.uint8, count=count).astype(np.intp)
        valid = indices < palette.shape[0]
        safe = np.where(valid, indices, 0)
        rgba[valid] = palette[safe[valid]]
        rgba[~valid] = 0
        rgba[~valid, 3] = 255
    elif is_grey:
        if pixel_depth == 8:
            values = np.frombuffer(raw, dtype=np.uint8, count=count)
            rgba[:, 0] = values
            rgba[:, 1] = values
            rgba[:, 2] = values
        elif pixel_depth == 16:
            values = np.frombuffer(raw, dtype=np.uint8, count=count * 2)
            grey = values[0::2]
            rgba[:, 0] = grey
            rgba[:, 1] = grey
            rgba[:, 2] = grey
            rgba[:, 3] = values[1::2]
        else:
            raise TGAError("unsupported greyscale TGA pixel depth")
    else:
        # True-colour image.
        if pixel_depth == 8:
            values = np.frombuffer(raw, dtype=np.uint8, count=count)
            rgba[:, 0] = values
            rgba[:, 1] = values
            rgba[:, 2] = values
        elif pixel_depth == 16:
            values = np.frombuffer(raw, dtype="<u2", count=count)
            r5 = (values >> 10) & 0x1F
            g5 = (values >> 5) & 0x1F
            b5 = values & 0x1F
            if xoreos_compat:
                # xoreos-tools expands 16-bit TGA colours by shifting only,
                # and always takes the alpha bit from bit 15.
                rgba[:, 0] = (r5 << 3).astype(np.uint8)
                rgba[:, 1] = (g5 << 3).astype(np.uint8)
                rgba[:, 2] = (b5 << 3).astype(np.uint8)
                rgba[:, 3] = np.where(values & 0x8000, 255, 0).astype(np.uint8)
            else:
                rgba[:, 0] = _expand_5bit(r5)
                rgba[:, 1] = _expand_5bit(g5)
                rgba[:, 2] = _expand_5bit(b5)
                attribute_bits = flags & 0x0F
                if attribute_bits:
                    rgba[:, 3] = np.where(values & 0x8000, 255, 0).astype(
                        np.uint8
                    )
        elif pixel_depth == 24:
            values = np.frombuffer(raw, dtype=np.uint8, count=count * 3).reshape(
                -1, 3
            )
            rgba[:, 0] = values[:, 2]
            rgba[:, 1] = values[:, 1]
            rgba[:, 2] = values[:, 0]
        elif pixel_depth == 32:
            values = np.frombuffer(raw, dtype=np.uint8, count=count * 4).reshape(
                -1, 4
            )
            rgba[:, 0] = values[:, 2]
            rgba[:, 1] = values[:, 1]
            rgba[:, 2] = values[:, 0]
            rgba[:, 3] = values[:, 3]
        else:
            raise TGAError("unsupported true-colour TGA pixel depth")

    return rgba.reshape(height, width, 4)


def _orient(pixels: np.ndarray, origin: int) -> np.ndarray:
    """Map file-order rows to the top-left canvas orientation."""

    if origin == _ORIGIN_UL:
        return pixels
    if origin == _ORIGIN_BL:
        return pixels[::-1]
    if origin == _ORIGIN_UR:
        return pixels[:, ::-1]
    if origin == _ORIGIN_BR:
        return pixels[::-1, ::-1]
    raise TGAError(f"unsupported TGA origin: {origin}")


def _xoreos_flip_flat(
    data: np.ndarray, width: int, height: int, bpp: int
) -> np.ndarray:
    """Reproduce xoreos' ``flipVertically`` on a flat pixel buffer.

    xoreos passes the *file* pixel depth to the flip helper even after it has
    converted 16-bit and 8-bit images to four bytes per pixel.  For those
    formats the flip therefore operates on a too-small pitch; this helper
    keeps that behaviour for byte-exact output.
    """

    if width <= 0 or height <= 0 or bpp <= 0:
        return data

    flat = np.ascontiguousarray(data).reshape(-1).copy()
    pitch = bpp * width
    for y in range(height // 2):
        start = y * pitch
        end = (height - 1 - y) * pitch
        first = flat[start : start + pitch].copy()
        flat[start : start + pitch] = flat[end : end + pitch]
        flat[end : end + pitch] = first
    return flat.reshape(data.shape)


def decode_tga(data: bytes, *, xoreos_compat: bool = False) -> TGAImage:
    """Decode TGA bytes into a :class:`TGAImage`.

    Normally ``pixels`` is in the conventional top-left canvas orientation.
    With ``xoreos_compat`` the decoder instead reproduces xoreos-tools' TGA
    loader: the result is xoreos' internal, bottom-up pixel buffer, the
    horizontal origin bit is ignored, 16-bit pixels use xoreos' truncating
    expansion, and its 8/16-bit flip quirk is preserved.
    """

    header = _read_header(data)
    _check_header(header)

    offset = 18 + header.id_length
    if offset > len(data):
        raise TGAError("truncated TGA header/id field")

    use_rle = header.image_type in (
        _TYPE_RLE_INDEXED,
        _TYPE_RLE_RGB,
        _TYPE_RLE_GREY,
    )
    use_pal = header.image_type in (_TYPE_INDEXED, _TYPE_RLE_INDEXED)
    pixel_size = header.pixel_depth >> 3
    pixel_count = header.width * header.height

    palette = None
    if use_pal:
        palette_size = header.colormap_length * (
            (header.colormap_size + 7) // 8
        )
        if offset + palette_size > len(data):
            raise TGAError("truncated TGA colour map")
        palette = _palette_rgba(
            data[offset : offset + palette_size],
            header.colormap_length,
            header.colormap_size,
        )
        offset += palette_size

    if use_rle:
        raw = _decode_rle(data, offset, pixel_count, pixel_size)
    else:
        end = offset + pixel_count * pixel_size
        if end > len(data):
            raise TGAError("truncated TGA pixel data")
        raw = data[offset:end]

    file_pixels = _decode_pixels(
        raw,
        header.width,
        header.height,
        header.pixel_depth,
        header.image_type,
        header.flags,
        palette,
        xoreos_compat=xoreos_compat,
    )
    if xoreos_compat:
        # Return xoreos-tools' internal (post-load) pixel buffer.  Its TGA
        # loader only honours the upper-left origin bit and, for 8/16-bit
        # input, flips using the file's original bytes-per-pixel.
        origin_bits = 2 if (header.flags & 0x20) else 0
        if header.pixel_depth == 16:
            xoreos_bytes = file_pixels
        elif header.pixel_depth == 24:
            xoreos_bytes = file_pixels[:, :, [2, 1, 0]]
        else:
            xoreos_bytes = file_pixels[:, :, [2, 1, 0, 3]]
        if header.flags & 0x20:
            xoreos_bytes = _xoreos_flip_flat(
                xoreos_bytes,
                header.width,
                header.height,
                max(header.pixel_depth // 8, 1),
            )
        if header.pixel_depth == 24:
            pixels = np.empty(
                (header.height, header.width, 4), dtype=np.uint8
            )
            pixels[:, :, :3] = xoreos_bytes[:, :, [2, 1, 0]]
            pixels[:, :, 3] = 255
        else:
            pixels = np.ascontiguousarray(xoreos_bytes[:, :, [2, 1, 0, 3]])
    else:
        origin_bits = (header.flags & 0x30) >> 4
        pixels = np.ascontiguousarray(_orient(file_pixels, origin_bits))

    return TGAImage(
        pixels=pixels,
        width=header.width,
        height=header.height,
        pixel_depth=header.pixel_depth,
        image_type=header.image_type,
        origin=_ORIGIN_NAMES.get(origin_bits, "BL"),
    )


def load_tga(path: str | Path) -> TGAImage:
    """Load and decode a TGA file from disk."""

    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise TGAError(f"unable to read {path}: {exc}") from exc
    try:
        return decode_tga(data)
    except TGAError as exc:
        raise TGAError(f"{path}: {exc}") from exc


def encode_tga(pixels: np.ndarray) -> bytes:
    """Encode internal RGBA pixels as a 32-bit uncompressed TGA image.

    This mirrors xoreos-tools' TGA dumper: the image is written as an
    uncompressed true-colour TGA with a bottom-left origin (descriptor 0),
    and rows are emitted in the order they appear in ``pixels``.
    """

    if pixels.ndim != 3 or pixels.shape[2] != 4:
        raise TGAError("TGA output must be an (h, w, 4) RGBA array")
    if pixels.dtype != np.uint8:
        pixels = pixels.astype(np.uint8, copy=False)

    height, width = pixels.shape[:2]
    if width <= 0 or height <= 0:
        raise TGAError("cannot write an empty TGA image")
    if width > 0xFFFF or height > 0xFFFF:
        raise TGAError(f"TGA dimensions are too large: {width}x{height}")

    header = struct.pack(
        "<BBBHHBHHHHBB",
        0,  # ID length
        0,  # colour-map type
        2,  # uncompressed true-colour
        0,  # colour-map index
        0,  # colour-map length
        0,  # colour-map entry size
        0,  # X origin
        0,  # Y origin
        width,
        height,
        32,  # bits per pixel
        0,  # bottom-left origin, no alpha bits
    )
    body = np.ascontiguousarray(pixels[:, :, [2, 1, 0, 3]]).tobytes()
    return header + body


def write_tga(path: str | Path, pixels: np.ndarray) -> int:
    """Write internal RGBA pixels to ``path`` and return the byte count."""

    data = encode_tga(pixels)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data)

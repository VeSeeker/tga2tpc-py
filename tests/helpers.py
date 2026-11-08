"""Test helpers: build synthetic TGA files and decode DXT blocks."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ORIGIN_FLAGS = {"BL": 0x00, "BR": 0x10, "UL": 0x20, "UR": 0x30}
RLE_TYPES = {1: 9, 2: 10, 3: 11}


@dataclass(frozen=True)
class TGACase:
    path: Path
    expected: np.ndarray
    width: int
    height: int
    depth: int
    image_type: int
    origin: str


def tga_header(
    width: int,
    height: int,
    pixel_depth: int,
    *,
    image_type: int = 2,
    origin: str = "BL",
    attribute_bits: int = 0,
    id_length: int = 0,
    colormap_type: int = 0,
    colormap_index: int = 0,
    colormap_length: int = 0,
    colormap_size: int = 0,
) -> bytes:
    flags = ORIGIN_FLAGS[origin] | (attribute_bits & 0x0F)
    return struct.pack(
        "<BBBHHBHHHHBB",
        id_length,
        colormap_type,
        image_type,
        colormap_index,
        colormap_length,
        colormap_size,
        0,
        0,
        width,
        height,
        pixel_depth,
        flags,
    )


def pack_rle(pixels: list[bytes]) -> bytes:
    """Encode pixel bytes using both TGA RLE run and raw packets."""

    out = bytearray()
    i = 0
    total = len(pixels)
    while i < total:
        run = 1
        while (
            i + run < total
            and run < 128
            and pixels[i + run] == pixels[i]
        ):
            run += 1
        if run >= 2:
            out.append(0x80 | (run - 1))
            out.extend(pixels[i])
            i += run
            continue

        start = i
        i += 1
        while i < total and i - start < 128:
            # Stop just before a repeat so the next packet can encode it.
            if i + 1 < total and pixels[i] == pixels[i + 1]:
                break
            i += 1
        count = i - start
        out.append(count - 1)
        for pixel in pixels[start:i]:
            out.extend(pixel)
    return bytes(out)


def _truecolour_file_pixels(
    width: int, height: int, depth: int
) -> tuple[list[bytes], np.ndarray]:
    pixels: list[bytes] = []
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            r = (x * 53 + y * 17 + 11) & 0xFF
            g = (x * 29 + y * 71 + 7) & 0xFF
            b = (x * 97 + y * 31 + 3) & 0xFF
            if depth == 24:
                a = 255
                pixels.append(bytes((b, g, r)))
            else:
                a = (x * 13 + y * 41 + 5) & 0xFF
                pixels.append(bytes((b, g, r, a)))
            rgba[y, x] = (r, g, b, a)
    return pixels, rgba


def _rgb16_file_pixels(
    width: int, height: int, attribute_bits: int
) -> tuple[list[bytes], np.ndarray]:
    pixels: list[bytes] = []
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            i = y * width + x
            r5 = (i * 7) & 0x1F
            g5 = (i * 13) & 0x1F
            b5 = (i * 3) & 0x1F
            opaque = bool((i // 3) & 1)
            value = (r5 << 10) | (g5 << 5) | b5
            if attribute_bits:
                value |= int(opaque) << 15
            pixels.append(struct.pack("<H", value))
            rgba[y, x] = (
                (r5 << 3) | (r5 >> 2),
                (g5 << 3) | (g5 >> 2),
                (b5 << 3) | (b5 >> 2),
                255 if (opaque or not attribute_bits) else 0,
            )
    return pixels, rgba


def _grey_file_pixels(
    width: int, height: int, depth: int
) -> tuple[list[bytes], np.ndarray]:
    pixels: list[bytes] = []
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            i = y * width + x
            grey = (i * 37 + 11) & 0xFF
            if depth == 8:
                pixels.append(bytes((grey,)))
                rgba[y, x] = (grey, grey, grey, 255)
            else:
                alpha = (i * 19 + 3) & 0xFF
                pixels.append(bytes((grey, alpha)))
                rgba[y, x] = (grey, grey, grey, alpha)
    return pixels, rgba


def _indexed_file_pixels(
    width: int, height: int, palette: list[tuple[int, int, int]]
) -> tuple[list[bytes], np.ndarray]:
    pixels: list[bytes] = []
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            index = (x * 5 + y * 3 + 1) % len(palette)
            pixels.append(bytes((index,)))
            r, g, b = palette[index]
            rgba[y, x] = (r, g, b, 255)
    return pixels, rgba


def orient_file_rgba(file_rgba: np.ndarray, origin: str) -> np.ndarray:
    if origin == "BL":
        return file_rgba[::-1]
    if origin == "BR":
        return file_rgba[::-1, ::-1]
    if origin == "UL":
        return file_rgba
    if origin == "UR":
        return file_rgba[:, ::-1]
    raise ValueError(origin)


def write_tga_bytes(
    path: Path,
    *,
    width: int,
    height: int,
    depth: int,
    file_pixels: list[bytes],
    file_rgba: np.ndarray,
    image_type: int,
    origin: str,
    rle: bool,
    palette: list[tuple[int, int, int]] | None = None,
    id_field: bytes = b"",
    attribute_bits: int = 0,
) -> TGACase:
    palette_bytes = b""
    colormap_type = 0
    colormap_length = 0
    colormap_size = 0
    if palette is not None:
        colormap_type = 1
        colormap_length = len(palette)
        colormap_size = 24
        palette_bytes = b"".join(
            bytes((b, g, r)) for r, g, b in palette
        )

    body = pack_rle(file_pixels) if rle else b"".join(file_pixels)
    header = tga_header(
        width,
        height,
        depth,
        image_type=RLE_TYPES[image_type] if rle else image_type,
        origin=origin,
        attribute_bits=attribute_bits,
        id_length=len(id_field),
        colormap_type=colormap_type,
        colormap_length=colormap_length,
        colormap_size=colormap_size,
    )
    path.write_bytes(header + id_field + palette_bytes + body)
    return TGACase(
        path=path,
        expected=np.ascontiguousarray(orient_file_rgba(file_rgba, origin)),
        width=width,
        height=height,
        depth=depth,
        image_type=RLE_TYPES[image_type] if rle else image_type,
        origin=origin,
    )


def make_truecolour_tga(
    path: Path,
    width: int,
    height: int,
    depth: int,
    *,
    origin: str = "BL",
    rle: bool = False,
    id_field: bytes = b"",
) -> TGACase:
    pixels, rgba = _truecolour_file_pixels(width, height, depth)
    return write_tga_bytes(
        path,
        width=width,
        height=height,
        depth=depth,
        file_pixels=pixels,
        file_rgba=rgba,
        image_type=2,
        origin=origin,
        rle=rle,
        id_field=id_field,
    )


def make_rgb16_tga(
    path: Path,
    width: int,
    height: int,
    *,
    origin: str = "BL",
    rle: bool = False,
    attribute_bits: int = 8,
) -> TGACase:
    pixels, rgba = _rgb16_file_pixels(width, height, attribute_bits)
    return write_tga_bytes(
        path,
        width=width,
        height=height,
        depth=16,
        file_pixels=pixels,
        file_rgba=rgba,
        image_type=2,
        origin=origin,
        rle=rle,
        attribute_bits=attribute_bits,
    )


def make_grey_tga(
    path: Path,
    width: int,
    height: int,
    depth: int,
    *,
    origin: str = "BL",
    rle: bool = False,
) -> TGACase:
    pixels, rgba = _grey_file_pixels(width, height, depth)
    return write_tga_bytes(
        path,
        width=width,
        height=height,
        depth=depth,
        file_pixels=pixels,
        file_rgba=rgba,
        image_type=3,
        origin=origin,
        rle=rle,
    )


def make_indexed_tga(
    path: Path,
    width: int,
    height: int,
    palette: list[tuple[int, int, int]] | None = None,
    *,
    origin: str = "BL",
    rle: bool = False,
) -> TGACase:
    if palette is None:
        palette = [
            ((i * 17) & 0xFF, (i * 31) & 0xFF, (i * 47) & 0xFF)
            for i in range(16)
        ]
    pixels, rgba = _indexed_file_pixels(width, height, palette)
    return write_tga_bytes(
        path,
        width=width,
        height=height,
        depth=8,
        file_pixels=pixels,
        file_rgba=rgba,
        image_type=1,
        origin=origin,
        rle=rle,
        palette=palette,
    )


def solid_rgba(
    width: int, height: int, colour: tuple[int, int, int, int]
) -> np.ndarray:
    array = np.empty((height, width, 4), dtype=np.uint8)
    array[...] = colour
    return array


# ---------------------------------------------------------------------------
# DXT reference decoder used only by the tests.
# ---------------------------------------------------------------------------


def _expand565(value: int) -> np.ndarray:
    r5 = (value >> 11) & 0x1F
    g6 = (value >> 5) & 0x3F
    b5 = value & 0x1F
    return np.array(
        [
            (r5 << 3) | (r5 >> 2),
            (g6 << 2) | (g6 >> 4),
            (b5 << 3) | (b5 >> 2),
        ],
        dtype=np.float64,
    )


def _decode_colour_block(
    data: bytes, offset: int, kind: str
) -> np.ndarray:
    c0, c1, bits = struct.unpack_from("<HHI", data, offset)
    e0 = _expand565(c0)
    e1 = _expand565(c1)
    if kind == "dxt1" and c0 <= c1:
        palette = np.array([e0, e1, (e0 + e1) / 2.0, np.zeros(3)])
        transparent = c0 <= c1
    else:
        palette = np.array(
            [e0, e1, (2.0 * e0 + e1) / 3.0, (e0 + 2.0 * e1) / 3.0]
        )
        transparent = False
    rgba = np.zeros((16, 4), dtype=np.uint8)
    for pixel in range(16):
        index = (bits >> (2 * pixel)) & 3
        rgba[pixel, :3] = np.clip(
            np.floor(palette[index] + 0.5), 0, 255
        ).astype(np.uint8)
        if kind == "dxt1":
            rgba[pixel, 3] = (
                0 if (transparent and index == 3) else 255
            )
        else:
            rgba[pixel, 3] = 255
    return rgba


def decode_dxt(
    data: bytes, width: int, height: int, kind: str
) -> np.ndarray:
    """Decode DXT1/DXT3/DXT5 data into a top-left RGBA image."""

    block_w = (width + 3) // 4
    block_h = (height + 3) // 4
    padded = np.zeros((block_h * 4, block_w * 4, 4), dtype=np.uint8)
    block_size = 8 if kind == "dxt1" else 16

    for block in range(block_w * block_h):
        offset = block * block_size
        alpha = None
        if kind == "dxt3":
            bits = int.from_bytes(data[offset : offset + 8], "little")
            alpha = np.array(
                [(bits >> (4 * i)) & 0xF for i in range(16)],
                dtype=np.int32,
            )
            alpha = (alpha * 255 + 7) // 15
            colour_offset = offset + 8
        elif kind == "dxt5":
            a0, a1 = data[offset], data[offset + 1]
            if a0 > a1:
                palette = [a0, a1] + [
                    ((7 - i) * a0 + i * a1) // 7 for i in range(1, 7)
                ]
            else:
                palette = [a0, a1] + [
                    ((5 - i) * a0 + i * a1) // 5 for i in range(1, 5)
                ] + [0, 255]
            bits = int.from_bytes(data[offset + 2 : offset + 8], "little")
            alpha = np.array(
                [palette[(bits >> (3 * i)) & 7] for i in range(16)],
                dtype=np.uint8,
            )
            colour_offset = offset + 8
        else:
            colour_offset = offset

        rgba = _decode_colour_block(data, colour_offset, kind)
        if alpha is not None:
            rgba[:, 3] = alpha
        by, bx = divmod(block, block_w)
        padded[by * 4 : by * 4 + 4, bx * 4 : bx * 4 + 4] = rgba.reshape(
            4, 4, 4
        )
    return padded[:height, :width]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(255.0 * 255.0 / mse)

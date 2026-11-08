# tga2tpc (Python)

A self-contained Python reimplementation of [ndix UR's
tga2tpc](https://github.com/ndixUR/tga2tpc) 4.0.0. It converts Targa (`.tga`)
textures into BioWare Odyssey engine `.tpc` textures, including DXT1/DXT3/DXT5
compression, mipmaps, animated sheets and cube maps.

The runtime dependency is **NumPy only** — no Electron, pykotor or external
compression library is required.

## Features

- TGA image types 1, 2, 3, 9, 10 and 11 (indexed, true-colour and greyscale,
  with or without RLE).
- 8, 16, 24 and 32-bit pixels, all four image origins.
- Automatic format selection or explicit `dxt1`, `dxt3`, `dxt5`, `none`
  (raw RGB/RGBA) and `gray` output.
- Bilinear (2×2) or bicubic mipmap generation matching tga2tpc's resampling.
- TXI autoloading, embedding and directive handling:
  `proceduretype cycle`, `cube 1`, `numx`, `numy`, `defaultwidth`,
  `defaultheight`, `isbumpmap`, `compresstexture` and
  `# alphablending`.
- Animated texture sheets and vertically stacked cube maps.
- Horizontal/vertical flips and TPC alpha-blending override.
- Five DXT quality profiles (`super_fast`, `fast`, `normal`, `slow`, `ultra`).
- Alpha-weighted DXT3/DXT5 colour fitting: transparent texels no longer pull
  block endpoints away from the colours that are actually visible.
- Parallel batch conversion using a process or thread pool, with recursive
  directory discovery and preserved relative output paths.
- Readable per-file output, dry-run, skip-existing and fail-fast modes.

## Requirements

- Python 3.10 or newer
- NumPy 1.24 or newer

## Installation

From this directory:

```bash
python3 -m pip install .
```

or, with `uv`:

```bash
uv pip install .
```

You can also run the package without installing it:

```bash
python3 -m tga2tpc --help
```

## Usage

Convert a directory recursively using automatic format selection:

```bash
tga2tpc -i ./textures -o ./override
```

Force DXT5 compression and use all CPU cores:

```bash
tga2tpc -i ./textures -o ./override -compression dxt5 -cores 0
```

Convert one file with a specific TXI file and flip it horizontally:

```bash
tga2tpc -i art/logo.tga -o out/logo.tpc -compression none --txi-file art/logo.txi --flip-x
```

Preview the planned work without writing anything:

```bash
tga2tpc -i ./textures -o ./override --dry-run
```

### Important options

| Option | Description |
| --- | --- |
| `-i`, `--input PATH` | Input `.tga` file or directory (required). |
| `-o`, `--output PATH` | Output directory, or a `.tpc` path for single-file input (required). |
| `-compression MODE` | `auto` (default), `none`, `dxt1`, `dxt3`, `dxt5` or `gray`. Aliases such as `rgba8`, `uncompressed` and `greyscale` are accepted. |
| `-cores`, `-j`, `--jobs N` | Number of parallel workers; `0` uses all available CPUs. |
| `--executor {auto,process,thread}` | Parallel strategy. `auto` uses processes for multi-file batches. |
| `-profile NAME` | DXT quality: `super_fast`, `fast`, `normal` (default), `slow`, `ultra`. |
| `--bicubic` / `--no-bicubic` | Bicubic (default) or plain 2×2 mipmap downsampling. |
| `--flip-x`, `--flip-y` | Mirror the texture. `--flip-y` cancels the usual bottom-up TPC flip. |
| `--txi-file PATH`, `--txi-text TEXT` | Use an explicit TXI source for every input. |
| `--no-txi`, `--no-txi-autoload`, `--no-txi-output` | Disable all TXI handling, only autoloading, or only embedding. |
| `--alpha-blending FLOAT` | Override the TPC alpha-blending field. |
| `--no-alpha-weighted` | Fit DXT colour equally across transparent texels (the original encoder's unweighted behaviour). |
| `--no-recursive`, `--flatten` | Do not descend into subdirectories; write all outputs into one directory. |
| `--no-overwrite`, `--dry-run`, `--fail-fast` | Batch safety and diagnostics. |

Run `tga2tpc --help` for the complete list.

## Automatic format selection

When `-compression auto` is used:

| Source | Output |
| --- | --- |
| 8-bit or 16-bit TGA | Greyscale `R8` |
| 24-bit TGA | `DXT1` |
| 32-bit TGA | `DXT5` |
| Bump map with `isbumpmap 1` and `compresstexture 0`, no `proceduretype`, depth ≥ 24 | Uncompressed RGBA |

Explicit `none` writes RGB for 24-bit input and RGBA otherwise. Explicit
`gray` always writes greyscale.

## TXI handling

A sibling `.txi` file with the same stem is loaded automatically:

```text
# alphablending 0.75
proceduretype cycle
numx 4
numy 4
fps 8
```

The `# alphablending` value is applied to the TPC header and the comment is
removed from the embedded TXI text. Other directives are preserved, normalised
to CRLF line endings and appended to the TPC after the pixel data, matching the
original tool.

Animated textures require DXT output; uncompressed animations are rejected
with a clear error, as in the original implementation.

## Differences from the original 4.0.0 build

The output is intentionally compatible rather than byte-for-byte identical in
every historical edge case:

- Greyscale textures write `dataSize = 0`. The original wrote an uncompressed
  size there, which contradicts its own uncompressed-format convention.
- 16-bit true-colour TGAs use correct 5/6/5 bit expansion and alpha-bit
  handling instead of the original loader's truncated shifts and inverted
  alpha.
- The final 2×2 → 1×1 bicubic mipmap is generated correctly instead of
  producing NaN pixels for square power-of-two textures.
- Requesting `dxt3` writes real 16-byte DXT3 blocks. The original build
  accidentally wrote an unreadable DXT1-sized payload while declaring a DXT3
  data size. Note that the TPC container has no separate DXT3/DXT5 tag:
  encoding 4 is usually interpreted as DXT5 by the game and by
  `xoreostex2tga`, so prefer `dxt5` for alpha textures that must render
  correctly. `dxt3` remains available for encoder experiments and for tools
  that can select the block format externally.
- DXT compression is a NumPy encoder rather than libsquish/Compressonator.
  File headers, sizes and mip layouts match, and quality is comparable, but
  compressed pixel bytes are not identical to the JavaScript build.
- DXT3/DXT5 colour fitting is alpha-weighted by default.  Source art often
  stores black or arbitrary RGB in texels whose alpha is zero; the original
  encoder fitted those invisible texels on equal terms, which pulled block
  endpoints toward black and made sparse additive FX textures look faint or
  washed out after a roundtrip.  Pass `--no-alpha-weighted` for the original
  unweighted fit.

### DXT roundtrip brightness

DXT1/DXT3/DXT5 are lossy: colours are reduced to RGB565 endpoints plus two
interpolated palette entries per 4×4 block.  Some TPC-to-TGA tools (including
`xoreostex2tga`) expand the five- and six-bit endpoint channels with truncating
shifts (`r << 3`, `g << 2`, `b << 3`).  That decoding is up to 7 levels darker
than the bit-replicating expansion used by Direct3D-style decoders, so a
decoded TGA will usually measure darker than the source even when the TPC is
correct.  Alpha-weighted fitting removes the encoder-side part of that loss;
the remaining endpoint-quantisation and decoder-expansion differences are
inherent to the format.

Uncompressed bilinear and bicubic output has been validated byte-for-byte
against the original writer (apart from the documented fixes above).

## Development

Run the test suite:

```bash
python3 -m pytest
```

Optional integration tests compare output against reference TPCs produced by
the original JavaScript tool. Provide a JSON manifest with entries containing
`tga`, `reference`, and optional conversion option fields, then set:

```bash
export TGA2TPC_REFERENCE_MANIFEST=/path/to/manifest.json
python3 -m pytest -m integration
```

Manifest example:

```json
[
  {
    "tga": "/path/to/startbutton.tga",
    "reference": "/path/to/startbutton_none_bilinear.tpc",
    "compression": "none",
    "bicubic": false
  }
]
```

## License

GPL-3.0-or-later, matching the original tga2tpc project. See `LICENSE`.

This project is an independent reimplementation and is not affiliated with
BioWare, LucasArts or the original tga2tpc author.

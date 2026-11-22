# tga2tpc-py

A self-contained AI Python rewrite of [ndix UR's tga2tpc](https://github.com/ndixUR/tga2tpc) 4.0.0. It converts Targa (`.tga`) textures into BioWare Odyssey engine `.tpc` textures, including DXT1/DXT3/DXT5 compression, mipmaps, animated sheets and cube maps. It also includes a compatible reverse converter, [xoreo-tools](https://github.com/xoreos/xoreos-tools) `xoreostex2tga` and `tex2tga`, which turns TPC, DDS, TXB and TGA textures back into TGA images.

The runtime dependency is only NumPy.

## Features

- TGA image types 1, 2, 3, 9, 10 and 11 (indexed, true-color and greyscale,
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
- Alpha-weighted DXT3/DXT5 color fitting: transparent texels no longer pull
  block endpoints away from the colors that are actually visible.
- Parallel batch conversion using a process or thread pool, with recursive
  directory discovery and preserved relative output paths.
- Readable per-file output, dry-run, skip-existing and fail-fast modes.
- A reverse converter, `xoreostex2tga` / `tex2tga`, for TPC, DDS, TXB and TGA
  inputs, with xoreos-tools-compatible S3TC decoding and TGA output.

## Requirements

- Python 3.10 or newer
- NumPy 1.24 or newer

## Installation

From this directory:

#### Windows
```
py -m pip install .
```
#### Mac/Linux
```
python -m pip install .
```
## Run without installing

#### Windows
```
py -m tga2tpc --help
```
#### Mac/Linux
```
python -m tga2tpc --help
```

The reverse converter can also be run without installing:

```
python -m tga2tpc.tex2tga --help
```
## Usage

Convert a directory recursively using automatic format selection:

```
tga2tpc -i ./inputFolder -o ./outputFolder
```

Force DXT5 compression and use all CPU cores:

```
tga2tpc -i ./textures -o ./override -compression dxt5 -cores 0
```

Convert one file with a specific TXI file and flip it horizontally:

```
tga2tpc -i art/logo.tga -o out/logo.tpc -compression none --txi-file art/logo.txi --flip-x
```

Preview the planned work without writing anything:

```
tga2tpc -i ./textures -o ./override --dry-run
```

### Important parameters

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
| `--no-alpha-weighted` | Fit DXT color equally across transparent texels. |
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

Explicit `none` writes RGB for 24-bit input and RGBA otherwise. Explicit `gray` always writes greyscale.

## TXI handling

A sibling `.txi` file with the same stem is loaded automatically:

Animated textures require DXT output; uncompressed animations are rejected with an error, as in the original implementation.

## Converting textures back to TGA

The package also installs `xoreostex2tga` and `tex2tga`, Python ports of
xoreos-tools' texture dumper. They read BioWare TPC, Xbox TXB, DirectDraw
Surface (DDS) and Targa inputs and write 32-bit uncompressed TGA images:

```bash
# One file; the output defaults to a sibling .tga
xoreostex2tga art/visas.tpc

# Explicit output path
tex2tga --tpc art/visas.tpc out/visas.tga

# Convert a directory recursively, preserving relative paths
tex2tga -i ./textures -o ./tga

# Preview without writing
tex2tga -i ./textures -o ./tga --dry-run
```

Inputs are detected from the DDS magic or the file extension; use `--dds`,
`--tpc`, `--txb` or `--tga` to override detection. TPC files with embedded TXI
data also write a `.txi` sidecar unless `--no-txi` is passed.

Output matches xoreos-tools: a 32-bit uncompressed true-colour TGA with a
bottom-left origin, only mip level 0 of each texture layer, and multiple layers
(for cube maps) stacked vertically. The DXT1/DXT3/DXT5 decoder deliberately
reproduces xoreos' truncating colour expansion, DXT3 alpha nibble handling and
DXT5 alpha bit ordering, so DDS conversions can be byte-identical to
`xoreostex2tga` output.

## Differences from the original JavaScript build

The output is intentionally compatible rather than byte-for-byte identical in every historical edge case:

- Greyscale textures write `dataSize = 0`. The original wrote an uncompressed size there, which contradicts its own uncompressed-format convention.
- 16-bit true-color TGAs use correct 5/6/5 bit expansion and alpha-bit handling instead of the original loader's truncated shifts and inverted alpha.
- The final 2×2 → 1×1 bicubic mipmap is generated correctly instead of producing NaN pixels for square power-of-two textures.
- Requesting `dxt3` writes real 16-byte DXT3 blocks. The original build wrote an unreadable DXT1-sized payload while declaring a DXT3 data size. Note that the TPC container has no separate DXT3/DXT5 tag: encoding 4 is usually interpreted as DXT5 by the game and by `xoreostex2tga`, so prefer `dxt5` for alpha textures that must render correctly. `dxt3` remains available for encoder experiments and for tools that can select the block format externally.
- DXT compression is a NumPy encoder rather than libsquish/Compressonator. File headers, sizes and mip layouts match, and quality is comparable, but compressed pixel bytes are not identical to the JavaScript build.
- DXT3/DXT5 color fitting is alpha-weighted by default.  Source art often stores black or arbitrary RGB in texels whose alpha is zero; the original encoder fitted those invisible texels on equal terms, which pulled block endpoints toward black and made sparse additive FX textures look faint or washed out after a roundtrip.  Pass `--no-alpha-weighted` for the original unweighted fit.

## Development

Run the test suite:

```bash
python3 -m pytest
```

Optional integration tests compare output against reference TPCs produced by the original JavaScript tool. Provide a JSON manifest with entries containing `tga`, `reference`, and optional conversion option fields, then set:

```bash
export TGA2TPC_REFERENCE_MANIFEST=/path/to/manifest.json
python3 -m pytest -m integration
```

The reverse-converter tests use `../exampleTextures` when it exists. To compare
against a real xoreos-tools binary byte for byte, point `XOREOSTEX2TGA` at it:

```bash
export XOREOSTEX2TGA=/path/to/xoreostex2tga
python3 -m pytest tests/test_tex2tga.py
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

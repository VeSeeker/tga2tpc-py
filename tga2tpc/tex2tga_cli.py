"""Command line interface for the xoreostex2tga-compatible converter."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .tex2tga import (
    TEXTURE_TYPES,
    TextureConversionError,
    TextureConversionResult,
    TextureOptions,
    convert_file,
)

_SUFFIXES = {".dds", ".tpc", ".txb", ".tga"}


def _format_result(result: TextureConversionResult, verbose: bool) -> str:
    if result.skipped:
        return f"skip  {result.input_path} (output exists)"
    bits = [result.format.upper(), f"{result.width}x{result.height}"]
    if result.layers > 1:
        bits.append(f"{result.layers} layers")
    bits.append(f"{result.mip_count} mip(s)")
    if verbose:
        bits.append(f"{result.file_size:,} B")
        bits.append(f"{result.elapsed:.3f}s")
    if result.txi_path:
        bits.append("+TXI")
    return f"ok    {result.input_path} -> {result.output_path}  ({', '.join(bits)})"


def _find_inputs(root: Path, recursive: bool) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    files = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in _SUFFIXES
    ]
    files.sort(key=lambda path: str(path.relative_to(root)).lower())
    return files


def _plan_jobs(
    input_root: Path,
    output_root: Path | None,
    recursive: bool,
) -> list[tuple[Path, Path]]:
    if input_root.is_file():
        if output_root is None:
            target = input_root.with_suffix(".tga")
        elif output_root.is_dir() or output_root.suffix == "":
            target = output_root / input_root.with_suffix(".tga").name
        else:
            target = output_root
        return [(input_root, target)]

    if not input_root.is_dir():
        raise TextureConversionError(f"input does not exist: {input_root}")
    if output_root is None:
        raise TextureConversionError(
            "an output directory is required when the input is a directory"
        )
    if output_root.exists() and not output_root.is_dir():
        raise TextureConversionError(
            f"output is not a directory: {output_root}"
        )

    inputs = _find_inputs(input_root, recursive)
    if not inputs:
        raise TextureConversionError(
            f"no .tpc, .dds, .txb or .tga files found under {input_root}"
        )

    jobs: list[tuple[Path, Path]] = []
    for path in inputs:
        relative = path.relative_to(input_root).with_suffix(".tga")
        jobs.append((path, output_root / relative))
    return jobs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xoreostex2tga",
        description=(
            "Convert BioWare TPC, Xbox TXB, DDS and TGA textures to 32-bit "
            "TGA images (a Python port of xoreos-tools' xoreostex2tga)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="input texture file or directory",
    )
    parser.add_argument(
        "output",
        nargs="?",
        type=Path,
        help="output .tga file or directory",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="input_option",
        type=Path,
        help="input texture file or directory (alternative to the positional form)",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_option",
        type=Path,
        help="output .tga file or directory (alternative to the positional form)",
    )

    type_group = parser.add_mutually_exclusive_group()
    type_group.add_argument(
        "--auto",
        dest="type",
        action="store_const",
        const="auto",
        help="autodetect input type (default)",
    )
    type_group.add_argument(
        "--dds",
        dest="type",
        action="store_const",
        const="dds",
        help="input is DDS",
    )
    type_group.add_argument(
        "--tpc",
        dest="type",
        action="store_const",
        const="tpc",
        help="input is TPC",
    )
    type_group.add_argument(
        "--tga",
        dest="type",
        action="store_const",
        const="tga",
        help="input is TGA",
    )
    type_group.add_argument(
        "--txb",
        dest="type",
        action="store_const",
        const="txb",
        help="input is an Xbox TXB texture",
    )
    type_group.add_argument(
        "--sbm",
        dest="type",
        action="store_const",
        const="sbm",
        help="input is SBM (not supported by this port)",
    )
    parser.set_defaults(type="auto")

    parser.add_argument(
        "-f",
        "--flip",
        action="store_true",
        help="flip the image vertically",
    )
    parser.add_argument(
        "-d",
        "--deswizzle",
        action="store_true",
        help="deswizzle an Xbox SBM input (accepted for compatibility)",
    )
    parser.add_argument(
        "--no-txi",
        action="store_true",
        help="do not write TXI data embedded in a TPC as a .txi sidecar",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="keep existing output files and report them as skipped",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="do not recurse into subdirectories when the input is a directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list the planned conversions without writing files",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="only print errors and the final summary",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="include sizes and timings in per-file output",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.input_option is not None and args.input is not None:
        print(
            "error: specify the input either positionally or with -i, not both",
            file=sys.stderr,
        )
        return 2
    if args.output_option is not None and args.output is not None:
        print(
            "error: specify the output either positionally or with -o, not both",
            file=sys.stderr,
        )
        return 2

    input_root = args.input_option or args.input
    output_root = args.output_option or args.output
    if input_root is None:
        print("error: an input texture is required", file=sys.stderr)
        return 2
    if args.type == "sbm":
        print("error: SBM input is not supported", file=sys.stderr)
        return 2
    if args.type not in TEXTURE_TYPES:
        print(f"error: unsupported input type: {args.type}", file=sys.stderr)
        return 2

    options = TextureOptions(
        type=args.type,
        flip=args.flip,
        write_txi=not args.no_txi,
        overwrite=not args.no_overwrite,
    )

    try:
        jobs = _plan_jobs(
            input_root.expanduser(),
            output_root.expanduser() if output_root is not None else None,
            recursive=not args.no_recursive,
        )
    except TextureConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        for source, target in jobs:
            print(f"{source} -> {target}")
        print(f"Would convert {len(jobs)} file(s).")
        return 0

    converted = 0
    skipped = 0
    failed = 0
    for index, (source, target) in enumerate(jobs, 1):
        if not args.quiet:
            print(f"[{index}/{len(jobs)}] {source}", file=sys.stderr)
        try:
            result = convert_file(source, target, options)
        except Exception as exc:
            failed += 1
            print(f"error {source}: {exc}", file=sys.stderr)
            continue
        if result.skipped:
            skipped += 1
        else:
            converted += 1
        if not args.quiet:
            print(_format_result(result, args.verbose), flush=True)

    print(
        f"Converted {converted} file(s), skipped {skipped}, failed {failed}."
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

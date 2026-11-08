"""Command line interface for tga2tpc-py."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import __version__
from .converter import (
    COMPRESSION_CHOICES,
    ConversionError,
    ConversionOptions,
    ConversionResult,
    convert_file,
)
from .dxt import PROFILE_ITERATIONS


@dataclass(frozen=True)
class Job:
    input_path: Path
    output_path: Path
    relative: Path


def _default_cores() -> int:
    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        try:
            value = process_cpu_count()
        except Exception:
            value = None
        if value:
            return int(value)
    return os.cpu_count() or 1


def _find_inputs(root: Path, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise ConversionError(f"input does not exist: {root}")
    iterator = root.rglob("*") if recursive else root.glob("*")
    files = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() == ".tga"
    ]
    files.sort(key=lambda p: str(p.relative_to(root)).lower())
    return files


def _plan_jobs(
    input_root: Path,
    output_root: Path,
    recursive: bool,
    flatten: bool,
) -> list[Job]:
    inputs = _find_inputs(input_root, recursive)
    if not inputs:
        raise ConversionError(f"no .tga files found under {input_root}")

    jobs: list[Job] = []
    if input_root.is_file():
        target = (
            output_root
            if output_root.suffix.lower() == ".tpc"
            else output_root / (input_root.stem + ".tpc")
        )
        jobs.append(
            Job(
                input_path=input_root,
                output_path=target,
                relative=Path(input_root.name),
            )
        )
        return jobs

    for path in inputs:
        relative = path.relative_to(input_root)
        name = Path(relative.name).with_suffix(".tpc")
        if flatten:
            output_path = output_root / name
            display = name
        else:
            output_path = output_root / relative.with_suffix(".tpc")
            display = relative.with_suffix(".tpc")
        jobs.append(
            Job(input_path=path, output_path=output_path, relative=display)
        )
    return jobs


def _check_collisions(jobs: list[Job]) -> None:
    seen: dict[Path, Path] = {}
    collisions: list[str] = []
    for job in jobs:
        key = job.output_path.resolve()
        previous = seen.get(key)
        if previous is not None:
            collisions.append(
                f"{previous} and {job.input_path} both map to {job.output_path}"
            )
        else:
            seen[key] = job.input_path
    if collisions:
        raise ConversionError(
            "output name collision(s); disable --flatten or rename inputs:\n  "
            + "\n  ".join(collisions)
        )


def _worker(payload):
    input_path, output_path, options = payload
    try:
        result = convert_file(input_path, output_path, options)
        return ("ok", result)
    except Exception as exc:  # pragma: no cover - exercised in integration
        return ("error", str(input_path), str(exc))


def _format_result(result: ConversionResult, verbose: bool) -> str:
    if result.skipped:
        return f"skip  {result.input_path} (output exists)"
    bits = [
        result.format_name or "?",
        f"{result.width}x{result.height}",
    ]
    if result.layers > 1:
        bits.append(f"{result.layers} layers")
    bits.append(f"{result.mip_count} mip(s)")
    if verbose:
        bits.append(f"dataSize={result.data_size}")
        bits.append(f"{result.file_size:,} B")
        bits.append(f"{result.elapsed:.3f}s")
    if result.txi:
        bits.append("+TXI")
    return f"ok    {result.input_path} -> {result.output_path}  ({', '.join(bits)})"


def _run_jobs(
    jobs: list[Job],
    options: ConversionOptions,
    cores: int,
    executor_kind: str,
    quiet: bool,
    verbose: bool,
    fail_fast: bool,
) -> tuple[int, int, int, float]:
    started = time.perf_counter()
    converted = 0
    skipped = 0
    failed = 0
    total = len(jobs)

    def accept(status, value, message=""):
        nonlocal converted, skipped, failed
        if status == "error":
            failed += 1
            print(f"error {value}: {message}", file=sys.stderr)
            return
        result: ConversionResult = value
        if result.skipped:
            skipped += 1
        else:
            converted += 1
        if not quiet:
            print(_format_result(result, verbose), flush=True)

    worker_count = max(1, cores)
    if executor_kind == "auto":
        if worker_count == 1 or total <= 1:
            executor_kind = "serial"
        else:
            executor_kind = "process"
    worker_count = min(worker_count, max(total, 1))

    if executor_kind == "serial":
        for index, job in enumerate(jobs, 1):
            if not quiet:
                print(f"[{index}/{total}] {job.input_path}", file=sys.stderr)
            try:
                result = convert_file(job.input_path, job.output_path, options)
                accept("ok", result)
            except Exception as exc:
                accept("error", str(job.input_path), str(exc))
                if fail_fast:
                    break
        return converted, skipped, failed, time.perf_counter() - started

    executor_class = (
        concurrent.futures.ProcessPoolExecutor
        if executor_kind == "process"
        else concurrent.futures.ThreadPoolExecutor
    )
    futures: dict[concurrent.futures.Future, Job] = {}
    cancelled = False
    with executor_class(max_workers=worker_count) as executor:
        for job in jobs:
            payload = (job.input_path, job.output_path, options)
            futures[executor.submit(_worker, payload)] = job
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures), 1
        ):
            job = futures[future]
            if not quiet:
                print(f"[{completed}/{total}] {job.input_path}", file=sys.stderr)
            try:
                status, value, *rest = future.result()
            except Exception as exc:
                accept(
                    "error",
                    str(job.input_path),
                    f"{type(exc).__name__}: {exc}",
                )
                if fail_fast:
                    cancelled = True
            else:
                accept(status, value, rest[0] if rest else "")
            if fail_fast and failed:
                cancelled = True
            if cancelled:
                for pending in futures:
                    pending.cancel()
                break
    return converted, skipped, failed, time.perf_counter() - started


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tga2tpc",
        description=(
            "Convert Targa textures to BioWare TPC files (DXT1/DXT3/DXT5, "
            "RGB, RGBA or greyscale) with mipmaps, TXI support, animation "
            "and cube-map layouts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="input .tga file or directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="output directory (or .tpc file path for single-file input)",
    )
    parser.add_argument(
        "-compression",
        "--compression",
        default="auto",
        metavar="MODE",
        help=(
            "output format: auto, none/rgba8, dxt1, dxt3, dxt5 or gray "
            f"({', '.join(COMPRESSION_CHOICES)})"
        ),
    )
    parser.add_argument(
        "-cores",
        "-j",
        "--cores",
        "--jobs",
        dest="cores",
        type=int,
        default=0,
        metavar="N",
        help="number of parallel workers (0 = all available CPUs)",
    )
    parser.add_argument(
        "--executor",
        choices=("auto", "process", "thread"),
        default="auto",
        help="parallel execution strategy",
    )
    parser.add_argument(
        "-profile",
        "--profile",
        default="normal",
        choices=tuple(PROFILE_ITERATIONS),
        help="DXT compression quality profile",
    )
    parser.add_argument(
        "--bicubic",
        dest="bicubic",
        action="store_true",
        default=argparse.SUPPRESS,
        help="use bicubic mipmap downsampling (the default)",
    )
    parser.add_argument(
        "--no-bicubic",
        dest="bicubic",
        action="store_false",
        default=argparse.SUPPRESS,
        help="use plain 2x2 (bilinear) mipmap downsampling",
    )
    parser.add_argument(
        "--flip-x",
        action="store_true",
        help="flip textures horizontally",
    )
    parser.add_argument(
        "--flip-y",
        action="store_true",
        help="flip textures vertically (cancels the usual bottom-up flip)",
    )
    parser.add_argument(
        "--txi-file",
        type=Path,
        help="use this TXI file for every input instead of sibling .txi files",
    )
    parser.add_argument(
        "--txi-text",
        help="use this TXI text for every input",
    )
    parser.add_argument(
        "--no-txi",
        action="store_true",
        help="do not read sibling TXI files and do not embed TXI data",
    )
    parser.add_argument(
        "--no-txi-autoload",
        action="store_true",
        help="do not read sibling .txi files",
    )
    parser.add_argument(
        "--no-txi-output",
        action="store_true",
        help="read/use TXI directives but do not append TXI text to the TPC",
    )
    parser.add_argument(
        "--alpha-blending",
        type=float,
        default=None,
        metavar="FLOAT",
        help="override the TPC alpha blending value (default: TXI or 1.0)",
    )
    parser.add_argument(
        "--no-alpha-weighted",
        dest="alpha_weighted",
        action="store_false",
        default=argparse.SUPPRESS,
        help=(
            "fit DXT3/DXT5 colour equally across transparent texels instead "
            "of weighting by alpha (legacy unweighted encoder)"
        ),
    )
    parser.add_argument(
        "--ignore-alpha-blending",
        action="store_true",
        help="do not read or strip '# alphablending' from TXI data",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="do not recurse into subdirectories",
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="write all outputs directly into the output directory",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="keep existing output files and report them as skipped",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list the planned conversions without writing files",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="stop after the first conversion error",
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


def _make_options(args: argparse.Namespace) -> ConversionOptions:
    if args.txi_file is not None and args.txi_text is not None:
        raise ConversionError("--txi-file and --txi-text are mutually exclusive")
    if args.cores < 0:
        raise ConversionError("--cores must be zero or a positive integer")
    autoload = not (args.no_txi or args.no_txi_autoload)
    embed = not (args.no_txi or args.no_txi_output)
    return ConversionOptions(
        compression=args.compression,
        profile=args.profile,
        bicubic=getattr(args, "bicubic", True),
        flip_x=args.flip_x,
        flip_y=args.flip_y,
        alpha_blending=args.alpha_blending,
        autoload_txi=autoload,
        txi_file=args.txi_file,
        txi_text=args.txi_text,
        embed_txi=embed,
        parse_alpha_blending=not args.ignore_alpha_blending,
        alpha_weighted=getattr(args, "alpha_weighted", True),
        overwrite=not args.no_overwrite,
    ).normalised()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        options = _make_options(args)
        input_root = args.input.expanduser()
        output_root = args.output.expanduser()
        jobs = _plan_jobs(
            input_root,
            output_root,
            recursive=not args.no_recursive,
            flatten=args.flatten,
        )
        _check_collisions(jobs)
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        for job in jobs:
            print(f"{job.input_path} -> {job.output_path}")
        print(f"Would convert {len(jobs)} file(s).")
        return 0

    cores = args.cores or _default_cores()
    if not args.quiet:
        print(
            f"Converting {len(jobs)} file(s) with {cores} worker(s) "
            f"({args.executor})...",
            file=sys.stderr,
        )
    try:
        converted, skipped, failed, elapsed = _run_jobs(
            jobs,
            options,
            cores=cores,
            executor_kind=args.executor,
            quiet=args.quiet,
            verbose=args.verbose,
            fail_fast=args.fail_fast,
        )
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130

    print(
        f"Converted {converted} file(s), skipped {skipped}, failed {failed} "
        f"in {elapsed:.2f}s."
    )
    return 0 if failed == 0 else 1

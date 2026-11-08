"""Command line interface tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tga2tpc.cli import _make_options, build_parser, main

from .helpers import make_truecolour_tga

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _make_tree(tmp_path: Path) -> tuple[Path, Path]:
    input_root = tmp_path / "input"
    input_root.mkdir()
    make_truecolour_tga(input_root / "a.tga", 4, 4, 24)
    sub = input_root / "sub"
    sub.mkdir()
    make_truecolour_tga(sub / "b.tga", 6, 4, 32)
    return input_root, tmp_path / "output"


def test_bicubic_flag_defaults_and_overrides():
    parser = build_parser()
    base = ["-i", "input", "-o", "output"]
    assert _make_options(parser.parse_args(base)).bicubic is True
    assert _make_options(parser.parse_args(base + ["--bicubic"])).bicubic is True
    assert (
        _make_options(parser.parse_args(base + ["--no-bicubic"])).bicubic
        is False
    )


def test_alpha_weighted_flag_defaults_and_overrides():
    parser = build_parser()
    base = ["-i", "input", "-o", "output"]
    assert _make_options(parser.parse_args(base)).alpha_weighted is True
    assert (
        _make_options(parser.parse_args(base + ["--no-alpha-weighted"]))
        .alpha_weighted
        is False
    )


def test_convert_directory_recursively(tmp_path, capsys):
    input_root, output_root = _make_tree(tmp_path)
    code = main(
        [
            "-i",
            str(input_root),
            "-o",
            str(output_root),
            "-compression",
            "none",
            "-j",
            "1",
            "-q",
        ]
    )
    assert code == 0
    assert (output_root / "a.tpc").is_file()
    assert (output_root / "sub" / "b.tpc").is_file()
    captured = capsys.readouterr()
    assert "Converted 2 file(s), skipped 0, failed 0" in captured.out


def test_no_recursive(tmp_path):
    input_root, output_root = _make_tree(tmp_path)
    code = main(
        [
            "-i",
            str(input_root),
            "-o",
            str(output_root),
            "-compression",
            "none",
            "--no-recursive",
            "-j",
            "1",
            "-q",
        ]
    )
    assert code == 0
    assert (output_root / "a.tpc").is_file()
    assert not (output_root / "sub" / "b.tpc").exists()


def test_flatten(tmp_path):
    input_root, output_root = _make_tree(tmp_path)
    code = main(
        [
            "-i",
            str(input_root),
            "-o",
            str(output_root),
            "-compression",
            "none",
            "--flatten",
            "-j",
            "1",
            "-q",
        ]
    )
    assert code == 0
    assert (output_root / "a.tpc").is_file()
    assert (output_root / "b.tpc").is_file()
    assert not (output_root / "sub").exists()


def test_flatten_collision_is_rejected(tmp_path, capsys):
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "sub").mkdir()
    make_truecolour_tga(input_root / "same.tga", 4, 4, 24)
    make_truecolour_tga(input_root / "sub" / "same.tga", 4, 4, 24)
    output_root = tmp_path / "output"
    code = main(
        [
            "-i",
            str(input_root),
            "-o",
            str(output_root),
            "--flatten",
            "-j",
            "1",
        ]
    )
    assert code == 2
    assert "collision" in capsys.readouterr().err
    assert not output_root.exists()


def test_single_file_to_directory(tmp_path):
    case = make_truecolour_tga(tmp_path / "texture.tga", 4, 4, 24)
    output_root = tmp_path / "output"
    code = main(
        [
            "-i",
            str(case.path),
            "-o",
            str(output_root),
            "-compression",
            "none",
            "-j",
            "1",
            "-q",
        ]
    )
    assert code == 0
    assert (output_root / "texture.tpc").is_file()


def test_single_file_to_explicit_path(tmp_path):
    case = make_truecolour_tga(tmp_path / "texture.tga", 4, 4, 24)
    output = tmp_path / "renamed.tpc"
    code = main(
        [
            "-i",
            str(case.path),
            "-o",
            str(output),
            "-compression",
            "none",
            "-j",
            "1",
            "-q",
        ]
    )
    assert code == 0
    assert output.is_file()


def test_no_overwrite_skips(tmp_path, capsys):
    input_root, output_root = _make_tree(tmp_path)
    args = [
        "-i",
        str(input_root),
        "-o",
        str(output_root),
        "-compression",
        "none",
        "-j",
        "1",
        "-q",
    ]
    assert main(args) == 0
    capsys.readouterr()
    output = output_root / "a.tpc"
    original = output.read_bytes()
    assert main(args + ["--no-overwrite"]) == 0
    captured = capsys.readouterr()
    assert "Converted 0 file(s), skipped 2, failed 0" in captured.out
    assert output.read_bytes() == original


def test_dry_run_writes_nothing(tmp_path, capsys):
    input_root, output_root = _make_tree(tmp_path)
    code = main(
        [
            "-i",
            str(input_root),
            "-o",
            str(output_root),
            "-compression",
            "none",
            "--dry-run",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "Would convert 2 file(s)" in captured.out
    assert not output_root.exists()


def test_thread_executor(tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    for index in range(3):
        make_truecolour_tga(
            input_root / f"texture{index}.tga", 8, 8, 32
        )
    output_root = tmp_path / "output"
    code = main(
        [
            "-i",
            str(input_root),
            "-o",
            str(output_root),
            "-compression",
            "dxt5",
            "-profile",
            "fast",
            "--executor",
            "thread",
            "-j",
            "2",
            "-q",
        ]
    )
    assert code == 0
    assert len(list(output_root.glob("*.tpc"))) == 3


def test_process_executor_subprocess(tmp_path):
    input_root = tmp_path / "input"
    input_root.mkdir()
    for index in range(2):
        make_truecolour_tga(
            input_root / f"texture{index}.tga", 8, 8, 32
        )
    output_root = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tga2tpc",
            "-i",
            str(input_root),
            "-o",
            str(output_root),
            "-compression",
            "dxt5",
            "--executor",
            "process",
            "-j",
            "2",
            "-q",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert len(list(output_root.glob("*.tpc"))) == 2


def test_version_subprocess():
    result = subprocess.run(
        [sys.executable, "-m", "tga2tpc", "--version"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    assert "4.0.0" in result.stdout


def test_missing_input(tmp_path, capsys):
    code = main(
        [
            "-i",
            str(tmp_path / "does-not-exist"),
            "-o",
            str(tmp_path / "output"),
        ]
    )
    assert code == 2
    assert "does not exist" in capsys.readouterr().err


def test_no_tga_files(tmp_path, capsys):
    input_root = tmp_path / "input"
    input_root.mkdir()
    (input_root / "readme.txt").write_text("nothing", encoding="utf-8")
    code = main(
        [
            "-i",
            str(input_root),
            "-o",
            str(tmp_path / "output"),
        ]
    )
    assert code == 2
    assert "no .tga files" in capsys.readouterr().err


def test_unknown_compression(tmp_path, capsys):
    input_root, output_root = _make_tree(tmp_path)
    code = main(
        [
            "-i",
            str(input_root),
            "-o",
            str(output_root),
            "-compression",
            "bc7",
        ]
    )
    assert code == 2
    assert "unknown compression" in capsys.readouterr().err


def test_negative_cores(tmp_path, capsys):
    input_root, output_root = _make_tree(tmp_path)
    code = main(
        [
            "-i",
            str(input_root),
            "-o",
            str(output_root),
            "-cores",
            "-1",
        ]
    )
    assert code == 2
    assert "cores" in capsys.readouterr().err


def test_mutually_exclusive_txi_sources(tmp_path, capsys):
    input_root, output_root = _make_tree(tmp_path)
    code = main(
        [
            "-i",
            str(input_root),
            "-o",
            str(output_root),
            "--txi-file",
            str(tmp_path / "a.txi"),
            "--txi-text",
            "cube 1",
        ]
    )
    assert code == 2
    assert "mutually exclusive" in capsys.readouterr().err

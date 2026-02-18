"""Test the entry_point module."""

# The structure of these tests are heavily based on TopoStats tests.
from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path

import pytest
from topostats.config import write_config_with_comments

from afmslicer import run_modules
from afmslicer.entry_point import entry_point


@pytest.mark.parametrize("option", [("-h"), ("--help")])
def test_entry_point_help(capsys, option) -> None:
    """Test the help argument of the general entry point."""
    with contextlib.suppress(SystemExit):
        entry_point(manually_provided_args=[option])
    output = capsys.readouterr().out
    assert "usage:" in output
    assert "program" in output


@pytest.mark.parametrize(
    (("argument", "option")),
    [
        pytest.param("process", "-h", id="process with -h"),
        pytest.param("process", "--help", id="process with --help"),
        pytest.param("filter", "-h", id="filter with -h"),
        pytest.param("filter", "--help", id="filter with --help"),
        pytest.param("slicer", "-h", id="slicer with -h"),
        pytest.param("slicer", "--help", id="slicer with --help"),
        pytest.param("create-config", "-h", id="create-config with -h"),
        pytest.param("create-config", "--help", id="create-config with --help"),
    ],
)
def test_entry_point_subprocess_help(capsys, argument: str, option: str) -> None:
    """Test the help argument to the master and sub entry points."""
    with contextlib.suppress(SystemExit):
        entry_point(manually_provided_args=[argument, option])
    output = capsys.readouterr().out

    assert "usage:" in output
    assert argument in output


@pytest.mark.parametrize(
    ("options", "expected_function", "expected_args"),
    [
        pytest.param(
            [
                "-c",
                "dummy/config/dir/config.yaml",
                "process",
            ],
            run_modules.process,
            {"config_file": Path("dummy/config/dir/config.yaml")},
            id="Process with config file argument",
        ),
        pytest.param(
            [
                "-b",
                "/tmp/",
                "process",
            ],
            run_modules.process,
            {"base_dir": Path("/tmp/")},
            id="Process with base dir argument",
        ),
        pytest.param(
            [
                "-b",
                "/tmp/",
                "--output-dir",
                "/tmp/output/",
                "process",
            ],
            run_modules.process,
            {"base_dir": Path("/tmp/"), "output_dir": Path("/tmp/output")},
            id="Process with base dir (short) and output (long) arguments",
        ),
        pytest.param(
            [
                "-l",
                "debug",
                "--cores",
                "16",
                "-f",
                ".spm",
                "process",
            ],
            run_modules.process,
            {"log_level": "debug", "cores": 16, "file_ext": ".spm"},
            id="Process with log_level (short), cores (long) and file extension (short) arguments",
        ),
        pytest.param(
            [
                "create-config",
                "--filename",
                "dummy/config/dir/config.yaml",
            ],
            write_config_with_comments,
            {"filename": Path("dummy/config/dir/config.yaml")},
            id="Create config with output filename",
        ),
        pytest.param(
            [
                "-c",
                "dummy/config/dir/config.yaml",
                "filter",
                "--row-alignment-quantile",
                "0.80",
            ],
            run_modules.filter,
            {
                "config_file": Path("dummy/config/dir/config.yaml"),
                "row_alignment_quantile": 0.80,
            },
            id="Filter with config file (short) argument, row alignment (long)",
        ),
        pytest.param(
            [
                "-c",
                "dummy/config/dir/config.yaml",
                "filter",
                "--gaussian-size",
                "36",
                "--remove-scars",
                "True",
            ],
            run_modules.filter,
            {
                "config_file": Path("dummy/config/dir/config.yaml"),
                "gaussian_size": 36,
                "scars_run": True,
            },
            id="Filter with config file (short) argument, gaussian size (long) and remove scars (long)",
        ),
        pytest.param(
            [
                "-c",
                "dummy/config/dir/config.yaml",
                "slicer",
                "--slices",
                "123",
                "--segment-method",
                "watershed",
                "--minimum-size",
                "10000",
            ],
            run_modules.slicer,
            {"slices": 123, "segment_method": "watershed", "minimum_size": 10000},
            id="Slicer with config file (short) argument, slices (long), segment_method (long), minimum_size (long)",
        ),
        pytest.param(
            [
                "-c",
                "dummy/config/dir/config.yaml",
                "process",
                "--row-alignment-quantile",
                "0.80",
                "--gaussian-size",
                "36",
                "--gaussian-mode",
                "wrap",
                "--remove-scars",
                "True",
                "--scars-removal-iterations",
                "42",
                "--scars-threshold-low",
                "3",
                "--scars-threshold-high",
                "9",
                "--scars-max-scar-width",
                "10",
                "--scars-max-scar-length",
                "20",
                "--slices",
                "123",
                "--segment-method",
                "watershed",
                "--minimum-size",
                "10000",
                "--warnings",
                "True",
            ],
            run_modules.process,
            {
                "config_file": Path("dummy/config/dir/config.yaml"),
                "filter_row_alignment_quantile": 0.80,
                "filter_gaussian_mode": "wrap",
                "filter_scars_run": True,
                "filter_scars_threshold_low": 3,
                "filter_scars_threshold_high": 9,
                "filter_scars_max_scar_width": 10,
                "filter_scars_max_scar_length": 20,
                "slicer_slices": 123,
                "slicer_segment_method": "watershed",
                "slicer_minimum_size": 10000,
                "warnings": True,
            },
            id="Process with config file argument and all parameters",
            # marks=pytest.mark.skip(reason="development"),
        ),
    ],
)
def test_entry_points(
    options: list, expected_function: Callable, expected_args: dict
) -> None:
    """Ensure the correct function is called for each program, and arguments are carried through correctly."""
    returned_args = entry_point(options, testing=True)
    # convert argparse's Namespace object to dictionary
    returned_args_dict = vars(returned_args)
    # check that the correct function is collected
    assert returned_args.func == expected_function
    # check that the argument has successfully been passed through into the dictionary
    for argument, value in expected_args.items():
        assert returned_args_dict[argument] == value


@pytest.mark.parametrize(
    ("config", "target_file"),
    [
        pytest.param(None, "config.yaml", id="default config no --config option"),
        pytest.param(
            "default", "config.yaml", id="default config with --config option"
        ),
    ],
)
def test_entry_point_create_config_file(
    config: str, target_file: str, tmp_path: Path
) -> None:
    """Test that the entry point is able to produce a default config file when asked to."""
    if config is None:
        entry_point(
            manually_provided_args=[
                "create-config",
                "--output-dir",
                f"{tmp_path}",
            ]
        )
    else:
        entry_point(
            manually_provided_args=[
                "create-config",
                "--config",
                f"{config}",
                "--output-dir",
                f"{tmp_path}",
            ]
        )
    assert Path(f"{tmp_path}/{target_file}").is_file()

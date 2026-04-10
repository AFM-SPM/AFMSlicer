"""Test run_modules module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from afmslicer import entry_point

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
SPM_DIR = RESOURCES / "spm"

# NB - Tests are not performed directly on the `run_modules.*()` functions but via the entry_point which runs these.


@pytest.mark.parametrize(
    ("manually_provided_arguments"),
    [
        pytest.param(
            [
                "--config",
                f"{BASE_DIR / 'src' / 'afmslicer' / 'default_config.yaml'}",
                "--base-dir",
                "./tests/resources/spm",
                "--file-ext",
                ".spm",
                "filter",
            ],
            id="basic filter",
        ),
    ],
)
def test_filter(
    manually_provided_arguments: list[str],
    capsys: pytest.CaptureFixture,
) -> None:
    """Test for run_modules.filter()."""
    entry_point.entry_point(manually_provided_arguments)
    captured = capsys.readouterr()
    assert "~~~~~~~~~~~~~~~~~~~~ COMPLETE ~~~~~~~~~~~~~~~~~~~~" in captured.err
    assert "Successfully Processed^1    : 2 (100.0%)" in captured.err


@pytest.mark.parametrize(
    ("manually_provided_arguments"),
    [
        pytest.param(
            [
                "--config",
                f"{BASE_DIR / 'src' / 'afmslicer' / 'default_config.yaml'}",
                "--base-dir",
                "./tests/resources/spm",
                "--file-ext",
                ".spm",
                "slicer",
                "--slices",  # ns-rse 2026-02-18 Need to update TopoStats to update dictionaries with
                "5",  #                   as these do not go into the config
            ],
            id="basic slicer",
        ),
    ],
)
def test_slicer(
    manually_provided_arguments: list[str],
    capsys: pytest.CaptureFixture,
) -> None:
    """Test for run_modules.filter()."""
    entry_point.entry_point(manually_provided_arguments)
    captured = capsys.readouterr()
    assert "~~~~~~~~~~~~~~~~~~~~ COMPLETE ~~~~~~~~~~~~~~~~~~~~" in captured.err
    assert "Successfully Processed^1    : 2 (100.0%)" in captured.err


@pytest.mark.parametrize(
    ("manually_provided_arguments"),
    [
        pytest.param(
            [
                "--config",
                f"{BASE_DIR / 'src' / 'afmslicer' / 'default_config.yaml'}",
                "--base-dir",
                "./tests/resources/spm",
                "--file-ext",
                ".spm",
                "process",
                "--slices",
                "5",
            ],
            id="basic process",
        ),
    ],
)
def test_process(
    manually_provided_arguments: list[str],
    capsys: pytest.CaptureFixture,
    tmp_path: Path,
    snapshot,
) -> None:
    """Test for run_modules.filter()."""
    # prepend the output directory as tmp_path
    manually_provided_arguments = [
        "-o",
        f"{tmp_path}",
        *manually_provided_arguments,
    ]
    entry_point.entry_point(manually_provided_arguments)
    captured = capsys.readouterr()
    assert "~~~~~~~~~~~~~~~~~~~~ COMPLETE ~~~~~~~~~~~~~~~~~~~~" in captured.err
    assert "Successfully Processed^1    : 2 (100.0%)" in captured.err
    assert Path(tmp_path / "all_statistics.csv").is_file()
    assert Path(tmp_path / "color_count.csv").is_file()
    # Loads the results and check against a Syrupy snapshot
    all_statistics_df = pd.read_csv(tmp_path / "all_statistics.csv")
    assert all_statistics_df.to_string() == snapshot
    color_count_df = pd.read_csv(tmp_path / "color_count.csv")
    assert color_count_df.to_string() == snapshot

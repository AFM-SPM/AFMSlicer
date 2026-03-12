"""Tests of the I/O module."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from afmslicer import io


@pytest.mark.parametrize(
    ("data_fixture", "df_fixture"),
    [
        pytest.param(
            "simple_statistics_dictionary",
            "simple_statistics_df",
            id="nested dictionary",
        ),
    ],
)
def test_dict_to_df(data_fixture: str, df_fixture: str, request) -> None:
    """Test for dict_to_df()."""
    data = request.getfixturevalue(data_fixture)[0]
    df = request.getfixturevalue(df_fixture)[0]
    pd.testing.assert_frame_equal(io.dict_to_df(data), df)


@pytest.mark.parametrize(
    ("df_fixture", "filename"),
    [
        pytest.param(
            "simple_statistics_df",
            "statistics.csv",
            id="simple dataframe",
        ),
    ],
)
def test_write_csv(df_fixture: str, filename: str, request, tmp_path: Path) -> None:
    """Test for dict_to_df()."""
    df = request.getfixturevalue(df_fixture)[0]
    io.write_csv(df, outdir=tmp_path, filename=filename)
    assert Path(tmp_path / filename).is_file()

"""Input/Output related functions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def dict_to_df(data: dict[int | str, Any]) -> pd.DataFrame:
    """
    Convert a nested dictionary to a dataframe.

    Layers of pore statistics are converted to a dataframe.

    Parameters
    ----------
    data : dict[int | str, Any]
        Nested dictionary to be converted.

    Returns
    -------
    pd.DataFrame
        Pandas dataframe of statistics.
    """
    df = pd.json_normalize(
        [
            {"layer": layer, "pore": pore, **props}
            for layer, pores in data.items()
            for pore, props in pores.items()
        ]
    )
    df.set_index(["layer", "pore"], inplace=True)  # noqa: PD002
    return df


def write_csv(
    df: pd.DataFrame, outdir: Path | str = "output", filename: str = "results.csv"
) -> None:
    """
    Write a dataframe to ``.csv`` file in ``output`` directory.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe to write to disk.
    outdir : Path | str
        Path where file is to be written.
    filename : str
        Filename for output, defaults to ``results.csv``.
    """
    df.to_csv(Path(outdir) / filename, index=True)

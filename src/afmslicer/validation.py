"""Schema for valdiating AFMSlicer configuration."""

from __future__ import annotations

import os
from pathlib import Path

from schema import And, Or, Schema, Use

AFMSLICER_CONFIG_SCHEMA = Schema(
    {
        "base_dir": Path,
        "output_dir": Path,
        "log_level": Or(
            "debug",
            "info",
            "warning",
            "error",
            error="Invalid value in config for 'log_level', valid values are 'info' (default), "
            "'debug', 'error' or 'warning",
        ),
        "cores": lambda n: 1 <= n <= os.cpu_count(),
        "file_ext": Or(
            ".spm",
            ".asd",
            ".jpk",
            ".ibw",
            ".gwy",
            ".topostats",
            ".stp",
            ".top",
            error="Invalid value in config for 'file_ext', valid values are '.spm', '.jpk', '.ibw', "
            "'.gwy', '.topostats', or '.asd'.",
        ),
        "loading": {
            "channel": str,
            "extract": Or(
                "all",
                "raw",
                "filters",
                error="Invalid value in config for 'extract', valid values are 'all', 'raw' or 'filters'",
            ),
        },
        "filter": {
            "run": Or(
                True,
                False,
                error="Invalid value in config for 'filter.run', valid values are 'True' or 'False'",
            ),
            "row_alignment_quantile": lambda n: 0.0 <= n <= 1.0,
            "threshold_method": Or(
                "absolute",
                "otsu",
                "std_dev",
                error=(
                    "Invalid value in config for 'filter.threshold_method', valid values "
                    "are 'absolute', 'otsu' or 'std_dev'"
                ),
            ),
            "otsu_threshold_multiplier": float,
            "threshold_std_dev": {
                "below": lambda n: n > 0,
                "above": lambda n: n > 0,
            },
            "threshold_absolute": {
                "below": Or(
                    int,
                    float,
                    error=(
                        "Invalid value in config for filter.threshold.absolute.below should be type int or float"
                    ),
                ),
                "above": Or(
                    int,
                    float,
                    error=(
                        "Invalid value in config for filter.threshold.absolute.below should be type int or float"
                    ),
                ),
            },
            "gaussian_size": float,
            "gaussian_mode": Or(
                "nearest",
                error="Invalid value in config for 'filter.gaussian_mode', valid values are 'nearest'",
            ),
            "remove_scars": {
                "run": bool,
                "removal_iterations": lambda n: 0 <= n < 10,
                "threshold_low": lambda n: n > 0,
                "threshold_high": lambda n: n > 0,
                "max_scar_width": lambda n: n >= 1,
                "min_scar_length": lambda n: n >= 1,
            },
        },
        "slice": {
            "n_slices": And(
                Use(int),
                lambda n: n > 0,
                error="Invalid 'n_slices' should be an int > 0.",
            )
        },
    }
)

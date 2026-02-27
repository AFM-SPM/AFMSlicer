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
        "slicing": {
            "slices": And(
                Use(int),
                lambda n: n > 0,
                error="Invalid 'slices' should be an int > 0.",
            ),
            "segment_method": Or(
                "watershed",
                "label",
                error="Invalid value in config for 'slicing.segment_method', valid values are 'watershed' or 'label'",
            ),
            "area": bool,
            "minimum_size": lambda n: n >= 1,
            "centroid": bool,
            "feret_maximum": bool,
        },
        "plotting": {
            "format": Or(
                "png",
                "tiff",
                error="Invalid value in config for 'plotting.format', valid values are 'png' or 'tiff'.",
            )
        },
    }
)

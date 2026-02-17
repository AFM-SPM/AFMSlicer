"""
Run AFMSlicer modules.

This module provides entry points for running AFMSlicer as a command line programme.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy.typing as npt
from loguru import logger
from topostats.classes import TopoStats

from afmslicer.classes import AFMSlicer
from afmslicer.filter import SlicingFilter


# Development Note...
#
# We define functions here that process either all stages (process) or individual stages. These are imported by the
# run_modules sub-module where they serve as input to the partial() function so that images can be run in parallele
# using Pool()
def process_scan(
    topostats_object: TopoStats,
    config: dict[str, Any] | None = None,
) -> dict[str, npt.NDArray] | None:
    """
    Process a single image, filtering, slicing and calculating statistics.

    Parameters
    ----------
    topostats_object : dict[str, Union[npt.NDArray, Path, float]]
        A dictionary with keys 'image', 'img_path' and 'pixel_to_nm_scaling' containing a file or frames' image, it's
        path and it's pixel to namometre scaling value.
    config : dict
        Dictionary of configuration options for processing images.

    Returns
    -------
    tuple(str, bool)
        Tuple of filename and whether it processed correctly.
    """
    config = topostats_object.config if config is None else config
    filter_scan(topostats_object=topostats_object, config=config)
    slicer_scan(topostats_object=topostats_object, config=config)


def filter_scan(
    topostats_object: TopoStats,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Filter an image and save to ''.topostats''.

    Runs just the first key step of flattening images to remove noise, tilt and optionally scars saving to
    ``.topostats`` for subsequent processing and analyses.

    Parameters
    ----------
    topostats_object : TopoStats
        A TopoStats object.
    config : dict, optional
        Dictionary of configuration options for running the Filter stage.
    """
    config = topostats_object.config if config is None else config
    if "run" in config:
        config.pop("run")
    output_dir = (
        Path("output") if config["output_dir"] is None else Path(config["output_dir"])
    )
    output_dir.mkdir(exist_ok=True)
    if "filter" in config:
        filter_config = config["filter"]
        filter_config.pop("run")
    else:
        filter_config = config
    # Flatten Image
    try:
        filters = SlicingFilter(topostats_object, **filter_config)
        filters.filter_image()
        # Save the topostats object to .topostats file.
        # save_topostats_file(
        #     output_dir=output_dir,
        #     filename=str(topostats_object.filename),
        #     topostats_object=topostats_object,
        # )
        logger.info(f"[{topostats_object.filename}] : Filtering complete 😻 ")
        return
    except KeyError as e:
        raise KeyError() from e
    except:  # noqa: E722  # pylint: disable=bare-except
        logger.info(f"[{topostats_object.filename}] : Filtering failed  😿")
        return


# Slicing : slicing_scan() to process a single image, slicing() to process in parallele
def slicer_scan(
    topostats_object: TopoStats,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Flatten images and save to ''.topostats''.

    Runs just the first key step of flattening images to remove noise, tilt and optionally scars saving to
    ''.topostats'' for subsequent processing and analyses.

    Parameters
    ----------
    topostats_object : TopoStats
        A TopoStats object.
    config : dict
        Dictionary of configuration options for running the Slicing stage.
    """
    config = topostats_object.config if config is None else config
    output_dir = (
        Path("output") if config["output_dir"] is None else Path(config["output_dir"])
    )
    output_dir.mkdir(exist_ok=True)

    # Flatten Image
    try:
        if isinstance(topostats_object, AFMSlicer):
            topostats_object.slice_image()
        else:
            topostats_object = AFMSlicer(topostats_object, **config)
            topostats_object.slice_image()

        # Save the topostats object to .topostats file.
        # save_topostats_file(
        #     output_dir=output_dir,
        #     topostats_object=topostats_object,
        #     topostats_version=__release__,
        # )
        logger.info(f"[{topostats_object.filename}] Slicing complete 😻")
        return
    except:  # noqa: E722  # pylint: disable=bare-except
        logger.info(f"[{topostats_object.filename}] Slicing failed 😿")
        return

"""
Run AFMSlicer modules.

This module provides entry points for running AFMSlicer as a command line programme.
"""

from __future__ import annotations

from pathlib import Path

import numpy.typing as npt
from loguru import logger
from topostats.io import save_topostats_file

from afmslicer.filter import Filters


# Development Note...
#
# We define functions here that process either all stages (process) or individual stages. These are imported by the
# run_modules sub-module where they serve as input to the partial() function so that images can be run in parallele using Pool()
def process(
    topostats_object: TopoStats,
    config: dict[str, Any] | None,
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
    filter(topostats_object=topostats_object, config=config)
    statistics = slicer(topostats_object=topostats_object, config=config)
    return statistics


def filter(
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
    output_dir = (
        Path("output") if config["output_dir"] is None else config["output_dir"]
    )
    output_dir.mkdir(exist_ok=True)

    # Flatten Image
    try:
        filters = Filters(topostats_object, **config["filter"])
        # TopoStats performs
        filters.filter_image()

        # Save the topostats object to .topostats file.
        save_topostats_file(
            output_dir=output_dir,
            filename=str(topostats_object.filename),
            topostats_object=topostats_object,
        )
        logger.info(f"Filtering complete for image : {topostats_object.filename}")
        return
    except:  # noqa: E722  # pylint: disable=bare-except
        logger.info(f"Filtering failed for image : {topostats_object.filename}")
        return


# Slicing : slicing_scan() to process a single image, slicing() to process in parallele
def slicer(
    topostats_object: TopoStats,
    config: dict[str, Any],
) -> dict[str, npt.NDArray]:
    """
     flattened images and save to ''.topostats''.

    Runs just the first key step of flattening images to remove noise, tilt and optionally scars saving to
    ''.topostats'' for subsequent processing and analyses.

    Parameters
    ----------
    topostats_object : TopoStats
        A TopoStats object.
    config : dict
        Dictionary of configuration options for running the Slicing stage.

    Returns
    -------
    tuple[str, bool]
        A tuple of the image and a boolean indicating if the image was successfully processed.
    """
    output_dir = (
        Path("output") if config["output_dir"] is None else config["output_dir"]
    )
    output_dir.mkdir(exist_ok=True)

    # Flatten Image
    try:
        if isinstance(topostats_object, AFMSlicer):
            topostats_object.slice_image()
        else:
            topostats_object = AFMSlicer(topostats_object, config=config)
            topostats_object.slice_image()
        image = AFMslicer(topostats_object, **slicing_config)
        slicing.slicing_image()

        # Save the topostats object to .topostats file.
        save_topostats_file(
            output_dir=output_dir,
            filename=str(topostats_object.filename),
            topostats_object=topostats_object,
        )
        logger.info(f"Slicing complete for image : {topostats_object.filename}")
        return slicing.statistics
    except:  # noqa: E722  # pylint: disable=bare-except
        logger.info(f"Slicing failed for image : {topostats_object.filename}")
        return None

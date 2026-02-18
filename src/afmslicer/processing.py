"""
Run AFMSlicer modules.

This module provides entry points for running AFMSlicer as a command line programme.
"""

from __future__ import annotations

import contextlib
from copy import deepcopy
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import ValidationError
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
) -> tuple[str, AFMSlicer]:
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
    config = deepcopy(topostats_object.config) if config is None else config
    _, _ = filter_scan(topostats_object=topostats_object, config=config)
    _, _ = slicer_scan(topostats_object=topostats_object, config=config)
    return (topostats_object.filename, topostats_object)


def filter_scan(
    topostats_object: TopoStats,
    config: dict[str, Any] | None = None,
) -> tuple[str, bool]:
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

    Returns
    -------
    tuple[str, bool]
        Returns a tuple of ``topostats_object.filename`` and a ``bool`` of whether it successfully filtered.
    """
    config = deepcopy(topostats_object.config) if config is None else config
    if "run" in config:
        config.pop("run")
    output_dir = (
        Path("output") if config["output_dir"] is None else Path(config["output_dir"])
    )
    output_dir.mkdir(exist_ok=True)
    if "filter" in config:
        filter_config = config["filter"]
        with contextlib.suppress(KeyError):
            filter_config.pop("run")
    else:
        with contextlib.suppress(KeyError):
            config.pop("output_dir")
        filter_config = config
    # Flatten Image
    # try:
    filters = SlicingFilter(topostats_object, **filter_config)
    filters.filter_image()
    # Save the topostats object to .topostats file.
    # save_topostats_file(
    #     output_dir=output_dir,
    #     filename=str(topostats_object.filename),
    #     topostats_object=topostats_object,
    # )
    logger.info(f"[{topostats_object.filename}] : Filtering complete 😻")
    return (topostats_object.filename, True)
    # except KeyError as e:
    #     raise KeyError() from e
    # except:  # pylint: disable=bare-except
    #     logger.info(f"[{topostats_object.filename}] : Filtering failed 😿")
    #     return (topostats_object.filename, False)


# Slicing : slicing_scan() to process a single image, slicing() to process in parallele
def slicer_scan(
    topostats_object: TopoStats,
    config: dict[str, Any] | None = None,
) -> tuple[str, AFMSlicer]:
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

    Returns
    -------
    tuple[str, AFMSlicer]
        Returns a tuple of the ``AFMSlicer.filename`` and the processed ``AFMSlicer`` object.
    """
    config = deepcopy(topostats_object.config) if config is None else config
    output_dir = (
        Path("output") if config["output_dir"] is None else Path(config["output_dir"])
    )
    output_dir.mkdir(exist_ok=True)

    # Slice Image
    try:
        if isinstance(topostats_object, AFMSlicer):
            topostats_object.slice_image()
        else:
            topostats_object = AFMSlicer(
                image=topostats_object.image,
                image_original=topostats_object.image_original,
                filename=topostats_object.filename,
                img_path=topostats_object.img_path,
                pixel_to_nm_scaling=topostats_object.pixel_to_nm_scaling,
                **config,
                config=config,
            )
            topostats_object.slice_image()

        # Save the topostats object to .topostats file.
        # save_topostats_file(
        #     output_dir=output_dir,
        #     topostats_object=topostats_object,
        #     topostats_version=__release__,
        # )
        logger.info(f"[{topostats_object.filename}] Slicing complete 😻")
        return (topostats_object.filename, topostats_object)
    except ValidationError as ve:
        raise ve
    except:  # noqa: E722  # pylint: disable=bare-except
        logger.info(f"[{topostats_object.filename}] Slicing failed 😿")
        return (topostats_object.filename, topostats_object)

"""
Run AFMSlicer modules.

This module provides entry points for running AFMSlicer as a command line programme.
"""

from __future__ import annotations

import argparse as arg
from collections import defaultdict
from functools import partial
from importlib import resources
from multiprocessing import Pool
from pathlib import Path

from art import tprint
from loguru import logger
from topostats.classes import TopoStats
from topostats.filters import Filters
from topostats.io import (
    LoadScans,
    merge_mappings,
    read_yaml,
    save_topostats_file,
    write_yaml,
)
from topostats.utils import update_config
from topostats.validation import validate_config
from tqdm import tqdm

from afmslicer import AFMSLICER_COMMIT, AFMSLICER_VERSION, logging
from afmslicer.validation import AFMSLICER_CONFIG_SCHEMA


def reconcile_config_args(args: arg.Namespace | None) -> dict:
    """
    Reconcile command line arguments with the default configuration.

    Command line arguments take precedence over the default configuration. If a partial configuration file is specified
    (with '-c' or '--config-file') the defaults are over-ridden by these values (internally the configuration
    dictionary is updated with these values). Any other command line arguments take precedence over both the default and
    those supplied in a configuration file (again the dictionary is updated).

    The final configuration is validated before processing begins.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments passed into TopoStats.

    Returns
    -------
    dict
        The configuration dictionary.
    """
    default_config = read_yaml(resources.files(__package__) / "default_config.yaml")
    if args is not None:
        config_file_arg: str | None = args.config_file
        if config_file_arg is not None:
            config = read_yaml(config_file_arg)
            # Merge the loaded config with the default config to fill in any defaults that are missing
            # Make sure to prioritise the loaded config, so it overrides the default
            config = merge_mappings(map1=default_config, map2=config)
        else:
            # If no config file is provided, use the default config
            config = default_config
    else:
        # If no args are provided, use the default config
        config = default_config
    # Override the config with command line arguments passed in, eg --output_dir ./output/
    if args is not None:
        config = update_config(config, args)
    return config


def parse_configuration(
    args: arg.Namespace | None = None,
) -> dict[str : float | int | Path]:
    """
    Load configurations, validate and check run steps are consistent.

    Parameters
    ----------
    args : arg.Namespace | None
        Arguments.

    Returns
    -------
    tuple[dict, dict]
        Returns the dictionary of configuration options and a dictionary of image files found on the input path.
    """
    # Parse command line options, load config (or default) and update with command line options
    config = reconcile_config_args(args=args)
    logger.remove()
    if "log_level" in vars(args) and vars(args)["log_level"] is not None:
        logging.setup(level=args["log_level"])
    else:
        logging.setup(level=config["log_level"])
    # Validate configuration
    validate_config(
        config, schema=AFMSLICER_CONFIG_SCHEMA, config_type="YAML configuration file"
    )
    # Create base output directory
    config["output_dir"].mkdir(parents=True, exist_ok=True)
    return config


# Development Note...
#
# We define two functions for each entry point/module we wish to process, the first processes an individual method, here
# process_scan(), the second process() runs the function in parallel for all of the images that are to be
# processed. Thus when adding new modules/functionality you should also define two such functions and stick with the
# convention of <processing_name>_scan() for the function that processes an individual scan and <processing_name> for
# the function that runs it in parallel.
def process_scan(
    topostats_object: TopoStats,
    output_dir: str | Path,
    filter_config: dict[str, str | int | float],
    slicing_config: dict[str, int],
) -> tuple(str, bool):
    """
    Process a single image, filtering, slicing and calculating statistics.

    Parameters
    ----------
    topostats_object : dict[str, Union[npt.NDArray, Path, float]]
        A dictionary with keys 'image', 'img_path' and 'pixel_to_nm_scaling' containing a file or frames' image, it's
        path and it's pixel to namometre scaling value.
    output_dir : str | Path
        Directory to which output is written.
    filter_config : dict
        Dictionary of configuration options for running the Filter stage.
    slicing_config : dict
        Dictionary of configuration options for running the Slicing stage.

    Returns
    -------
    tuple(str, bool)
        Tuple of filename and whether it processed correctly.
    """
    filtering_scan(
        topostats_object=topostats_object,
        filter_config=filter_config,
        output_dir=output_dir,
    )
    slicing_scan(
        topostats_object=topostats_object,
        slicing_config=slicing_config,
        output_dir=output_dir,
    )
    # ns-rse 2025-10-10 : Need to return something!
    return (topostats_object.filename, True)


def process(args: arg.Namespace | None = None) -> None:
    """
    Find and process all files.

    Parameters
    ----------
    args : arg.Namespace | None
        Arguments.
    """
    config = parse_configuration(args)
    processing_function = partial(
        process_scan, base_dir=config["base_dir"], filter_config=config["filter"]
    )
    # Ensure we load the original images as we are running the whole pipeline
    if config["file_ext"] == ".topostats":
        config["loading"]["extract"] = "raw"

    # ns-rse 2025-10-10 - Get the list of image files over which to LoadScans
    img_files = {}
    # Get a dictionary of all the image data dictionaries
    # Keys   : image name
    # Values : individual image data
    all_scan_data = LoadScans(img_files, **config["loading"])
    all_scan_data.get_data()
    scan_data_dict = all_scan_data.img_dict

    with Pool(processes=config["cores"]) as pool:
        results = defaultdict()
        with tqdm(
            total=len(img_files),
            desc=f"Processing images from {config['base_dir']}, results are under {config['output_dir']}",
        ) as pbar:
            for (
                img,
                result,
            ) in pool.imap_unordered(
                processing_function,
                scan_data_dict.values(),
            ):
                results[str(img)] = result.dropna(axis=1, how="all")
                pbar.update()
                logger.info(f"[{img.name}] Processing completed.")


# Filter : filtering_scan() to process a single image, filtering() to process in parallel.
def filtering_scan(
    topostats_object: TopoStats,
    filter_config: dict,
    output_dir: str | Path = "output",
) -> tuple[str, bool]:
    """
    Filter an image return the flattened images and save to ''.topostats''.

    Runs just the first key step of flattening images to remove noise, tilt and optionally scars saving to
    ''.topostats'' for subsequent processing and analyses.

    Parameters
    ----------
    topostats_object : TopoStats
        A TopoStats object.
    filter_config : dict
        Dictionary of configuration options for running the Filter stage.
    output_dir : str | Path | None
        Directory to save output to, if ``None`` the defaults to ``output/``.

    Returns
    -------
    tuple[str, bool]
        A tuple of the image and a boolean indicating if the image was successfully processed.
    """
    output_dir = Path("output") if output_dir is None else output_dir
    output_dir.mkdir(exist_ok=True)

    # Flatten Image
    try:
        filters = Filters(topostats_object, **filter_config)
        filters.filter_image()

        # Save the topostats object to .topostats file.
        save_topostats_file(
            output_dir=output_dir,
            filename=str(topostats_object.filename),
            topostats_object=topostats_object,
        )
        logger.info(f"Filtering complete for image : {topostats_object.filename}")
        return (topostats_object.filename, True)
    except:  # noqa: E722  # pylint: disable=bare-except
        logger.info(f"Filtering failed for image : {topostats_object.filename}")
        return (topostats_object.filename, False)


def filtering(args: arg.Namespace | None) -> None:
    """
    Load files from disk and run filtering.

    Parameters
    ----------
    args : arg.Namespace None
        Arguments.
    """
    config = parse_configuration(args)
    # If loading existing .topostats files the images need filtering again so we need to extract the raw image
    if config["file_ext"] == ".topostats":
        config["loading"]["extract"] = "raw"

    # ns-rse 2025-10-10 - Get the list of image files over which to LoadScans
    img_files = {}

    all_scan_data = LoadScans(img_files, **config["loading"])
    all_scan_data.get_data()

    processing_function = partial(
        filtering_scan,
        base_dir=config["base_dir"],
        filter_config=config["filter"],
        output_dir=config["output_dir"],
    )

    with Pool(processes=config["cores"]) as pool:
        results = defaultdict()
        with tqdm(
            total=len(img_files),
            desc=f"Processing images from {config['base_dir']}, results are under {config['output_dir']}",
        ) as pbar:
            for img, result in pool.imap_unordered(
                processing_function,
                all_scan_data.img_dict.values(),
            ):
                results[str(img)] = result
                pbar.update()

                # Display completion message for the image
                logger.info(f"[{img}] Filtering completed.")

    # Write config to file
    write_yaml(config, output_dir=config["output_dir"])
    logger.debug(f"Images processed : {len(results)}")
    # Update config with plotting defaults for printing
    completion_message(config, img_files, images_processed=sum(results.values()))


# Slicing : slicing_scan() to process a single image, slicing() to process in parallele
def slicing_scan(
    topostats_object: TopoStats,
    output_dir: str | Path,
    slicing_config: dict[str:int],
) -> None:
    """
    Filter an image return the flattened images and save to ''.topostats''.

    Runs just the first key step of flattening images to remove noise, tilt and optionally scars saving to
    ''.topostats'' for subsequent processing and analyses.

    Parameters
    ----------
    topostats_object : TopoStats
        A TopoStats object.
    output_dir : str | Path | None
        Directory to save output to, if ``None`` the defaults to ``output/``.
    slicing_config : dict
        Dictionary of configuration options for running the Slicing stage.

    Returns
    -------
    tuple[str, bool]
        A tuple of the image and a boolean indicating if the image was successfully processed.
    """
    output_dir = Path("output") if output_dir is None else output_dir
    output_dir.mkdir(exist_ok=True)

    # Flatten Image
    try:
        # slicing = Slicing(topostats_object, **slicing_config)
        # slicing.slicing_image()

        # Save the topostats object to .topostats file.
        logger.info(slicing_config)
        save_topostats_file(
            output_dir=output_dir,
            filename=str(topostats_object.filename),
            topostats_object=topostats_object,
        )
        logger.info(f"Slicing complete for image : {topostats_object.filename}")
        return (topostats_object.filename, True)
    except:  # noqa: E722  # pylint: disable=bare-except
        logger.info(f"Slicing failed for image : {topostats_object.filename}")
        return (topostats_object.filename, False)


def slicing(args: arg.Namespace | None) -> None:
    """
    Load files from disk and run slicing.

    Parameters
    ----------
    args : arg.Namespace None
        Arguments.
    """
    config, img_files = parse_configuration(args)
    # If loading existing .topostats files the images need filtering again so we need to extract the raw image
    if config["file_ext"] == ".topostats":
        config["loading"]["extract"] = "raw"
    all_scan_data = LoadScans(img_files, **config["loading"])
    all_scan_data.get_data()

    processing_function = partial(
        slicing_scan,
        base_dir=config["base_dir"],
        filter_config=config["filter"],
        output_dir=config["output_dir"],
    )

    with Pool(processes=config["cores"]) as pool:
        results = defaultdict()
        with tqdm(
            total=len(img_files),
            desc=f"Processing images from {config['base_dir']}, results are under {config['output_dir']}",
        ) as pbar:
            for img, result in pool.imap_unordered(
                processing_function,
                all_scan_data.img_dict.values(),
            ):
                results[str(img)] = result
                pbar.update()

                # Display completion message for the image
                logger.info(f"[{img}] Filtering completed.")

    # Write config to file
    write_yaml(config, output_dir=config["output_dir"])
    logger.debug(f"Images processed : {len(results)}")
    # Update config with plotting defaults for printing
    completion_message(config, img_files, images_processed=sum(results.values()))


def completion_message(config: dict, img_files: list, images_processed: int) -> None:
    """
    Print a completion message summarising images processed.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    img_files : list
        List of found image paths.
    images_processed : int
        Number of images processed.
    """
    logger.info(
        "\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
    )
    tprint("AFMSlicer", font="twisted")
    logger.info(
        f"\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COMPLETE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
        f"  AFMSlicer Version           : {AFMSLICER_VERSION}\n"
        f"  AFMSlicer Commit            : {AFMSLICER_COMMIT}\n"
        f"  Base Directory              : {config['base_dir']}\n"
        f"  File Extension              : {config['file_ext']}\n"
        f"  Files Found                 : {len(img_files)}\n"
        f"  Successfully Processed^1    : {images_processed} ({(images_processed * 100) / len(img_files)}%)\n"
        f"  Configuration               : {config['output_dir']}/config.yaml\n\n"
        f"  Email                       : topostats@sheffield.ac.uk\n"
        f"  Documentation               : https://afm-spm.github.io/topostats/\n"
        f"  Source Code                 : https://github.com/ns-rse/AFMSlicer/\n"
        f"  Bug Reports/Feature Request : https://github.com/ns-rse/AFMSlicer/issues/new/choose\n"
        f"  Citation File Format        : https://github.com/ns-rse/AFMSlicer/blob/main/CITATION.cff\n\n"
        f"  ^1 Successful processing of an image is detection of grains and calculation of at least\n"
        f"     grain statistics. If these have been disabled the percentage will be 0.\n\n"
        f"  If you encounter bugs/issues or have feature requests please report them at the above URL\n"
        f"  or email us.\n\n"
        f"  If you have found AFMSlicer useful please consider citing it. A Citation File Format is\n"
        f"  linked above and available from the Source Code page.\n"
        f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
    )

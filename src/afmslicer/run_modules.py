"""Run modules in parallel."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pkgutil import get_data
from pprint import pformat
from typing import Any

import yaml
from art import tprint
from loguru import logger
from topostats.classes import TopoStats
from topostats.config import (
    reconcile_config_args,
)
from topostats.io import (
    LoadScans,
    find_files,
    write_yaml,
)
from topostats.validation import validate_config
from tqdm import tqdm

import afmslicer
from afmslicer import AFMSLICER_BASE_VERSION, AFMSLICER_COMMIT, processing
from afmslicer.validation import AFMSLICER_CONFIG_SCHEMA


def _log_setup(config: dict, args: argparse.Namespace | None, img_files: dict) -> None:
    """
    Log the current configuration.

    Parameters
    ----------
    config : dict
        Dictionary of configuration options.
    args : argparse.Namespace | None
        Arguments function was invoked with.
    img_files : dict
        Dictionary of image files that have been found.
    """
    logger.debug(
        f"Plotting configuration after update :\n{pformat(config['plotting'], indent=4)}"
    )

    logger.info(f"Configuration file loaded from      : {args.config_file}")
    logger.info(f"Scanning for images in              : {config['base_dir']}")
    logger.info(f"Output directory                    : {config['output_dir']!s}")
    logger.info(f"Looking for images with extension   : {config['file_ext']}")
    logger.info(
        f"Images with extension {config['file_ext']} in {config['base_dir']} : {len(img_files)}"
    )
    logger.info(f"Slices per image                    : {config['slicing']['slices']}")
    if len(img_files) == 0:
        logger.error(
            f"No images with extension {config['file_ext']} in {config['base_dir']}"
        )
        logger.error("Please check your configuration and directories.")
        sys.exit()
    logger.debug(f"Configuration after update         : \n{pformat(config, indent=4)}")


def _set_logging(log_level: str | None) -> None:
    """
    Set up loguru logging.

    Parameters
    ----------
    log_level : str
        Logging level.
    """
    logger.remove()
    logger.add(sys.stderr, level=log_level)


def _parse_configuration(args: argparse.Namespace | None = None) -> tuple[dict, dict]:
    """
    Load configurations, validate and check run steps are consistent.

    Parameters
    ----------
    args : argparse.Namespace | None
        Arguments.

    Returns
    -------
    tuple[dict, dict]
        Returns the dictionary of configuration options and a dictionary of image files found on the input path.
    """
    # Parse command line options, load config (or default) and update with command line options
    default_config = get_data(
        package=afmslicer.__package__, resource="default_config.yaml"
    )
    default_config = yaml.full_load(default_config)
    config = reconcile_config_args(args=args, default_config=default_config)
    # Validate configuration
    validate_config(
        config, schema=AFMSLICER_CONFIG_SCHEMA, config_type="YAML configuration file"
    )

    # Set logging level
    _set_logging(log_level=config["log_level"].upper())

    # Create base output directory
    config["output_dir"].mkdir(parents=True, exist_ok=True)

    # Ensures each image has all plotting options which are passed as **kwargs
    img_files = find_files(config["base_dir"], file_ext=config["file_ext"])
    _log_setup(config, args, img_files)
    return config, img_files


def _load_scans(
    img_files: dict[str, Any], config: dict[str, Any]
) -> dict[str, TopoStats]:
    """
    Load all scans to dictionary.

    Parameters
    ----------
    img_files : dict[str, Any]
        Dictionary of filenames and paths.
    config : dict[str, Any]
        Dictionary of configuration.

    Returns
    -------
    dict[str, TopoStats]
        Dictionary of TopoStats objects.
    """
    all_scan_data = LoadScans(img_files, config=config)
    all_scan_data.get_data()
    return all_scan_data.img_dict


def process(args: argparse.Namespace | None = None) -> None:
    """
    Process images in parallel.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments to process images with.
    """
    config, img_files = _parse_configuration(args)
    processing_function = partial(processing.process_scan, config=config)

    # ns-rse 2026-02-18 Bump number of slices up in config need to fix this, should get directly from config within
    #                   AFMSlicer.config["slicing"]["slices"] when needed
    config["slices"] = config["slicing"]["slices"]

    # Get a dictionary of all the image data dictionaries.
    scan_data_dict = _load_scans(img_files, config)

    with Pool(processes=config["cores"]) as pool:
        processed_all = defaultdict()
        with tqdm(
            total=len(img_files),
            desc=f"Processing images from {config['base_dir']}, results are under {config['output_dir']}",
        ) as pbar:
            for (
                filename,
                processed_image,
            ) in pool.imap_unordered(
                processing_function,
                scan_data_dict.values(),
            ):
                # Append each images returned dataframes to the dictionaries
                if processed_image is not None:
                    processed_all[str(filename)] = processed_image

                pbar.update()
                # Display completion message for the image
                logger.info(f"[{filename}] Processing completed.")
    # Concatenate all the dictionary's values into a dataframe. Ignore the keys since
    # the dataframes have the file names in them already.
    # statistics_all_df = pd.concat(statistics_all.values())
    # statistics_all_df.to_csv(config["output_dir"] / "statistics.csv")
    # Write config to file
    write_yaml(config, output_dir=config["output_dir"])
    images_processed = len(processed_all)
    completion_message(config, img_files, images_processed)


def filter(args: argparse.Namespace | None = None) -> None:  # pylint: disable=redefined-builtin
    """
    Filter/flatten images in parallel.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments to process images with.
    """
    config, img_files = _parse_configuration(args)
    processing_function = partial(processing.filter_scan, config=config)

    # Get a dictionary of all the image data dictionaries.
    scan_data_dict = _load_scans(img_files, config)

    with Pool(processes=config["cores"]) as pool:
        images_processed = 0
        success = 0
        with tqdm(
            total=len(img_files),
            desc=f"Filtering images from {config['base_dir']}, results are under {config['output_dir']}",
        ) as pbar:
            for filename, processed in pool.imap_unordered(
                processing_function,
                scan_data_dict.values(),
            ):
                images_processed += 1
                if processed:
                    success += 1
                pbar.update()
                # Display completion message for the image
                logger.info(f"[{filename}] Processing completed.")
    # Write config to file
    write_yaml(config, output_dir=config["output_dir"])
    completion_message(config, img_files, images_processed)


def slicer(args: argparse.Namespace | None = None) -> None:
    """
    Slice images in parallel.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments to process images with.
    """
    config, img_files = _parse_configuration(args)
    processing_function = partial(processing.slicer_scan, config=config)

    # Get a dictionary of all the image data dictionaries.
    scan_data_dict = _load_scans(img_files, config)

    # ns-rse 2026-02-18 Bump number of slices up in config need to fix this, should get directly from config within
    #                   AFMSlicer.config["slicing"]["slices"] when needed
    config["slices"] = config["slicing"]["slices"]
    images_processed = 0
    with Pool(processes=config["cores"]) as pool:
        processed_all = defaultdict()
        with tqdm(
            total=len(img_files),
            desc=f"Slicing images from {config['base_dir']}, results are under {config['output_dir']}",
        ) as pbar:
            for filename, processed_object in pool.imap_unordered(
                processing_function,
                scan_data_dict.values(),
            ):
                # Append each images returned dataframes to the dictionaries
                if processed_object is not None:
                    processed_all[str(filename)] = processed_object
                pbar.update()
                # Display completion message for the image
                logger.info(f"[{filename}] Processing completed.")
    # Write config to file
    write_yaml(config, output_dir=config["output_dir"])
    images_processed = len(processed_all)
    completion_message(config, img_files, images_processed)


def completion_message(
    config: dict, img_files: dict[str, Any], images_processed: int
) -> None:
    """
    Print a completion message summarising images processed.

    Parameters
    ----------
    config : dict
        Configuration dictionary.
    img_files : list
        List of found image paths.
    images_processed : int
        Pandas DataFrame of results.
    """
    logger.info(
        "\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
    )
    tprint("AFMSlicer", font="twisted")
    logger.info(
        f"\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COMPLETE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
        f"  AFMSlicer Version           : {AFMSLICER_BASE_VERSION}\n"
        f"  AFMSlicer Commit            : {AFMSLICER_COMMIT}\n"
        f"  Base Directory              : {config['base_dir']}\n"
        f"  File Extension              : {config['file_ext']}\n"
        f"  Files Found                 : {len(img_files)}\n"
        f"  Successfully Processed^1    : {images_processed} ({(images_processed * 100) / len(img_files)}%)\n"
        f"  All statistics              : {config['output_dir']!s}/statistics.csv\n"
        f"  Configuration               : {config['output_dir']}/config.yaml\n\n"
        f"  Email                       : afmslicer@sheffield.ac.uk\n"
        f"  Documentation               : https://afm-spm.github.io/afmslicer/\n"
        f"  Source Code                 : https://github.com/AFM-SPM/AFMSlicer/\n"
        f"  Bug Reports/Feature Request : https://github.com/AFM-SPM/AFMSlicer/issues/new/choose\n"
        f"  Citation File Format        : https://github.com/AFM-SPM/AFMSlicer/blob/main/CITATION.cff\n\n"
        f"  ^1 Successful processing of an image is slicing an image, calculating and plotting\n"
        f"     statistics. If these have been disabled the percentage will be 0.\n\n"
        f"  If you encounter bugs/issues or have feature requests please report them at the above URL\n"
        f"  or email us.\n\n"
        f"  If you have found AFMSlicer useful please consider citing it. A Citation File Format is\n"
        f"  linked above and available from the Source Code page.\n"
        f"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n"
    )

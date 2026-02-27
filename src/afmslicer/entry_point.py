"""
Entry point for all TopoStats programs.

Parses command-line arguments and passes input on to the relevant functions / modules.
"""

from __future__ import annotations

import argparse as arg
import sys
from pathlib import Path
from typing import Any

from topostats.config import write_config_with_comments

from afmslicer import __version__, run_modules

# pylint: disable=too-many-statements


def afmslicer_parser() -> arg.ArgumentParser:
    """
    Create a parser for reading options.

    Creates a parser, with multiple sub-parsers for reading options to run ``afmslicer``.

    Returns
    -------
    arg.ArgumentParser
        Argument parser.
    """
    parser = arg.ArgumentParser(
        description="Run various programs relating to AFM data. Add the name of the program you wish to run."
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"Installed version of AFMSlicer: {__version__}",
        help="Report the current version of AFMSlicer that is installed",
    )
    parser.add_argument(
        "-c",
        "--config-file",
        dest="config_file",
        type=Path,
        required=False,
        help="Path to a YAML configuration file.",
    )
    parser.add_argument(
        "-s",
        "--summary-config",
        dest="summary_config",
        required=False,
        help="Path to a YAML configuration file for summary plots and statistics.",
    )
    parser.add_argument(
        "--matplotlibrc",
        dest="matplotlibrc",
        type=Path,
        required=False,
        help="Path to a matplotlibrc file.",
    )
    parser.add_argument(
        "-b",
        "--base-dir",
        dest="base_dir",
        type=Path,
        required=False,
        help="Base directory to scan for images.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=Path,
        required=False,
        help="Output directory to write results to.",
    )
    parser.add_argument(
        "-l",
        "--log-level",
        dest="log_level",
        type=str,
        required=False,
        help="Logging level to use, default is 'info' for verbose output use 'debug'.",
    )
    parser.add_argument(
        "-j",
        "--cores",
        dest="cores",
        type=int,
        required=False,
        help="Number of CPU cores to use when processing.",
    )
    parser.add_argument(
        "-f",
        "--file-ext",
        dest="file_ext",
        type=str,
        required=False,
        help="File extension to scan for.",
    )
    parser.add_argument(
        "--output-stats",
        dest="output_stats",
        type=str,
        required=False,
        help="'basic' (image and grain) or 'full' (image, grain, branch and molecule) statistics written to CSV.",
    )
    parser.add_argument(
        "--channel",
        dest="channel",
        type=str,
        required=False,
        help="Channel to extract.",
    )
    parser.add_argument(
        "--extract",
        dest="extract",
        type=str,
        required=False,
        help="Array to extract when loading '.topostats' files.",
    )

    subparsers = parser.add_subparsers(
        title="program", description="Available programs, listed below:", dest="module"
    )

    # Sub-parser for processing
    process_parser = subparsers.add_parser(
        "process",
        description="Process all images, slicing and summarising statistics across slices and plotting results.",
        help="Process all images, slicing and summarising statistics across slices and plotting results.",
    )
    process_parser.add_argument(
        "--row-alignment-quantile",
        dest="filter_row_alignment_quantile",
        type=float,
        required=False,
        help="Lower values may improve flattening of larger features.",
    )
    process_parser.add_argument(
        "--gaussian-size",
        dest="filter_gaussian_size",
        type=float,
        required=False,
        help="Gaussian blur intensity in pixels.",
    )
    process_parser.add_argument(
        "--gaussian-mode",
        dest="filter_gaussian_mode",
        type=str,
        required=False,
        help="Gaussian blur method. Options are 'nearest' (default), 'reflect', 'constant', 'mirror' or 'wrap'.",
    )
    process_parser.add_argument(
        "--remove-scars",
        dest="filter_scars_run",
        type=bool,
        required=False,
        help="Whether to remove scars.",
    )
    process_parser.add_argument(
        "--scars-removal-iterations",
        dest="filter_scars_removal_iterations",
        type=int,
        required=False,
        help="Number of times to run scar removal",
    )
    process_parser.add_argument(
        "--scars-threshold-low",
        dest="filter_scars_threshold_low",
        type=float,
        required=False,
        help="Lower values make scar removal more sensitive",
    )
    process_parser.add_argument(
        "--scars-threshold-high",
        dest="filter_scars_threshold_high",
        type=float,
        required=False,
        help="Lower values make scar removal more sensitive",
    )
    process_parser.add_argument(
        "--scars-max-scar-width",
        dest="filter_scars_max_scar_width",
        type=int,
        required=False,
        help="Maximum thickness of scars in pixels",
    )
    process_parser.add_argument(
        "--scars-max-scar-length",
        dest="filter_scars_max_scar_length",
        type=int,
        required=False,
        help="Maximum length of scars in pixels",
    )
    process_parser.add_argument(
        "--slices",
        dest="slicer_slices",
        type=int,
        required=False,
        help="Number of slices to take through the image.",
    )
    process_parser.add_argument(
        "--segment-method",
        dest="slicer_segment_method",
        required=False,
        type=str,
        help="Method for segmenting images, options are 'label' or 'watershed'.",
    )
    process_parser.add_argument(
        "--area",
        dest="slicer_area",
        required=False,
        type=bool,
        help="Whether to calculate the area of pores on each slice.",
    )
    process_parser.add_argument(
        "--minimum-size",
        dest="slicer_minimum_size",
        required=False,
        type=int,
        help="Minimum size in nanometres squared of objects to retain, <= minimum_area are masked & excluded.",
    )
    process_parser.add_argument(
        "--centroid",
        dest="slicer_centroid",
        required=False,
        type=bool,
        help="Whether to calculate the centroid of pores on each slice.",
    )
    process_parser.add_argument(
        "--feret-maximum",
        dest="slicer_feret_maximum",
        required=False,
        type=bool,
        help="Whether to calculate the maximum feret distance of pores on each slice.",
    )
    process_parser.add_argument(
        "--warnings",
        dest="warnings",
        type=bool,
        required=False,
        help="Whether to ignore warnings.",
    )
    # Run the relevant function with the arguments
    process_parser.set_defaults(func=run_modules.process)

    # Sub-parser for filtering/flattening images
    filter_parser = subparsers.add_parser(
        "filter",
        description="Load and filter images, saving as .afmslicer files for subsequent processing.",
        help="Load and filter images, saving as .afmslicer files for subsequent processing.",
    )
    filter_parser.add_argument(
        "--row-alignment-quantile",
        dest="row_alignment_quantile",
        type=float,
        required=False,
        help="Lower values may improve flattening of larger features.",
    )
    filter_parser.add_argument(
        "--gaussian-size",
        dest="gaussian_size",
        type=float,
        required=False,
        help="Gaussian blur intensity in pixels.",
    )
    filter_parser.add_argument(
        "--gaussian-mode",
        dest="gaussian_mode",
        type=str,
        required=False,
        help="Gaussian blur method. Options are 'nearest' (default), 'reflect', 'constant', 'mirror' or 'wrap'.",
    )
    filter_parser.add_argument(
        "--remove-scars",
        dest="scars_run",
        type=bool,
        required=False,
        help="Whether to remove scars.",
    )
    filter_parser.add_argument(
        "--scars-removal-iterations",
        dest="scars_removal_iterations",
        type=int,
        required=False,
        help="Number of times to run scar removal",
    )
    filter_parser.add_argument(
        "--scars-threshold-low",
        dest="scars_threshold_low",
        type=float,
        required=False,
        help="Lower values make scar removal more sensitive",
    )
    filter_parser.add_argument(
        "--scars-threshold-high",
        dest="scars_threshold_high",
        type=float,
        required=False,
        help="Lower values make scar removal more sensitive",
    )
    filter_parser.add_argument(
        "--scars-max-scar-width",
        dest="scars_max_scar_width",
        type=int,
        required=False,
        help="Maximum thickness of scars in pixels",
    )
    filter_parser.add_argument(
        "--scars-max-scar-length",
        dest="scars_max_scar_length",
        type=int,
        required=False,
        help="Maximum length of scars in pixels",
    )
    # Run the relevant function with the arguments
    filter_parser.set_defaults(func=run_modules.filter)

    # Sub-parser for slicing images
    slicer_parser = subparsers.add_parser(
        "slicer",
        description="Load and slice images, saving as .afmslicer files for subsequent processing.",
        help="Load and slice images, saving as .afmslicer files for subsequent processing.",
    )
    slicer_parser.add_argument(
        "--slices",
        dest="slices",
        type=int,
        required=False,
        help="Number of slices to take through the image.",
    )
    slicer_parser.add_argument(
        "--segment-method",
        dest="segment_method",
        required=False,
        type=str,
        help="Method for segmenting images, options are 'label' or 'watershed'.",
    )
    slicer_parser.add_argument(
        "--area",
        dest="area",
        required=False,
        type=bool,
        help="Whether to calculate the area of pores on each slice.",
    )
    slicer_parser.add_argument(
        "--minimum-size",
        dest="minimum_size",
        required=False,
        type=int,
        help="Minimum size in nanometres squared of objects to retain, <= minimum_area are masked & excluded.",
    )
    slicer_parser.add_argument(
        "--centroid",
        dest="centroid",
        required=False,
        type=bool,
        help="Whether to calculate the centroid of pores on each slice.",
    )
    slicer_parser.add_argument(
        "--feret-maximum",
        dest="feret_maximum",
        required=False,
        type=bool,
        help="Whether to calculate the maximum feret distance of pores on each slice.",
    )
    slicer_parser.set_defaults(func=run_modules.slicer)

    create_config_parser = subparsers.add_parser(
        "create-config",
        description="Create a configuration file using the defaults.",
        help="Create a configuration file using the defaults.",
    )
    create_config_parser.add_argument(
        "-f",
        "--filename",
        dest="filename",
        type=Path,
        required=False,
        help="Name of YAML file to save configuration to (default 'config.yaml').",
    )
    create_config_parser.add_argument(
        "-o",
        "--output-dir",
        dest="output_dir",
        type=Path,
        required=False,
        default="./",
        help="Path to where the YAML file should be saved (default './' the current directory).",
    )
    create_config_parser.add_argument(
        "-c",
        "--config",
        dest="config",
        type=str,
        default="default",
        help="Configuration to use, currently 'default', 'simple', 'mplstyle' and 'var_to_label' are supported.",
    )
    create_config_parser.set_defaults(func=write_config_with_comments)

    # Return the parser (and its subparsers!)
    return parser


def entry_point(
    manually_provided_args: list[Any] | None = None, testing: bool = False
) -> None | arg.Namespace:
    """
    Entry point for all AFMSlicer programs.

    Main entry point for running 'afmslicer' which allows the different processing, plotting and testing modules to be
    run.

    Parameters
    ----------
    manually_provided_args : None
        Manually provided arguments.
    testing : bool
        Whether testing is being carried out.

    Returns
    -------
    None
        Function does not return anything.
    """
    # Create AFMSlicer parser
    parser = afmslicer_parser()
    args = (
        parser.parse_args()
        if manually_provided_args is None
        else parser.parse_args(manually_provided_args)
    )

    # If no module has been specified print help and exit
    if not args.module:
        parser.print_help()
        sys.exit()
    if testing:
        return args

    # Run the specified module(s)
    args.func(args)

    return None

"""Fixtures for testing."""

from __future__ import annotations

import importlib.resources as pkg_resources

import numpy as np
import pytest

from afmslicer.slicer import AFMSlicer


@pytest.fixture
def default_config() -> dict[
    str : int | float | str | list | dict[str : int | float | str | list]
]:
    """Sample configuration"""
    config = pkg_resources.open_text(__package__, "default_config.yaml")
    # Modify parameters for all tests here
    config["filter"]["remove_scars"]["run"] = True
    return config


@pytest.fixture
def afmslicer_basic() -> AFMSlicer:
    """
    A simple AFMSlicer object with just the heights and metadata.

    On instantiation the image should be sliced by the ``__post_init__()`` method using parameters derived from the data
    itself.
    """
    return AFMSlicer(
        image=np.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0],
                [0, 1, 2, 3, 3, 3, 3, 3, 2, 1, 0],
                [0, 1, 2, 3, 4, 4, 4, 3, 2, 1, 0],
                [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0],
                [0, 1, 2, 3, 4, 4, 4, 3, 2, 1, 0],
                [0, 1, 2, 3, 3, 3, 3, 3, 2, 1, 0],
                [0, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        filename="simple_afmslice",
        img_path="tmp",
        pixel_to_nm_scaling=1.0,
        slices=5,
    )


@pytest.fixture
def afmslicer_with_attributes() -> AFMSlicer:
    """
    An AFMSlicer object with heights and user specified min, max, layers and metadata.

    On instantiation the image should be sliced by the ``__post_init__()`` method using the supplied parameters.
    """
    return AFMSlicer(
        image=np.asarray(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0],
                [0, 1, 2, 3, 3, 3, 3, 3, 2, 1, 0],
                [0, 1, 2, 3, 4, 4, 4, 3, 2, 1, 0],
                [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0],
                [0, 1, 2, 3, 4, 4, 4, 3, 2, 1, 0],
                [0, 1, 2, 3, 3, 3, 3, 3, 2, 1, 0],
                [0, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0],
                [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        ),
        filename="simple_afmslice_with_attr",
        img_path="tmp",
        pixel_to_nm_scaling=0.5,
        slices=2,
        min_height=1.0,
        max_height=4.0,
    )

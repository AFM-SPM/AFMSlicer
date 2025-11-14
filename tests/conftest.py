"""Fixtures for testing."""

from __future__ import annotations

from pathlib import Path
from pkgutil import get_data
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import yaml
from topostats.filters import Filters
from topostats.io import LoadScans

from afmslicer.classes import AFMSlicer  # type: ignore[import-not-found]

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
RESOURCES_SLICER = RESOURCES / "slicer"
RESOURCES_SPM = RESOURCES / "spm"


@pytest.fixture(name="default_config")
def fixture_default_config() -> dict[
    str, int | float | str | list[Any] | dict[str, int | float | str | list[Any]]
]:
    """Sample configuration"""
    default_config_file: bytes | None = get_data(
        package="afmslicer", resource="default_config.yaml"
    )
    config = yaml.safe_load(default_config_file.decode("utf-8"))  # type: ignore[union-attr]
    # Modify parameters for all tests here
    config["filter"]["remove_scars"]["run"] = True
    return config  # type: ignore[no-any-return]


@pytest.fixture(name="pyramid_array")
def fixture_pyramid_array() -> npt.NDArray[np.int32]:
    """Simple pyramidal two-dimensional numpy array."""
    return np.asarray(
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
        ],
        dtype=np.int32,
    )


@pytest.fixture
def pyramid_array_sliced() -> npt.NDArray[np.float64]:
    """Simple pyramidal two-dimensional numpy array sliced 5 times."""
    with np.load(RESOURCES_SLICER / "pyramid_array_sliced.npz") as data:
        return data["arr_0"]


@pytest.fixture
def pyramid_segment_label_5() -> npt.NDArray[np.int32]:
    """A 5-layer sliced image with each layer segmented."""
    return np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ],
        dtype=np.int32,
    )


@pytest.fixture
def pyramid_array_mask_2() -> npt.NDArray[np.bool]:
    """Simple pyramidal two-dimensional numpy array stacked 2 times and masked for thinner layer."""
    return np.asarray(
        [
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 0],
                [1, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 0],
                [1, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 0],
                [1, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
        ]
    )


@pytest.fixture
def pyramid_array_mask_stacked_2() -> npt.NDArray[np.bool]:
    """Simple pyramidal two-dimensional numpy array stacked 2 times and masked for thinner layer."""
    return np.asarray(
        [
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 0],
                [1, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 0],
                [1, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
                [1, 1],
                [1, 1],
                [1, 1],
                [1, 0],
                [1, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [1, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
            ],
        ]
    )


@pytest.fixture
def pyramid_array_sliced_mask() -> npt.NDArray[np.float64]:
    """Simple pyramidal two-dimensional numpy array sliced 5 times."""
    with np.load(RESOURCES_SLICER / "pyramid_array_sliced_mask.npz") as data:
        return data["arr_0"]


@pytest.fixture
def pyramid_array_sliced_mask_segment() -> npt.NDArray[np.float64]:
    """Simple pyramidal two-dimensional numpy array sliced 5 times."""
    with np.load(RESOURCES_SLICER / "pyramid_array_sliced_mask_segment.npz") as data:
        return data["arr_0"]


@pytest.fixture
def pyramid_height_array_5(
    pyramid_array: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    """Repeated layers (n = 5) of the ``pyramid_array``."""
    array = pyramid_array.copy()
    return np.repeat(array[:, :, np.newaxis], 5, axis=2)


@pytest.fixture
def pyramid_height_array_2(
    pyramid_array: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    """Repeated layers (n = 2) of the ``pyramid_array``."""
    array = pyramid_array.copy()
    return np.repeat(array[:, :, np.newaxis], 2, axis=2)


@pytest.fixture
def afmslicer_basic(pyramid_array: npt.NDArray[np.int32]) -> AFMSlicer:
    """
    A simple AFMSlicer object with just the heights and metadata.

    On instantiation the image should be sliced by the ``__post_init__()`` method using parameters derived from the data
    itself.
    """
    return AFMSlicer(
        image=pyramid_array,
        filename="simple_afmslice",
        img_path="tmp",
        pixel_to_nm_scaling=1.0,
        slices=5,
        segment_method="label",
    )


@pytest.fixture
def afmslicer_with_attributes(pyramid_array: npt.NDArray[np.int32]) -> AFMSlicer:
    """
    An AFMSlicer object with heights and user specified min, max, layers and metadata.

    On instantiation the image should be sliced by the ``__post_init__()`` method using the supplied parameters.
    """
    return AFMSlicer(
        image=pyramid_array,
        filename="simple_afmslice_with_attr",
        img_path="tmp",
        pixel_to_nm_scaling=0.5,
        slices=2,
        min_height=1.0,
        max_height=4.0,
        segment_method="label",
    )


# Square of 5x5x5
# Volume : 125
# Centroid : (3, 3, 3)
@pytest.fixture(name="square_array")
def fixture_square_array() -> npt.NDArray[np.int32]:
    """Simple square two-dimensional numpy array."""
    return np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 5, 5, 5, 5, 5, 0],
            [0, 5, 5, 5, 5, 5, 0],
            [0, 5, 5, 5, 5, 5, 0],
            [0, 5, 5, 5, 5, 5, 0],
            [0, 5, 5, 5, 5, 5, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )


@pytest.fixture
def square_array_sliced() -> npt.NDArray[np.float64]:
    """Simple square two-dimensional numpy array sliced 5 times."""
    with np.load(RESOURCES_SLICER / "square_array_sliced.npz") as data:
        return data["arr_0"]


@pytest.fixture
def square_array_sliced_mask() -> npt.NDArray[np.float64]:
    """Simple squareal two-dimensional numpy array sliced 5 times."""
    with np.load(RESOURCES_SLICER / "square_array_sliced_mask.npz") as data:
        return data["arr_0"]


@pytest.fixture
def square_array_sliced_mask_segment() -> npt.NDArray[np.float64]:
    """Simple squareal two-dimensional numpy array sliced 5 times."""
    with np.load(RESOURCES_SLICER / "square_array_sliced_mask_segment.npz") as data:
        return data["arr_0"]


@pytest.fixture(name="basic_three_segments")
def fixture_basic_three_segments() -> npt.NDArray[np.int32]:
    """A basic two-dimensional binary numpy array with three objects to be labelled."""
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 1, 0, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )


@pytest.fixture
def three_layer_three_segments(
    basic_three_segments: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    """A three-dimensiomal binary numpy array with three objects in each layer to be labelled."""
    return np.stack(
        (basic_three_segments, basic_three_segments, basic_three_segments), axis=2
    )


@pytest.fixture
def three_layer_three_segments_label() -> npt.NDArray[np.int32]:
    """A three-dimensional numpy array with three labelled objects in each layer."""
    single_layer = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 2, 2, 2, 0],
            [0, 1, 0, 0, 1, 0, 2, 0, 2, 0],
            [0, 1, 1, 1, 1, 0, 2, 0, 2, 0],
            [0, 0, 0, 0, 0, 0, 2, 0, 2, 0],
            [0, 3, 3, 3, 3, 0, 2, 0, 2, 0],
            [0, 3, 0, 0, 3, 0, 2, 2, 2, 0],
            [0, 3, 3, 3, 3, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    return np.stack((single_layer, single_layer, single_layer), axis=2)


@pytest.fixture
def three_layer_three_segments_watershed() -> npt.NDArray[np.int32]:
    """A three-dimensional numpy array with three watershed objects in each layer."""
    single_layer = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 2, 2, 2, 1, 1, 3, 1, 1],
            [1, 1, 2, 2, 2, 1, 1, 3, 3, 1],
            [1, 1, 1, 1, 1, 1, 1, 3, 3, 1],
            [1, 1, 1, 1, 1, 1, 1, 3, 3, 1],
            [1, 1, 1, 1, 1, 1, 1, 3, 3, 1],
            [1, 1, 4, 4, 4, 1, 1, 3, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=np.int32,
    )
    return np.stack((single_layer, single_layer, single_layer), axis=2)


@pytest.fixture(name="height_array")
def fixture_height_multiple_objects() -> npt.NDArray[np.int32]:
    """A two-dimensional numpy array of heights."""
    return np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0, 1, 2, 2, 2, 2, 2, 1, 0],
            [0, 1, 2, 3, 3, 3, 3, 3, 2, 1, 0, 1, 2, 3, 3, 3, 2, 1, 0],
            [0, 1, 2, 3, 4, 4, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1, 0],
            [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 1, 2, 3, 3, 3, 2, 1, 0],
            [0, 1, 2, 3, 4, 4, 4, 3, 2, 1, 0, 1, 2, 2, 2, 2, 2, 1, 0],
            [0, 1, 2, 3, 3, 3, 3, 3, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0],
            [0, 1, 2, 2, 2, 2, 1, 0, 0, 1, 2, 3, 3, 3, 3, 3, 2, 1, 0],
            [0, 1, 2, 2, 2, 2, 1, 0, 0, 1, 2, 3, 4, 4, 4, 3, 2, 1, 0],
            [0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0],
            [0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0],
            [0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 4, 4, 4, 3, 2, 1, 0],
            [0, 1, 2, 3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 3, 3, 3, 2, 1, 0],
            [0, 1, 2, 2, 2, 2, 1, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )


@pytest.fixture(name="sample1_spm")
def fixture_sample1_spm(default_config: dict[str, Any]) -> tuple[npt.NDArray, float]:
    """Load an image and filter/flatten it ready for analysis."""
    scan_loader = LoadScans(
        [RESOURCES_SPM / "sample1.spm"], channel="Height", config=default_config
    )
    scan_loader.get_data()
    # @ns-rse 2025-11-07 Expect TopoStats to change as TopoStats class will include the configuration and the arguments
    # will be optional.
    _filter = Filters(
        topostats_object=scan_loader.img_dict["sample1"],
        threshold_std_dev=default_config["filter"]["threshold_std_dev"],
        threshold_method=default_config["filter"]["threshold_method"],
        threshold_absolute=default_config["filter"]["threshold_absolute"],
        otsu_threshold_multiplier=default_config["filter"]["otsu_threshold_multiplier"],
        gaussian_size=default_config["filter"]["gaussian_size"],
        gaussian_mode=default_config["filter"]["gaussian_mode"],
        row_alignment_quantile=default_config["filter"]["row_alignment_quantile"],
        remove_scars=default_config["filter"]["remove_scars"],
    )
    _filter.filter_image()
    return (_filter.image, _filter.pixel_to_nm_scaling)


@pytest.fixture(name="sample2_spm")
def fixture_sample2_spm(default_config: dict[str, Any]) -> tuple[npt.NDArray, float]:
    """Load an image and filter/flatten it ready for analysis."""
    scan_loader = LoadScans(
        [RESOURCES_SPM / "sample2.spm"], channel="Height", config=default_config
    )
    scan_loader.get_data()
    # @ns-rse 2025-11-07 Expect TopoStats to change as TopoStats class will include the configuration and the arguments
    # will be optional.
    _filter = Filters(
        topostats_object=scan_loader.img_dict["sample2"],
        threshold_std_dev=default_config["filter"]["threshold_std_dev"],
        threshold_method=default_config["filter"]["threshold_method"],
        threshold_absolute=default_config["filter"]["threshold_absolute"],
        otsu_threshold_multiplier=default_config["filter"]["otsu_threshold_multiplier"],
        gaussian_size=default_config["filter"]["gaussian_size"],
        gaussian_mode=default_config["filter"]["gaussian_mode"],
        row_alignment_quantile=default_config["filter"]["row_alignment_quantile"],
        remove_scars=default_config["filter"]["remove_scars"],
    )
    _filter.filter_image()
    return (_filter.image, _filter.pixel_to_nm_scaling)


# Fixtures for testing the instantiation of AFMSlicer with sample1.spm
@pytest.fixture(name="afmslicer_sample1")
def fixture_afmslicer_sample1(sample1_spm) -> AFMSlicer:
    """Fixture of AFMSlicer using sample1.spm."""
    height, pixel_to_nm_scaling = sample1_spm
    return AFMSlicer(
        image=height,
        filename="sample1",
        img_path="tmp",
        pixel_to_nm_scaling=pixel_to_nm_scaling,
        slices=5,
        segment_method="label",
    )


@pytest.fixture
def sample1_scaling(sample1_spm) -> float:
    """Scaling (aka pixel_to_nm_scaling) for Sample 1."""
    _, scaling = sample1_spm
    return scaling  # type: ignore[no-any-return]


@pytest.fixture
def sample1_spm_sliced() -> npt.NDArray[np.float64]:
    """Expected sliced array for sample1."""
    with np.load(RESOURCES_SLICER / "sample1_spm_sliced.npz") as data:
        return data["arr_0"]


@pytest.fixture
def sample1_spm_sliced_mask() -> npt.NDArray[np.float64]:
    """Expected sliced array after masking for sample1."""
    with np.load(RESOURCES_SLICER / "sample1_spm_sliced_mask.npz") as data:
        return data["arr_0"]


@pytest.fixture
def sample1_spm_sliced_segment() -> npt.NDArray[np.float64]:
    """Expected sliced segments array for sample1."""
    with np.load(RESOURCES_SLICER / "sample1_spm_sliced_mask_segment.npz") as data:
        return data["arr_0"]


# Fixtures for testing the instantiation of AFMSlicer with sample2.spm
@pytest.fixture(name="afmslicer_sample2")
def fixture_afmslicer_sample2(sample2_spm) -> AFMSlicer:
    """Fixture of AFMSlicer using sample2.spm."""
    height, pixel_to_nm_scaling = sample2_spm
    return AFMSlicer(
        image=height,
        filename="sample2",
        img_path="tmp",
        pixel_to_nm_scaling=pixel_to_nm_scaling,
        slices=5,
        segment_method="label",
    )


@pytest.fixture
def sample2_scaling(sample2_spm) -> float:
    """Scaling (aka pixel_to_nm_scaling) for Sample 2."""
    _, scaling = sample2_spm
    return scaling  # type: ignore[no-any-return]


@pytest.fixture
def sample2_spm_sliced() -> npt.NDArray[np.float64]:
    """Expected sliced array for sample2."""
    with np.load(RESOURCES_SLICER / "sample2_spm_sliced.npz") as data:
        return data["arr_0"]


@pytest.fixture
def sample2_spm_sliced_mask() -> npt.NDArray[np.float64]:
    """Expected sliced array after masking for sample2."""
    with np.load(RESOURCES_SLICER / "sample2_spm_sliced_mask.npz") as data:
        return data["arr_0"]


@pytest.fixture
def sample2_spm_sliced_segment() -> npt.NDArray[np.float64]:
    """Expected sliced segments array for sample2."""
    with np.load(RESOURCES_SLICER / "sample2_spm_sliced_mask_segment.npz") as data:
        return data["arr_0"]


# Fixtures for test_classes.py
@pytest.fixture
def afmslicer_sample1_sliced() -> npt.NDArray[np.float64]:
    """Sample 1 sliced."""
    with np.load(RESOURCES_SLICER / "afmslicer_sample1_sliced.npz") as data:
        return data["arr_0"]


@pytest.fixture
def afmslicer_sample1_sliced_mask() -> npt.NDArray[np.int32]:
    """Sample 1 sliced."""
    with np.load(RESOURCES_SLICER / "afmslicer_sample1_sliced_mask.npz") as data:
        return data["arr_0"]


@pytest.fixture
def afmslicer_sample1_sliced_segment() -> npt.NDArray[np.float64]:
    """Sample 1 sliced."""
    with np.load(RESOURCES_SLICER / "afmslicer_sample1_sliced_segment.npz") as data:
        return data["arr_0"]


@pytest.fixture
def afmslicer_sample2_sliced() -> npt.NDArray[np.float64]:
    """Sample 2 sliced."""
    with np.load(RESOURCES_SLICER / "afmslicer_sample2_sliced.npz") as data:
        return data["arr_0"]


@pytest.fixture
def afmslicer_sample2_sliced_mask() -> npt.NDArray[np.int32]:
    """Sample 2 sliced."""
    with np.load(RESOURCES_SLICER / "afmslicer_sample2_sliced_mask.npz") as data:
        return data["arr_0"]


@pytest.fixture
def afmslicer_sample2_sliced_segment() -> npt.NDArray[np.float64]:
    """Sample 2 sliced."""
    with np.load(RESOURCES_SLICER / "afmslicer_sample2_sliced_segment.npz") as data:
        return data["arr_0"]

"""Fixtures for testing."""

from __future__ import annotations

import pickle as pkl
from pathlib import Path
from pkgutil import get_data
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import yaml
from skimage.measure import label, regionprops  # pylint: disable=no-name-in-module
from topostats.filters import Filters
from topostats.io import LoadScans

from afmslicer.classes import AFMSlicer

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
RESOURCES_SLICER = RESOURCES / "slicer"
RESOURCES_SPM = RESOURCES / "spm"


RNG = np.random.default_rng(seed=65011934213)

# pylint: disable=too-many-lines


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
def pyramid_sliced_segments_clean_2() -> npt.NDArray:
    """Sliced segments after cleaning for two layer pyramid."""
    with np.load(RESOURCES_SLICER / "pyramid_sliced_segments_clean_2.npz") as data:
        return data["arr_0"]


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
def pyramid_sliced_segments_clean_5() -> npt.NDArray:
    """Sliced segments after cleaning for five layer pyramid."""
    with np.load(RESOURCES_SLICER / "pyramid_sliced_segments_clean_5.npz") as data:
        return data["arr_0"]


@pytest.fixture
def pyramid_height_array_5(
    pyramid_array: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    """Repeated layers (n = 5) of the ``pyramid_array``."""
    array = pyramid_array.copy()
    return np.repeat(array[:, :, np.newaxis], 5, axis=2)


@pytest.fixture
def pyramid_sliced_region_properties_5() -> Any:
    """Region properties for slices of pyramid with five layers."""
    with Path.open(  # pylint: disable=unspecified-encoding
        RESOURCES_SLICER / "pyramid_sliced_region_properties_5.pkl",
        mode="rb",
    ) as f:
        return pkl.load(f)


@pytest.fixture
def pyramid_sliced_clean_region_properties_5() -> Any:
    """Region properties for cleaned slices of pyramid with five layers."""
    with Path.open(  # pylint: disable=unspecified-encoding
        RESOURCES_SLICER / "pyramid_sliced_clean_region_properties_5.pkl",
        mode="rb",
    ) as f:
        return pkl.load(f)


@pytest.fixture
def pyramid_height_array_2(
    pyramid_array: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    """Repeated layers (n = 2) of the ``pyramid_array``."""
    array = pyramid_array.copy()
    return np.repeat(array[:, :, np.newaxis], 2, axis=2)


@pytest.fixture
def pyramid_sliced_region_properties_2() -> Any:
    """Region properties for slices of pyramid with two layers."""
    with Path.open(  # pylint: disable=unspecified-encoding
        RESOURCES_SLICER / "pyramid_sliced_region_properties_2.pkl",
        mode="rb",
    ) as f:
        return pkl.load(f)


@pytest.fixture
def pyramid_sliced_clean_region_properties_2() -> Any:
    """Region properties for cleaned slices of pyramid with two layers."""
    with Path.open(  # pylint: disable=unspecified-encoding
        RESOURCES_SLICER / "pyramid_sliced_clean_region_properties_2.pkl",
        mode="rb",
    ) as f:
        return pkl.load(f)


@pytest.fixture
def pyramid_array_volume() -> npt.NDArray[np.float64]:
    """Array of the volume of the pyramid test object."""
    return np.asarray([165])


@pytest.fixture
def afmslicer_basic(
    pyramid_array: npt.NDArray[np.int32], default_config: dict[str, Any]
) -> AFMSlicer:
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
        config=default_config,
    )


@pytest.fixture
def afmslicer_with_attributes(
    pyramid_array: npt.NDArray[np.int32], default_config: dict[str, Any]
) -> AFMSlicer:
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
        config=default_config,
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
def square_array_sliced_mask() -> npt.NDArray[np.int32]:
    """Simple squareal two-dimensional numpy array sliced 5 times."""
    with np.load(RESOURCES_SLICER / "square_array_sliced_mask.npz") as data:
        return data["arr_0"]


@pytest.fixture
def square_array_sliced_mask_segment() -> npt.NDArray[np.int32]:
    """Simple squareal two-dimensional numpy array sliced 5 times."""
    with np.load(RESOURCES_SLICER / "square_array_sliced_mask_segment.npz") as data:
        return data["arr_0"]


@pytest.fixture
def square_array_volume() -> npt.NDArray[np.float64]:
    """Array of the volume of the square test object."""
    return np.asarray([125])


# Different segments
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


# Sample 1 fixtures...
@pytest.fixture(name="sample1_spm")
def fixture_sample1_spm(default_config: dict[str, Any]) -> tuple[npt.NDArray, float]:
    """Load an image and filter/flatten it ready for analysis."""
    scan_loader = LoadScans(
        [RESOURCES_SPM / "sample1.spm"], channel="Height", config=default_config
    )
    scan_loader.get_data()
    # @ns-rse 2025-11-07 Expect TopoStats to change as TopoStats class will include the configuration and the arguments
    # will be optional.
    config = default_config["filter"].copy()
    config.pop("run")
    _filter = Filters(topostats_object=scan_loader.img_dict["sample1.spm"], **config)
    _filter.filter_image()
    return (_filter.image, _filter.pixel_to_nm_scaling)


@pytest.fixture(name="afmslicer_sample1")
def fixture_afmslicer_sample1(sample1_spm, default_config: dict[str, Any]) -> AFMSlicer:
    """Fixture of AFMSlicer using sample1.spm."""
    height, pixel_to_nm_scaling = sample1_spm
    return AFMSlicer(
        image=height,
        filename="sample1",
        img_path="tmp",
        pixel_to_nm_scaling=pixel_to_nm_scaling,
        slices=5,
        segment_method="label",
        config=default_config,
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


# Sample 2 fixtures...
@pytest.fixture(name="sample2_spm")
def fixture_sample2_spm(default_config: dict[str, Any]) -> tuple[npt.NDArray, float]:
    """Load an image and filter/flatten it ready for analysis."""
    scan_loader = LoadScans(
        [RESOURCES_SPM / "sample2.spm"], channel="Height", config=default_config
    )
    scan_loader.get_data()
    # @ns-rse 2025-11-07 Expect TopoStats to change as TopoStats class will include the configuration and the arguments
    # will be optional.
    config = default_config["filter"]
    config.pop("run")
    _filter = Filters(topostats_object=scan_loader.img_dict["sample2.spm"], **config)
    _filter.filter_image()
    return (_filter.image, _filter.pixel_to_nm_scaling)


@pytest.fixture(name="afmslicer_sample2")
def fixture_afmslicer_sample2(sample2_spm, default_config: dict[str, Any]) -> AFMSlicer:
    """Fixture of AFMSlicer using sample2.spm."""
    height, pixel_to_nm_scaling = sample2_spm
    return AFMSlicer(
        image=height,
        filename="sample2",
        img_path="tmp",
        pixel_to_nm_scaling=pixel_to_nm_scaling,
        slices=5,
        segment_method="label",
        config=default_config,
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
def afmslicer_sample1_segments_clean() -> npt.NDArray:
    """Sliced segments after cleaning for sample 1."""
    with np.load(RESOURCES_SLICER / "afmslicer_sample1_segments_clean.npz") as data:
        return data["arr_0"]


@pytest.fixture
def afmslicer_sample1_region_properties() -> Any:
    """Region properties for slices of sample1."""
    with Path.open(  # pylint: disable=unspecified-encoding
        RESOURCES_SLICER / "afmslicer_sample1_region_properties.pkl",
        mode="rb",
    ) as f:
        return pkl.load(f)


@pytest.fixture
def afmslicer_sample1_clean_region_properties() -> Any:
    """Region properties for cleaned slices of sample1."""
    with Path.open(  # pylint: disable=unspecified-encoding
        RESOURCES_SLICER / "afmslicer_sample1_clean_region_properties.pkl",
        mode="rb",
    ) as f:
        return pkl.load(f)


@pytest.fixture
def sample1_volumes() -> npt.NDArray[np.float64]:
    """Array of the volume of objects in from sample1."""
    return np.asarray(
        [
            15645682811.73706,
            73313713.07373047,
            465214252.4719238,
            533878803.2531738,
            142395496.3684082,
            232100486.7553711,
            26643276.21459961,
            202953815.46020508,
            28371810.913085938,
            56922435.76049805,
            505328178.4057617,
            208199024.20043945,
            38385391.23535156,
            286102294.921875,
            759005546.5698242,
            302374362.94555664,
            28252601.623535156,
            41127204.89501953,
            176846981.04858398,
            11265277.862548828,
            1057267189.0258789,
            59843063.35449219,
            25272369.384765625,
            14781951.904296875,
            40888786.31591797,
            122725963.5925293,
            43928623.19946289,
            20325183.868408203,
            18537044.525146484,
            32484531.40258789,
            5841255.187988281,
            16689300.537109375,
            90539455.41381836,
            1013278.9611816406,
            6794929.504394531,
            46491622.92480469,
            78439712.52441406,
            129878520.96557617,
            14662742.614746094,
            81241130.82885742,
            476837.158203125,
            8404254.913330078,
            17881393.432617188,
            7152557.373046875,
            54001808.166503906,
            30815601.348876953,
            596046.4477539062,
            894069.6716308594,
            15556812.286376953,
            19848346.710205078,
            2622604.3701171875,
            4887580.871582031,
            14364719.39086914,
            9000301.361083984,
            3874301.9104003906,
            13887882.232666016,
            6437301.6357421875,
            4529953.0029296875,
            2384185.791015625,
            6258487.701416016,
            6198883.056640625,
            9238719.940185547,
            2026557.9223632812,
        ],
        dtype=np.float64,
    )


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


@pytest.fixture
def afmslicer_sample2_segments_clean() -> npt.NDArray:
    """Sliced segments after cleaning for sample 2."""
    with np.load(RESOURCES_SLICER / "afmslicer_sample2_segments_clean.npz") as data:
        return data["arr_0"]


@pytest.fixture
def afmslicer_sample2_region_properties() -> Any:
    """Region properties for slices of sample2."""
    with Path.open(  # pylint: disable=unspecified-encoding
        RESOURCES_SLICER / "afmslicer_sample2_region_properties.pkl",
        mode="rb",
    ) as f:
        return pkl.load(f)


@pytest.fixture
def afmslicer_sample2_clean_region_properties() -> Any:
    """Region properties for cleaned slices of sample2."""
    with Path.open(  # pylint: disable=unspecified-encoding
        RESOURCES_SLICER / "afmslicer_sample2_clean_region_properties.pkl",
        mode="rb",
    ) as f:
        return pkl.load(f)


@pytest.fixture
def sample2_volumes() -> npt.NDArray[np.float64]:
    """Array of the volume of objects in from sample2."""
    return np.asarray(
        [
            184957.763671875,
            42.724609375,
            7.080078125,
            1.220703125,
            46643.310546875,
            9181.396484375,
            1.46484375,
            3.41796875,
            2.197265625,
            15.13671875,
            1.220703125,
            0.732421875,
            36.1328125,
            2.685546875,
            2.44140625,
            1.220703125,
            1.708984375,
            8.7890625,
            1.708984375,
            0.9765625,
            2.685546875,
            2.44140625,
            5.859375,
            5.859375,
            0.9765625,
            2.44140625,
            60.05859375,
            2.197265625,
            1.708984375,
            0.732421875,
            114.501953125,
            7.568359375,
            28.076171875,
            8.30078125,
            2.9296875,
            0.732421875,
            1.708984375,
            2.197265625,
            4.39453125,
            3.41796875,
            6.103515625,
            2.197265625,
            0.9765625,
            2.44140625,
            2.44140625,
            87.890625,
            2.9296875,
            60.546875,
            1.46484375,
            2.685546875,
            16.6015625,
            3.41796875,
            0.9765625,
            5.37109375,
            2.685546875,
            1.220703125,
            0.732421875,
            0.48828125,
            1.708984375,
            3.41796875,
            0.732421875,
            1.46484375,
            0.9765625,
            1.708984375,
            0.48828125,
            2.197265625,
            0.732421875,
            0.48828125,
            1.220703125,
            2.197265625,
            1.220703125,
            0.732421875,
            0.9765625,
            0.732421875,
            1.220703125,
            1.708984375,
            0.244140625,
            0.244140625,
            0.244140625,
            0.244140625,
            0.48828125,
            0.9765625,
            0.732421875,
            0.732421875,
        ],
        dtype=np.float64,
    )


@pytest.fixture(name="small_artefacts_array")
def fixture_small_artefacts_array() -> npt.NDArray[np.int32]:
    """Basic array with small artefacts."""
    return np.asarray(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int32,
    )


@pytest.fixture(name="small_artefacts_labelled")
def fixture_small_artefacts_labelled(
    small_artefacts_array: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    """Basic array with small artefacts labelled using ``skimage.label()``."""
    return label(small_artefacts_array)


@pytest.fixture(name="small_artefacts_region_properties")
def fixture_small_artefacts_region_properties(
    small_artefacts_labelled: npt.NDArray[np.int32],
) -> Any:
    """Region properties for ``small_artefacts_array``."""
    return regionprops(small_artefacts_labelled)


@pytest.fixture
def small_artefacts_layered(
    small_artefacts_labelled: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    """Three-dimensional array of labelled layers (both are identical)."""
    return np.stack((small_artefacts_labelled, small_artefacts_labelled), axis=2)


@pytest.fixture
def small_artefacts_layered_region_properties(
    small_artefacts_region_properties: Any,
) -> list[Any]:
    """List of region properties for stacked layers (both are identical)."""
    return [small_artefacts_region_properties, small_artefacts_region_properties]

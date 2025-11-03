"""Test the slice module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from afmslicer import slicer

# pylint: disable=too-many-arguments,too-many-positional-arguments


@pytest.mark.parametrize(
    ("height_fixture", "slices", "shape", "height_stacked_fixture"),
    [
        pytest.param(
            "simple_height_array",
            5,
            (11, 11, 5),
            "simple_height_array_stacked",
            id="simple",
        ),
    ],
)
def test_slicer(
    height_fixture: str,
    slices: int,
    shape: tuple[int],
    height_stacked_fixture: str,
    request,
) -> None:
    """Test for slicer() function."""
    heights = request.getfixturevalue(height_fixture)
    stacked_mask = request.getfixturevalue(height_stacked_fixture)
    sliced_mask = slicer.slicer(heights=heights, slices=slices)
    assert sliced_mask.shape == shape
    np.testing.assert_array_equal(sliced_mask, stacked_mask)


@pytest.mark.parametrize(
    (
        "height_stacked_fixture",
        "slices",
        "min_height",
        "max_height",
        "mask_stacked_fixture",
    ),
    [
        pytest.param(
            "simple_height_array_stacked",
            None,
            None,
            None,
            "simple_height_array_mask_stacked",
            id="simple array, no slices/min/max",
        ),
    ],
)
def test_mask_slices(
    height_stacked_fixture: npt.NDArray[np.int8],
    slices: int,
    min_height: float,
    max_height: float,
    mask_stacked_fixture: npt.NDArray[np.int8],
    request,
) -> None:
    """Test for mask_slices()."""
    stacked_array = request.getfixturevalue(height_stacked_fixture)
    sliced_mask = request.getfixturevalue(mask_stacked_fixture)
    masked_slices = slicer.mask_slices(
        stacked_array=stacked_array,
        slices=slices,
        min_height=min_height,
        max_height=max_height,
    )
    np.testing.assert_array_equal(masked_slices, sliced_mask)


@pytest.mark.parametrize(
    (
        "fixture",
        "filename",
        "img_path",
        "slices",
        "min_height",
        "max_height",
        "layers",
        "sliced_array_fixture",
        "sliced_mask_fixture",
        "pixel_to_nm_scaling",
    ),
    [
        pytest.param(
            "afmslicer_basic",
            "simple_afmslice",
            "tmp",
            5,
            0,
            5,
            np.asarray([0.0, 1.25, 2.5, 3.75, 5.0]),
            "layered_height_array",
            "simple_height_array_mask_stacked",
            1.0,
            id="basic",
        ),
        pytest.param(
            "afmslicer_with_attributes",
            "simple_afmslice_with_attr",
            "tmp",
            2,
            1.0,
            4.0,
            np.asarray([1.0, 4.0]),
            "layered_height_array",
            "simple_height_array_mask_stacked_thin",
            0.5,
            id="basic with min_height=1, max_height=4, layers=2",
        ),
    ],
)
def test_AFMSlicer(
    fixture: str,
    filename: str,
    img_path: Path,
    slices: int,
    min_height: int | float,
    max_height: int | float,
    layers: npt.NDArray[np.float64],
    sliced_array_fixture: str,
    sliced_mask_fixture: str,
    pixel_to_nm_scaling: float,
    request,
) -> None:
    """Test for creating ``AFMSlicer`` object."""
    sliced_array = request.getfixturevalue(sliced_array_fixture)
    sliced_mask = request.getfixturevalue(sliced_mask_fixture)
    afmslicer_object = request.getfixturevalue(fixture)
    assert afmslicer_object.filename == filename
    assert afmslicer_object.img_path == img_path
    assert afmslicer_object.slices == slices
    assert afmslicer_object.min_height == min_height
    assert afmslicer_object.max_height == max_height
    np.testing.assert_array_equal(afmslicer_object.layers, layers)
    np.testing.assert_array_equal(afmslicer_object.sliced_array, sliced_array)
    np.testing.assert_array_equal(afmslicer_object.sliced_mask, sliced_mask)
    assert afmslicer_object.pixel_to_nm_scaling == pixel_to_nm_scaling
    assert sliced_array.shape == (11, 11, 5)
    assert sliced_mask.shape == (11, 11, 5)


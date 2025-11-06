"""Tests of the classes module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments,protected-access


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
        "sliced_segments_fixture",
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
            "layered_height_array_5",
            "simple_height_array_mask_stacked_5",
            "sliced_segment_label_5",
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
            "layered_height_array_2",
            "simple_height_array_mask_stacked_2",
            "sliced_segment_label_2",
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
    sliced_segments_fixture: str,
    pixel_to_nm_scaling: float,
    request,
) -> None:
    """Test for creating ``AFMSlicer`` object."""
    sliced_array = request.getfixturevalue(sliced_array_fixture)
    sliced_mask = request.getfixturevalue(sliced_mask_fixture)
    sliced_segments = request.getfixturevalue(sliced_segments_fixture)
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
    assert afmslicer_object.sliced_array.shape == sliced_array.shape
    assert afmslicer_object.sliced_mask.shape == sliced_mask.shape
    np.testing.assert_array_equal(afmslicer_object.sliced_segments, sliced_segments)

"""Test the slice module."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from afmslicer import slicer

# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments,protected-access


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
            "simple_height_array_mask_stacked_5",
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
    ("array", "expected"),
    [
        pytest.param(
            np.array(
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
            ),
            np.array(
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
            ),
            id="unconnected",
        ),
    ],
)
def test_label(array: npt.NDArray, expected: npt.NDArray) -> None:
    """Test for slicer._label()."""
    np.testing.assert_array_equal(slicer._label(array), expected)


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        pytest.param(
            "basic_three_segments",
            np.array(
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
            ),
            id="unconnected",
        ),
    ],
)
def test_watershed(fixture: str, expected: npt.NDArray, request) -> None:
    """Test for slicer._watershed()."""
    array = request.getfixturevalue(fixture)
    np.testing.assert_array_equal(slicer._watershed(array), expected)


@pytest.mark.parametrize(
    ("sliced_mask_fixture", "method", "sliced_segments_fixture"),
    [
        pytest.param(
            "simple_height_array_mask_stacked_5",
            "label",
            "simple_height_array_mask_stacked_5",
            id="simple height array (5 layers)",
        ),
        pytest.param(
            "simple_height_array_mask_stacked_2",
            "label",
            "simple_height_array_mask_stacked_2",
            id="simple height array (2 layers)",
        ),
        pytest.param(
            "three_layer_three_segments",
            "label",
            "three_layer_three_segments_label",
            id="simple three layers with three segments using label",
        ),
        pytest.param(
            "three_layer_three_segments",
            "watershed",
            "three_layer_three_segments_watershed",
            id="simple three layers with three segments using watershed",
        ),
    ],
)
def test_slices(
    sliced_mask_fixture: npt.NDArray[np.bool],
    method: str,
    sliced_segments_fixture: str,
    request,
) -> None:
    """Test slicer.segment_slices()."""
    sliced_mask = request.getfixturevalue(sliced_mask_fixture)
    sliced_segments = request.getfixturevalue(sliced_segments_fixture)
    np.testing.assert_array_equal(
        slicer.segment_slices(sliced_mask, method), sliced_segments
    )

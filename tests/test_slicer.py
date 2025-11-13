"""Test the slice module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from afmslicer import slicer

# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments,protected-access

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
RESOURCES_SLICER = RESOURCES / "slicer"


@pytest.mark.parametrize(
    ("height_fixture", "slices", "shape"),
    [
        pytest.param(
            "simple_height_array",
            5,
            (11, 11, 5),
            id="simple",
        ),
        pytest.param(
            "sample1_spm",
            5,
            (512, 512, 5),
            id="simple1",
        ),
        pytest.param(
            "sample2_spm",
            5,
            (640, 640, 5),
            id="simple2",
        ),
    ],
)
def test_slicer(
    height_fixture: str,
    slices: int,
    shape: tuple[int],
    request,
    snapshot,
) -> None:
    """Test for slicer() function."""
    if height_fixture == "simple_height_array":
        heights = request.getfixturevalue(height_fixture)
    else:
        heights, _ = request.getfixturevalue(height_fixture)
    sliced = slicer.slicer(heights=heights, slices=slices)
    assert sliced.shape == shape
    # np.save(RESOURCES_SLICER / f"{sliced_fixture}_sliced.npy", sliced)
    # ns-rse: syrupy doesn't yet support numpy arrays so we convert to string
    #         https://github.com/syrupy-project/syrupy/issues/887
    assert np.array2string(sliced) == snapshot


@pytest.mark.parametrize(
    (
        "sliced_fixture",
        "slices",
        "min_height",
        "max_height",
    ),
    [
        pytest.param(
            "simple_height_array_sliced",
            None,
            None,
            None,
            id="simple array, no slices/min/max",
        ),
        pytest.param(
            "sample1_spm_sliced",
            None,
            None,
            None,
            id="sample1, no slices/min/max",
        ),
        pytest.param(
            "sample2_spm_sliced",
            None,
            None,
            None,
            id="sample2, no slices/min/max",
        ),
    ],
)
def test_mask_slices(
    sliced_fixture: npt.NDArray[np.int8],
    slices: int,
    min_height: float,
    max_height: float,
    request,
    snapshot,
) -> None:
    """Test for mask_slices()."""
    sliced_array = request.getfixturevalue(sliced_fixture)
    masked_slices = slicer.mask_slices(
        stacked_array=sliced_array,
        slices=slices,
        min_height=min_height,
        max_height=max_height,
    )
    # np.save(RESOURCES_SLICER / f"{sliced_fixture}_mask.npy", masked_slices)
    # ns-rse: syrupy doesn't yet support numpy arrays so we convert to string
    #         https://github.com/syrupy-project/syrupy/issues/887
    assert np.array2string(masked_slices) == snapshot


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
    ("sliced_mask_fixture", "method"),
    [
        pytest.param(
            "simple_height_array_sliced_mask",
            "label",
            id="simple height array (5 layers)",
        ),
        pytest.param(
            "simple_height_array_mask_stacked_2",
            "label",
            id="simple height array (2 layers)",
        ),
        pytest.param(
            "three_layer_three_segments",
            "label",
            id="simple three layers with three segments using label",
        ),
        pytest.param(
            "three_layer_three_segments",
            "watershed",
            id="simple three layers with three segments using watershed",
        ),
        pytest.param(
            "sample1_spm_sliced_mask", "label", id="sample1 segment with label"
        ),
        pytest.param(
            "sample2_spm_sliced_mask", "label", id="sample2 segment with label"
        ),
    ],
)
def test_segment_slices(
    sliced_mask_fixture: npt.NDArray[np.bool],
    method: str,
    request,
    snapshot,
) -> None:
    """Test slicer.segment_slices()."""
    sliced_mask = request.getfixturevalue(sliced_mask_fixture)
    sliced_mask_segment = slicer.segment_slices(sliced_mask, method)
    # np.save(
    #     RESOURCES_SLICER / f"{sliced_mask_fixture}_segment.npy", sliced_mask_segment
    # )
    # ns-rse: syrupy doesn't yet support numpy arrays so we convert to string
    #         https://github.com/syrupy-project/syrupy/issues/887
    assert np.array2string(sliced_mask_segment) == snapshot

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

ABSOLUTE_TOLERANCE = 1e-5
RELATIVE_TOLERANCE = 1e-5


@pytest.mark.parametrize(
    ("height_fixture", "slices", "shape"),
    [
        pytest.param(
            "pyramid_array",
            5,
            (11, 11, 5),
            id="pyramid",
        ),
        pytest.param(
            "square_array",
            5,
            (7, 7, 5),
            id="square",
        ),
        pytest.param(
            "sample1_spm",
            5,
            (512, 512, 5),
            id="sample1",
        ),
        pytest.param(
            "sample2_spm",
            5,
            (640, 640, 5),
            id="sample2",
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
    if height_fixture in ["pyramid_array", "square_array"]:
        heights = request.getfixturevalue(height_fixture)
    else:
        heights, _ = request.getfixturevalue(height_fixture)
    sliced = slicer.slicer(heights=heights, slices=slices)
    assert sliced.shape == shape
    np.savez_compressed(RESOURCES_SLICER / f"{height_fixture}_sliced.npz", sliced)
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
            "pyramid_array_sliced",
            None,
            None,
            None,
            id="pyramid array, no slices/min/max",
        ),
        pytest.param(
            "square_array_sliced",
            None,
            None,
            None,
            id="square array, no slices/min/max",
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
    sliced_array: npt.NDArray = request.getfixturevalue(sliced_fixture)
    masked_slices = slicer.mask_slices(
        stacked_array=sliced_array,
        slices=slices,
        min_height=min_height,
        max_height=max_height,
    )
    np.savez_compressed(RESOURCES_SLICER / f"{sliced_fixture}_mask.npz", masked_slices)
    # ns-rse: syrupy doesn't yet support numpy arrays so we convert to string
    #         https://github.com/syrupy-project/syrupy/issues/887
    assert np.array2string(masked_slices) == snapshot


@pytest.mark.parametrize(
    ("array_fixture", "expected"),
    [
        pytest.param(
            "basic_three_segments",
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
def test_label(array_fixture: str, expected: npt.NDArray[np.int32], request) -> None:
    """Test for slicer._label()."""
    array: npt.NDArray[np.int32] = request.getfixturevalue(array_fixture)
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
def test_watershed(fixture: str, expected: npt.NDArray[np.int32], request) -> None:
    """Test for slicer._watershed()."""
    array: npt.NDArray[np.int32] = request.getfixturevalue(fixture)
    np.testing.assert_array_equal(slicer._watershed(array), expected)


@pytest.mark.parametrize(
    ("sliced_mask_fixture", "method"),
    [
        pytest.param(
            "pyramid_array_sliced_mask",
            "label",
            id="pyramid height array (5 layers)",
        ),
        pytest.param(
            "square_array_sliced_mask",
            "label",
            id="square height array (5 layers)",
        ),
        pytest.param(
            "pyramid_array_mask_stacked_2",
            "label",
            id="pyramid height array (2 layers)",
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
    np.savez_compressed(
        RESOURCES_SLICER / f"{sliced_mask_fixture}_segment.npz", sliced_mask_segment
    )
    # ns-rse: syrupy doesn't yet support numpy arrays so we convert to string
    #         https://github.com/syrupy-project/syrupy/issues/887
    assert np.array2string(sliced_mask_segment) == snapshot


# ns-rse 2025-11-13 : Currently feret calculations and tests are disabled as the memory requirements go through the roof
# which suggests that I've got incorrectly labelled images (my suspicion is the background is not 0 in sample1 and
# sample2 labelled images but I need to theck)
@pytest.mark.parametrize(
    (
        "sliced_labels_fixture",
        "scaling_fixture",
        "objects",
    ),
    [
        pytest.param(
            "pyramid_array_sliced_mask_segment",
            1,
            1,
            id="pyramid height array (5 layers)",
            # marks=pytest.mark.skip(reason="development"),
        ),
        pytest.param(
            "square_array_sliced_mask_segment",
            1,
            1,
            id="square height array (5 layers)",
            # marks=pytest.mark.skip(reason="development"),
        ),
        pytest.param(
            "sample1_spm_sliced_segment",
            "sample1_scaling",
            63,
            id="sample1",
            # marks=pytest.mark.skip(reason="development"),
        ),
        pytest.param(
            "sample2_spm_sliced_segment",
            "sample2_scaling",
            84,
            id="sample2",
            marks=pytest.mark.skip(reason="development"),
        ),
    ],
)
def test_calculate_regionprops(
    sliced_labels_fixture: str,
    scaling_fixture: int | str,
    objects: int,
    request,
    snapshot,
) -> None:
    """Test for slicer.calculate_regionprops()."""
    labelled_arrays = request.getfixturevalue(sliced_labels_fixture)
    spacing = (
        request.getfixturevalue(scaling_fixture)
        if isinstance(scaling_fixture, str)
        else scaling_fixture
    )
    region_properties = slicer.calculate_region_properties(
        labelled_arrays, spacing=spacing
    )
    # Extract area(i.e. volume) and centroid for checking
    area = [props.area for props in region_properties]
    centroid = [props.centroid for props in region_properties]
    # feret_diameter_max = [region.feret_diameter_max for region in region_properties]
    assert len(region_properties) == objects
    assert area == snapshot
    assert centroid == snapshot
    # assert feret_diameter_max == snapshot


@pytest.mark.parametrize(
    (
        "sliced_labels_fixture",
        "scaling_fixture",
        "layers",
    ),
    [
        pytest.param(
            "pyramid_array_sliced_mask_segment",
            1,
            5,
            id="pyramid array",
        ),
        pytest.param(
            "square_array_sliced_mask_segment",
            1,
            5,
            id="square array",
        ),
        pytest.param(
            "sample1_spm_sliced_segment",
            "sample1_scaling",
            5,
            id="sample1",
        ),
        pytest.param(
            "sample2_spm_sliced_segment",
            "sample2_scaling",
            5,
            id="sample2",
        ),
    ],
)
def test_region_properties_by_slices(
    sliced_labels_fixture: str,
    scaling_fixture: int | str,
    layers: int,
    request,
    snapshot,
) -> None:
    """Test for region_properties_by_slices."""
    labelled_array = request.getfixturevalue(sliced_labels_fixture)
    spacing = (
        request.getfixturevalue(scaling_fixture)
        if isinstance(scaling_fixture, str)
        else scaling_fixture
    )
    sliced_region_properties = slicer.region_properties_by_slices(
        labelled_array, spacing
    )
    assert len(sliced_region_properties) == layers
    assert len(sliced_region_properties) == labelled_array.shape[2]
    # We extract area/centroid/feret_diameter_max across all layers to a single list for comparison
    area = [props.area for layer in sliced_region_properties for props in layer]
    centroid = [props.centroid for layer in sliced_region_properties for props in layer]
    feret_diameter_max = [
        props.feret_diameter_max
        for layer in sliced_region_properties
        for props in layer
    ]
    assert area == snapshot
    assert centroid == snapshot
    assert feret_diameter_max == snapshot

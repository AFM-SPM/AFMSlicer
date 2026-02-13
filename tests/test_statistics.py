"""Tests of the statistics module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from scipy.stats import norm

from afmslicer import slicer, statistics

# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments,protected-access

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
RESOURCES_SLICER = RESOURCES / "slicer"

ABSOLUTE_TOLERANCE = 1e-5
RELATIVE_TOLERANCE = 1e-5


@pytest.mark.parametrize(
    (
        "sliced_labels_fixture",
        "scaling_fixture",
        "layers",
        "objects_per_layer",
    ),
    [
        pytest.param(
            "pyramid_array_sliced_mask_segment",
            1,
            5,
            np.asarray([1, 1, 1, 1, 1]),
            id="pyramid array",
        ),
        pytest.param(
            "square_array_sliced_mask_segment",
            1,
            5,
            np.asarray([1, 1, 1, 1, 1]),
            id="square array",
            # marks=pytest.mark.skip(reason="testing"),
        ),
        pytest.param(
            "sample1_spm_sliced_segment",
            "sample1_scaling",
            5,
            np.asarray([1, 43, 31, 63, 1]),
            id="sample1",
        ),
        pytest.param(
            "sample2_spm_sliced_segment",
            "sample2_scaling",
            5,
            np.asarray([1, 76, 84, 56, 1]),
            id="sample2",
        ),
    ],
)
def test_count_pores(
    sliced_labels_fixture: str,
    scaling_fixture: int | str,
    layers: int,
    objects_per_layer: list[int],
    request,
) -> None:
    """Test counting the number of pores on each layer."""
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
    np.testing.assert_array_equal(
        statistics.count_pores(sliced_region_properties=sliced_region_properties),
        objects_per_layer,
    )


@pytest.mark.parametrize(
    (
        "sliced_labels_fixture",
        "scaling_fixture",
        "area_per_layer",
    ),
    [
        pytest.param(
            "pyramid_array_sliced_mask_segment",
            1,
            [[81], [49], [25], [9], [1]],
            id="pyramid array",
        ),
        pytest.param(
            "square_array_sliced_mask_segment",
            1,
            [[25], [25], [25], [25], [25]],
            id="square array",
            # marks=pytest.mark.skip(reason="testing"),
        ),
        pytest.param(
            "sample1_spm_sliced_segment",
            "sample1_scaling",
            None,
            id="sample1",
            # marks=pytest.mark.skip(reason="testing"),
        ),
        pytest.param(
            "sample2_spm_sliced_segment",
            "sample2_scaling",
            None,
            id="sample2",
            # marks=pytest.mark.skip(reason="testing"),
        ),
    ],
)
def test_area_pores(
    sliced_labels_fixture: str,
    scaling_fixture: int | str,
    area_per_layer: list[list[float]] | None,
    request,
    snapshot,
) -> None:
    """Test extracting the area of pores on each layer."""
    labelled_array = request.getfixturevalue(sliced_labels_fixture)
    spacing = (
        request.getfixturevalue(scaling_fixture)
        if isinstance(scaling_fixture, str)
        else scaling_fixture
    )
    sliced_region_properties = slicer.region_properties_by_slices(
        labelled_array, spacing
    )
    if area_per_layer is not None:
        assert (
            statistics.area_pores(sliced_region_properties=sliced_region_properties)
            == area_per_layer
        )
    else:
        assert (
            statistics.area_pores(sliced_region_properties=sliced_region_properties)
            == snapshot
        )


@pytest.mark.parametrize(
    (
        "sliced_labels_fixture",
        "scaling_fixture",
        "min_size",
        "expected_area_per_layer",
        "total_area",
    ),
    [
        pytest.param(
            "pyramid_array_sliced_mask_segment",
            1,
            None,
            [81, 49, 25, 9, 1],
            165.0,
            id="pyramid array, no min size",
        ),
        pytest.param(
            "pyramid_array_sliced_mask_segment",
            1,
            10.0,
            [81, 49, 25, 0, 0],
            155.0,
            id="pyramid array, min size 10",
        ),
        pytest.param(
            "square_array_sliced_mask_segment",
            1,
            None,
            [25, 25, 25, 25, 25],
            125.0,
            id="square array, no min_size",
        ),
        pytest.param(
            "sample1_spm_sliced_segment",
            "sample1_scaling",
            None,
            None,
            560981750.4882812,
            id="sample1, no min_size",
        ),
        pytest.param(
            "sample1_spm_sliced_segment",
            "sample1_scaling",
            10000,
            None,
            560919189.453125,
            id="sample1, min_size 10000",
            # marks=pytest.mark.skip(reason="testing"),
        ),
        pytest.param(
            "sample2_spm_sliced_segment",
            "sample2_scaling",
            None,
            None,
            386241.796875,
            id="sample2, no min_size",
            # marks=pytest.mark.skip(reason="testing"),
        ),
    ],
)
def test_sum_area_by_layer(
    sliced_labels_fixture: str,
    scaling_fixture: int | str,
    min_size: float | None,
    expected_area_per_layer: list[list[float]] | None,
    total_area: float,
    request,
    snapshot,
) -> None:
    """Test summation of area of pores on each layer."""
    labelled_array = request.getfixturevalue(sliced_labels_fixture)
    spacing = (
        request.getfixturevalue(scaling_fixture)
        if isinstance(scaling_fixture, str)
        else scaling_fixture
    )
    sliced_region_properties = slicer.region_properties_by_slices(
        labelled_array, spacing
    )
    pore_areas_per_layer = statistics.area_pores(
        sliced_region_properties=sliced_region_properties
    )
    area_per_layer = statistics.sum_area_by_layer(
        areas=pore_areas_per_layer,
        min_size=min_size,
    )
    if expected_area_per_layer is not None:
        assert area_per_layer == expected_area_per_layer
    else:
        assert area_per_layer == snapshot
    assert sum(area_per_layer) == total_area


@pytest.mark.parametrize(
    (
        "sliced_labels_fixture",
        "scaling_fixture",
        "objects_per_layer",
    ),
    [
        pytest.param(
            "pyramid_array_sliced_mask_segment",
            1,
            [[(5.0, 5.0)], [(5.0, 5.0)], [(5.0, 5.0)], [(5.0, 5.0)], [(5.0, 5.0)]],
            id="pyramid array",
        ),
        pytest.param(
            "square_array_sliced_mask_segment",
            1,
            [[(3.0, 3.0)], [(3.0, 3.0)], [(3.0, 3.0)], [(3.0, 3.0)], [(3.0, 3.0)]],
            id="square array",
            # marks=pytest.mark.skip(reason="testing"),
        ),
        pytest.param(
            "sample1_spm_sliced_segment",
            "sample1_scaling",
            None,
            id="sample1",
            # marks=pytest.mark.skip(reason="testing"),
        ),
        pytest.param(
            "sample2_spm_sliced_segment",
            "sample2_scaling",
            None,
            id="sample2",
            # marks=pytest.mark.skip(reason="testing"),
        ),
    ],
)
def test_centroid_pores(
    sliced_labels_fixture: str,
    scaling_fixture: int | str,
    objects_per_layer: list[list[tuple[float, float]]] | None,
    request,
    snapshot,
) -> None:
    """Test counting the number of pores on each layer."""
    labelled_array = request.getfixturevalue(sliced_labels_fixture)
    spacing = (
        request.getfixturevalue(scaling_fixture)
        if isinstance(scaling_fixture, str)
        else scaling_fixture
    )
    sliced_region_properties = slicer.region_properties_by_slices(
        labelled_array, spacing
    )
    if objects_per_layer is not None:
        assert (
            statistics.centroid_pores(sliced_region_properties=sliced_region_properties)
            == objects_per_layer
        )
    else:
        assert (
            statistics.centroid_pores(sliced_region_properties=sliced_region_properties)
            == snapshot
        )


@pytest.mark.parametrize(
    (
        "sliced_labels_fixture",
        "scaling_fixture",
        "objects_per_layer",
    ),
    [
        pytest.param(
            "pyramid_array_sliced_mask_segment",
            1,
            [
                [12.041594578792296],
                [9.219544457292887],
                [6.4031242374328485],
                [3.605551275463989],
                [1.0],
            ],
            id="pyramid array",
        ),
        pytest.param(
            "square_array_sliced_mask_segment",
            1,
            [
                [6.4031242374328485],
                [6.4031242374328485],
                [6.4031242374328485],
                [6.4031242374328485],
                [6.4031242374328485],
            ],
            id="square array",
            # marks=pytest.mark.skip(reason="testing"),
        ),
        pytest.param(
            "sample1_spm_sliced_segment",
            "sample1_scaling",
            None,
            id="sample1",
            # marks=pytest.mark.skip(reason="testing"),
        ),
        pytest.param(
            "sample2_spm_sliced_segment",
            "sample2_scaling",
            None,
            id="sample2",
            # marks=pytest.mark.skip(reason="testing"),
        ),
    ],
)
def test_feret_diameter_maximum_pores(
    sliced_labels_fixture: str,
    scaling_fixture: int | str,
    objects_per_layer: list[list[float]] | None,
    request,
    snapshot,
) -> None:
    """Test calculating the maximum feret diameter of the pores on each layer."""
    labelled_array = request.getfixturevalue(sliced_labels_fixture)
    spacing = (
        request.getfixturevalue(scaling_fixture)
        if isinstance(scaling_fixture, str)
        else scaling_fixture
    )
    sliced_region_properties = slicer.region_properties_by_slices(
        labelled_array, spacing
    )
    if objects_per_layer is not None:
        assert (
            statistics.feret_diameter_maximum_pores(
                sliced_region_properties=sliced_region_properties
            )
            == objects_per_layer
        )
    else:
        assert (
            statistics.feret_diameter_maximum_pores(
                sliced_region_properties=sliced_region_properties
            )
            == snapshot
        )


@pytest.mark.parametrize(
    ("pdf", "expected_fwhm"),
    [
        pytest.param(
            np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]),
            [2, 8],
            id="Array of 11",
        ),
        pytest.param(
            1000 * norm.pdf(np.linspace(0, 100, 100), loc=47, scale=11),
            [34, 59],
            id="Array of 100, mean 47 (11)",
        ),
        pytest.param(
            1000 * norm.pdf(np.linspace(0, 100, 100), loc=50, scale=20),
            [27, 72],
            id="Array of 100 mean 50 (20)",
        ),
        pytest.param(
            1000 * norm.pdf(np.linspace(0, 100, 255), loc=50, scale=20),
            [69, 185],
            id="Array of 255 mean 50 (20)",
        ),
    ],
)
def test_full_width_half_max(pdf: npt.NDArray, expected_fwhm: dict[str, int]) -> None:
    """Test for testname()."""
    fwhm = statistics.full_width_half_max(pdf=pdf)
    assert fwhm == expected_fwhm


# @pytest.mark.parametrize(
#     ("fixture_image1", "fixture_image2", "expected_shape"),
#     [
#         pytest.param(
#             "pyramid_array_volume",
#             "square_array_volume",
#             (2,),
#             id="square and pyramid",
#         ),
#         pytest.param(
#             "sample1_volumes",
#             "sample2_volumes",
#             (147,),
#             id="sample1 and sample2",
#         ),
#     ],
# )
# def test_aggregate_arrays(
#     fixture_image1: str, fixture_image2: str, expected_shape: int, request
# ) -> None:
#     volume_image1 = request.getfixturevalue(fixture_image1)
#     volume_image2 = request.getfixturevalue(fixture_image2)
#     aggregated_array = statistics.aggregate_arrays(
#         {"image1": volume_image1, "image2": volume_image2}
#     )
#     assert aggregated_array.shape == expected_shape

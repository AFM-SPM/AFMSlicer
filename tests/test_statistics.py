"""Tests of the statistics module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
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


@pytest.mark.parametrize(
    ("afmslicer_fixture", "feret_maximum", "centroid"),
    [
        pytest.param("afmslicer_basic", False, False, id="basic pyramid, area only"),
        pytest.param(
            "afmslicer_basic", True, True, id="basic pyramid, area, feret & centroid"
        ),
        pytest.param("afmslicer_sample1", False, False, id="sample1"),
        pytest.param("afmslicer_sample2", False, False, id="sample2"),
    ],
)
def test_create_statistics_dictionary(
    afmslicer_fixture: str, feret_maximum: bool, centroid: bool, request, snapshot
) -> None:
    """Test for create_statistics_dictionary()."""
    afmslicer_object = request.getfixturevalue(afmslicer_fixture)
    afmslicer_object.slice_image()
    afmslicer_object._extract_statistics()
    assert (
        statistics.create_statistics_dictionary(
            sliced_region_properties=afmslicer_object.sliced_region_properties,
            feret_maximum=feret_maximum,
            centroid=centroid,
        )
        == snapshot
    )


@pytest.mark.parametrize(
    ("df", "area_thresholds", "area_colors", "area_val", "expected"),
    [
        pytest.param(
            pd.DataFrame(
                {
                    "image": ["a", "a", "a", "a"],
                    "layer": [1, 1, 2, 2],
                    "pore": [0, 1, 0, 1],
                    "area": [0, 2, 4, 6],
                }
            ),
            {
                "low": 1,
                "medium": 3,
                "high": 5,
            },
            ["yellow", "green", "magenta", "blue"],
            "area",
            pd.DataFrame(
                {
                    "image": ["a", "a", "a", "a"],
                    "layer": [1, 1, 2, 2],
                    "pore": [0, 1, 0, 1],
                    "area": [0, 2, 4, 6],
                    "pore_color": ["yellow", "green", "magenta", "blue"],
                },
            ),
            id="simple",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "image": ["a", "a", "a", "a", "a", "a", "a", "a", "a", "a"],
                    "layer": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "pore": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    "area": [10, 250, 368, 2001, 1896, 645, 238, 1, 8, 20],
                },
            ),
            {
                "low": 20,
                "medium": 500,
                "high": 1500,
            },
            ["yellow", "green", "magenta", "blue"],
            "area",
            pd.DataFrame(
                {
                    "image": ["a", "a", "a", "a", "a", "a", "a", "a", "a", "a"],
                    "layer": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "pore": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    "area": [10, 250, 368, 2001, 1896, 645, 238, 1, 8, 20],
                    "pore_color": [
                        "yellow",
                        "green",
                        "green",
                        "blue",
                        "blue",
                        "magenta",
                        "green",
                        "yellow",
                        "yellow",
                        "green",
                    ],
                }
            ),
            id="realistic",
        ),
    ],
)
def test_classify_pore_size(
    df: pd.DataFrame,
    area_thresholds: list[str],
    area_colors: dict[str, int],
    area_val: str,
    expected: pd.DataFrame,
) -> None:
    """Test for testname()."""
    df.set_index(["image", "layer", "pore"], inplace=True)  # noqa: PD002
    df_labelled = statistics.classify_pore_size(
        df=df,
        area_thresholds=area_thresholds,
        area_colors=area_colors,
        area_val=area_val,
    )
    pd.testing.assert_frame_equal(df_labelled, expected)


@pytest.mark.parametrize(
    ("area_colors", "area_thresholds", "error"),
    [
        pytest.param(
            ["yellow", "green", "magenta", "blue", "cyan"],
            {
                "low": 20,
                "medium": 500,
                "high": 1500,
            },
            ValueError,
            id="len(area_colors) == 5",
        ),
        pytest.param(
            ["yellow", "green", "magenta"],
            {
                "low": 20,
                "medium": 500,
                "high": 1500,
            },
            ValueError,
            id="len(area_colors) == 3",
        ),
        pytest.param(
            ["yellow", "green", "magenta", "blue"],
            {
                "low": 20,
                "medium": 500,
                "high": 1500,
                "super high": 300000000,
            },
            ValueError,
            id="len(area_thresholds) == 4",
        ),
        pytest.param(
            ["yellow", "green", "magenta", "blue", "cyan"],
            {
                "low": 20,
                "medium": 500,
            },
            ValueError,
            id="len(area_thresholds) == 2",
        ),
        pytest.param(
            ["yellow", "green", "magenta", "blue"],
            {
                "low": 20,
                "medium": 500,
                "super high": 300000000,
            },
            ValueError,
            id="area_thresholds incorrect key(s)",
        ),
    ],
)
def test_classify_pore_size_value_error(
    area_colors: list[str], area_thresholds: dict[str, int], error
) -> None:
    """Test for classify_pore_size_value_error()."""
    with pytest.raises(error):
        statistics.classify_pore_size(
            df=pd.DataFrame({"area": [1, 2, 3]}),
            area_thresholds=area_thresholds,
            area_colors=area_colors,
            area_val="area",
        )


@pytest.mark.parametrize(
    ("df", "pore_color", "expected"),
    [
        pytest.param(
            pd.DataFrame(
                {
                    "image": ["a", "a"],
                    "layer": [0, 1],
                    "blue": [1, 0],
                    "green": [1, 1],
                    "magenta": [0, 1],
                    "yellow": [1, 1],
                    "total": [3, 3],
                }
            ),
            "yellow",
            pd.DataFrame(
                {
                    "image": ["a", "a"],
                    "layer": [0, 1],
                    "blue": [1, 0],
                    "green": [1, 1],
                    "magenta": [0, 1],
                    "yellow": [1, 1],
                    "total": [3, 3],
                }
            ),
            id="all present, check yellow",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "image": ["a", "a"],
                    "layer": [0, 1],
                    "blue": [1, 0],
                    "green": [1, 1],
                    "magenta": [0, 1],
                    "yellow": [1, 1],
                    "total": [3, 3],
                }
            ),
            "magenta",
            pd.DataFrame(
                {
                    "image": ["a", "a"],
                    "layer": [0, 1],
                    "blue": [1, 0],
                    "green": [1, 1],
                    "magenta": [0, 1],
                    "yellow": [1, 1],
                    "total": [3, 3],
                }
            ),
            id="all present, check magenta",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "image": ["a", "a"],
                    "layer": [0, 1],
                    "green": [1, 1],
                    "magenta": [0, 1],
                    "yellow": [1, 1],
                    "total": [3, 3],
                }
            ),
            "blue",
            pd.DataFrame(
                {
                    "image": ["a", "a"],
                    "layer": [0, 1],
                    "green": [1, 1],
                    "magenta": [0, 1],
                    "yellow": [1, 1],
                    "total": [3, 3],
                    "blue": [0, 0],
                }
            ),
            id="blue missing, check blue",
        ),
    ],
)
def test_add_missing_column(
    df: pd.DataFrame, pore_color: str, expected: pd.DataFrame
) -> None:
    """Test for ``add_missing_columns()``."""
    pd.testing.assert_frame_equal(
        statistics._add_missing_column(df=df, pore_color=pore_color), expected
    )


@pytest.mark.parametrize(
    ("df", "pore_colors", "expected"),
    [
        pytest.param(
            pd.DataFrame(
                {
                    "image": ["a", "a", "a", "a", "a", "a", "a", "a"],
                    "layer": [0, 0, 0, 0, 1, 1, 1, 1],
                    "pore": [0, 1, 2, 3, 0, 1, 2, 3],
                    "area": [1, 400, 1000, 1600, 600, 600, 1100, 1700],
                    "pore_color": [
                        "yellow",
                        "green",
                        "magenta",
                        "blue",
                        "green",
                        "green",
                        "green",
                        "blue",
                    ],
                }
            ),
            ["yellow", "green", "magenta", "blue"],
            pd.DataFrame(
                {
                    "image": ["a", "a"],
                    "layer": [0, 1],
                    "blue": [1, 1],
                    "green": [1, 3],
                    "magenta": [1, 0],
                    "yellow": [1, 0],
                    "total": [4, 4],
                }
            ),
            id="All pores present",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "image": ["a", "a", "a", "a", "a", "a", "a", "a"],
                    "layer": [0, 0, 0, 0, 1, 1, 1, 1],
                    "pore": [0, 1, 2, 3, 0, 1, 2, 3],
                    "area": [1, 12, 1000, 1600, 1650, 1660, 1670, 1700],
                    "pore_color": [
                        "yellow",
                        "yellow",
                        "magenta",
                        "blue",
                        "blue",
                        "blue",
                        "blue",
                        "blue",
                    ],
                }
            ),
            ["yellow", "green", "magenta", "blue"],
            pd.DataFrame(
                {
                    "image": ["a", "a"],
                    "layer": [0, 1],
                    "blue": [1, 4],
                    "magenta": [1, 0],
                    "yellow": [2, 0],
                    "green": [0, 0],
                    "total": [4, 4],
                }
            ),
            id="green pores missing present",
        ),
    ],
)
def test_summarise_pores(
    df: pd.DataFrame, pore_colors: list[str], expected: pd.DataFrame
) -> None:
    """Test ``summarise_pores()`` function."""
    pd.testing.assert_frame_equal(
        statistics.summarise_pores(df=df, pore_colors=pore_colors), expected
    )

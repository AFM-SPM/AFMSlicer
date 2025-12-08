"""Tests of the statistics module."""

from __future__ import annotations

from pathlib import Path

import pytest

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
            [1, 1, 1, 1, 1],
            id="pyramid array",
        ),
        pytest.param(
            "square_array_sliced_mask_segment",
            1,
            5,
            [1, 1, 1, 1, 1],
            id="square array",
            # marks=pytest.mark.skip(reason="testing"),
        ),
        pytest.param(
            "sample1_spm_sliced_segment",
            "sample1_scaling",
            5,
            [1, 43, 31, 63, 1],
            id="sample1",
        ),
        pytest.param(
            "sample2_spm_sliced_segment",
            "sample2_scaling",
            5,
            [1, 76, 84, 56, 1],
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
    assert (
        statistics.count_pores(sliced_region_properties=sliced_region_properties)
        == objects_per_layer
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
    objects_per_layer: list[list[float]] | None,
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
            statistics.area_pores(sliced_region_properties=sliced_region_properties)
            == objects_per_layer
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
    objects_per_layer: list[list[float]] | None,
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
    print(
        f"\n{statistics.feret_diameter_maximum_pores(sliced_region_properties=sliced_region_properties)=}\n"
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

"""Tests of the classes module."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest

from afmslicer.classes import AFMSlicer

# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments,protected-access

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
RESOURCES_SLICER = RESOURCES / "slicer"


# @pytest.mark.mpl_image_compare(baseline_dir="img/classes/")
@pytest.mark.parametrize(
    (
        "fixture",
        "filename",
        "img_path",
        "slices",
        "min_height",
        "max_height",
        "layers",
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
            0.5,
            id="basic with min_height=1, max_height=4, layers=2",
        ),
        pytest.param(
            "afmslicer_sample1",
            "sample1",
            "tmp",
            5,
            -312.40853721614576,
            551.5217325223152,
            np.asarray([-312.408537, -96.42597, 119.556598, 335.539165, 551.521733]),
            39.0625,
            id="sample1 layers=5",
        ),
        pytest.param(
            "afmslicer_sample2",
            "sample2",
            "tmp",
            5,
            -296.85145995382425,
            -152.62116556541318,
            np.asarray(
                [-296.85146, -260.793886, -224.736313, -188.678739, -152.621166]
            ),
            0.625,
            id="sample2 layers=5",
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
    pixel_to_nm_scaling: float,
    request,
) -> plt.Figure:
    """Test for creating ``AFMSlicer`` object."""
    afmslicer_object = request.getfixturevalue(fixture)
    assert isinstance(afmslicer_object, AFMSlicer)
    assert isinstance(afmslicer_object.config, dict)
    assert afmslicer_object.filename == filename
    assert afmslicer_object.img_path == img_path
    assert afmslicer_object.slices == slices
    assert afmslicer_object.min_height == min_height
    assert afmslicer_object.max_height == max_height
    assert afmslicer_object.pixel_to_nm_scaling == pixel_to_nm_scaling
    # Check different arrays
    np.testing.assert_array_almost_equal(afmslicer_object.layers, layers)
    # np.savez_compressed(
    #     RESOURCES_SLICER / f"{sliced_segments_clean_fixture}.npz",
    #     afmslicer_object.sliced_segments_clean,
    # )
    # output = RESOURCES_SLICER / f"{sliced_clean_region_properties_fixture}.pkl"
    # with output.open(mode="wb") as f:
    #     pkl.dump(afmslicer_object.sliced_clean_region_properties, f)


@pytest.mark.mpl_image_compare(baseline_dir="img/classes/")
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
        "sliced_segments_clean_fixture",
        "sliced_region_properties_fixture",
        "sliced_clean_region_properties_fixture",
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
            "pyramid_height_array_5",
            "pyramid_array_sliced_mask_segment",
            "pyramid_segment_label_5",
            "pyramid_sliced_segments_clean_5",
            "pyramid_sliced_region_properties_5",
            "pyramid_sliced_clean_region_properties_5",
            1.0,
            id="basic",
        ),
        pytest.param(
            "afmslicer_sample1",
            "sample1",
            "tmp",
            5,
            -312.40853721614576,
            551.5217325223152,
            np.asarray([-312.408537, -96.42597, 119.556598, 335.539165, 551.521733]),
            "afmslicer_sample1_sliced",
            "afmslicer_sample1_sliced_mask",
            "afmslicer_sample1_sliced_segment",
            "afmslicer_sample1_segments_clean",
            "afmslicer_sample1_region_properties",
            "afmslicer_sample1_clean_region_properties",
            39.0625,
            id="sample1 layers=5",
        ),
        pytest.param(
            "afmslicer_sample2",
            "sample2",
            "tmp",
            5,
            -296.85145995382425,
            -152.62116556541318,
            np.asarray(
                [-296.85146, -260.793886, -224.736313, -188.678739, -152.621166]
            ),
            "afmslicer_sample2_sliced",
            "afmslicer_sample2_sliced_mask",
            "afmslicer_sample2_sliced_segment",
            "afmslicer_sample2_segments_clean",
            "afmslicer_sample2_region_properties",
            "afmslicer_sample2_clean_region_properties",
            0.625,
            id="sample2 layers=5",
        ),
    ],
)
def test_slice_image(
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
    sliced_segments_clean_fixture: str,
    sliced_region_properties_fixture: str,
    sliced_clean_region_properties_fixture: str,
    pixel_to_nm_scaling: float,
    request,
) -> plt.Figure:
    """Test for creating ``AFMSlicer`` object."""
    afmslicer_object = request.getfixturevalue(fixture)
    afmslicer_object.slice_image()
    sliced_array = request.getfixturevalue(sliced_array_fixture)
    sliced_mask = request.getfixturevalue(sliced_mask_fixture)
    sliced_segments = request.getfixturevalue(sliced_segments_fixture)
    sliced_segments_clean = request.getfixturevalue(sliced_segments_clean_fixture)
    sliced_region_properties = request.getfixturevalue(sliced_region_properties_fixture)
    sliced_clean_region_properties = request.getfixturevalue(
        sliced_clean_region_properties_fixture
    )
    assert isinstance(afmslicer_object, AFMSlicer)
    assert isinstance(afmslicer_object.config, dict)
    assert afmslicer_object.filename == filename
    assert afmslicer_object.img_path == img_path
    assert afmslicer_object.slices == slices
    assert afmslicer_object.min_height == min_height
    assert afmslicer_object.max_height == max_height
    # Check different arrays
    np.testing.assert_array_almost_equal(afmslicer_object.layers, layers)
    assert afmslicer_object.sliced_array.shape == sliced_array.shape
    np.testing.assert_array_equal(afmslicer_object.sliced_array, sliced_array)
    assert afmslicer_object.sliced_mask.shape == sliced_mask.shape
    np.testing.assert_array_equal(afmslicer_object.sliced_mask, sliced_mask)
    assert afmslicer_object.pixel_to_nm_scaling == pixel_to_nm_scaling
    assert afmslicer_object.sliced_segments.shape == sliced_segments.shape
    np.testing.assert_array_equal(afmslicer_object.sliced_segments, sliced_segments)
    np.testing.assert_array_equal(
        afmslicer_object.sliced_segments_clean, sliced_segments_clean
    )
    # np.savez_compressed(
    #     RESOURCES_SLICER / f"{sliced_segments_clean_fixture}.npz",
    #     afmslicer_object.sliced_segments_clean,
    # )
    # output = RESOURCES_SLICER / f"{sliced_clean_region_properties_fixture}.pkl"
    # with output.open(mode="wb") as f:
    #     pkl.dump(afmslicer_object.sliced_clean_region_properties, f)

    # Check region properties across slices
    # 2026-01-08 - skip three of the region properties for now as getting from the __eq__ method...
    #   NotImplementedError: `moments_hu` supports spacing = (1, 1) only
    if sliced_region_properties_fixture not in {
        "pyramid_sliced_region_properties_2",
        "afmslicer_sample1_region_properties",
        "afmslicer_sample2_region_properties",
    }:
        assert afmslicer_object.sliced_region_properties == sliced_region_properties
    if sliced_clean_region_properties_fixture not in {
        "pyramid_sliced_clean_region_properties_5",  # Whole image is masked
        "pyramid_sliced_clean_region_properties_2",  # Whole image is masked
        "afmslicer_sample1_clean_region_properties",  # Same moments_hu error as above
        "afmslicer_sample2_clean_region_properties",  # Same moments_hu error as above
    }:
        assert (
            afmslicer_object.sliced_clean_region_properties
            == sliced_clean_region_properties
        )
    # Check plots
    assert isinstance(afmslicer_object.fig_objects_per_layer, tuple)
    assert isinstance(afmslicer_object.fig_objects_per_layer[0], plt.Figure)
    assert isinstance(afmslicer_object.fig_objects_per_layer[1], plt.Axes)
    assert isinstance(afmslicer_object.fig_log_objects_per_layer, tuple)
    assert isinstance(afmslicer_object.fig_log_objects_per_layer[0], plt.Figure)
    assert isinstance(afmslicer_object.fig_log_objects_per_layer[1], plt.Axes)
    return afmslicer_object.fig_objects_per_layer[0]


@pytest.mark.parametrize(
    "fixture",
    [
        pytest.param(
            "afmslicer_with_attributes",
            id="basic with min_height=1, max_height=4, layers=2",
        ),
    ],
)
def test_slice_image_value_error(
    fixture: str,
    request,
) -> None:
    """Test for creating ``AFMSlicer`` object."""
    afmslicer_object = request.getfixturevalue(fixture)
    with pytest.raises(ValueError):
        assert afmslicer_object.slice_image()

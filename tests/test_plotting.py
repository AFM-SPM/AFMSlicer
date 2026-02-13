"""Tests of the plotting module."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest

from afmslicer import plotting, slicer, statistics

# pylint: disable=too-many-positional-arguments


@pytest.mark.parametrize(
    ("array"),
    [
        pytest.param(np.zeros(shape=10), id="1d array"),
        pytest.param(np.zeros(shape=(2, 2, 2)), id="3d array"),
    ],
)
def test_plot_layer_attribute_error_wrong_dimensions(array: npt.NDArray) -> None:
    """Test AttributeError is raised if dimensions of array != 2"""
    with pytest.raises(AttributeError):
        plotting.plot_layer(array)


@pytest.mark.parametrize(
    ("img_name", "layer"),
    [
        pytest.param("test1", None, id="img_name, no layer"),
        pytest.param("test1", None, id="no img_name, layer"),
    ],
)
def test_plot_layer_assert_params(
    img_name: str | None, layer: int | None, tmp_path: Path
) -> plt.Figure:
    """Test for ``plot_layer()`` that if ``outdir`` is supplied both ``img_name`` and ``layer`` are not ``None``."""
    array = np.zeros(shape=(2, 2))
    with pytest.raises(AssertionError):
        plotting.plot_layer(array, img_name, layer, outdir=tmp_path)


@pytest.mark.mpl_image_compare(baseline_dir="img/plot_layer/")
@pytest.mark.parametrize(
    ("array_fixture", "img_name", "layer", "format"),
    [
        pytest.param(
            "pyramid_array_sliced",
            "pyramid_heights",
            3,
            ".png",
            id="pyramid heights, layer 3 as png",
        ),
        pytest.param(
            "pyramid_segment_label_5",
            "pyramid_label",
            3,
            ".png",
            id="pyramid labelled, layer 3 as png",
        ),
        pytest.param(
            "pyramid_array_sliced_mask_segment",
            "pyramid_segment",
            3,
            ".png",
            id="pyramid segmented, layer 3 as png",
        ),
        pytest.param(
            "square_array_sliced",
            "square_heights",
            3,
            ".png",
            id="square heights, layer 3 as png",
        ),
        pytest.param(
            "square_array_sliced_mask",
            "square_label",
            3,
            ".png",
            id="square labelled, layer 3 as png",
        ),
        pytest.param(
            "square_array_sliced_mask_segment",
            "square_segment",
            3,
            ".png",
            id="square segmented, layer 3 as png",
        ),
        pytest.param(
            "basic_three_segments",
            "three_segments",
            0,
            ".png",
            id="three segments, no layer as png",
        ),
        pytest.param(
            "three_layer_three_segments_label",
            "three_segments",
            1,
            ".png",
            id="three segments, layer 1 as png",
        ),
        pytest.param(
            "height_array",
            "height_array",
            1,
            ".png",
            id="multiple object heights, layer 1 as png",
        ),
        pytest.param(
            "sample1_spm_sliced",
            "sample1_spm_sliced",
            3,
            ".png",
            id="sample1 sliced heights, layer 3 as png",
        ),
        pytest.param(
            "sample1_spm_sliced_mask",
            "sample1_spm_sliced_mask",
            3,
            ".png",
            id="sample1 sliced mask, layer 3 as png",
        ),
        pytest.param(
            "sample1_spm_sliced_segment",
            "sample1_spm_sliced_segment",
            3,
            ".png",
            id="sample1 sliced segment, layer 3 as png",
        ),
        pytest.param(
            "sample2_spm_sliced",
            "sample2_spm_sliced",
            3,
            ".png",
            id="sample2 sliced heights, layer 3 as png",
        ),
        pytest.param(
            "sample2_spm_sliced_mask",
            "sample2_spm_sliced_mask",
            3,
            ".png",
            id="sample2 sliced mask, layer 3 as png",
        ),
        pytest.param(
            "sample2_spm_sliced_segment",
            "sample2_spm_sliced_segment",
            3,
            ".png",
            id="sample2 sliced segment, layer 3 as png",
        ),
    ],
)
def test_plot_layer(
    array_fixture: str,
    img_name: str,
    layer: int,
    format: str,  # pylint: disable=redefined-builtin
    tmp_path: Path,
    request,
) -> plt.Figure:
    """Test for ``plot_layer()``."""
    array = request.getfixturevalue(array_fixture)
    array = (
        array[:, :, layer]
        if array_fixture not in ["basic_three_segments", "height_array"]
        else array
    )
    fig, _ = plotting.plot_layer(
        array=array,
        img_name=img_name,
        layer=layer,
        format=format,
        outdir=tmp_path,
    )
    return fig


@pytest.mark.parametrize(
    ("array"),
    [
        pytest.param(np.zeros(shape=10), id="1d array"),
        pytest.param(np.zeros(shape=(2, 2)), id="2d array"),
    ],
)
def test_plot_all_layers_attribute_error_wrong_dimensions(array: npt.NDArray) -> None:
    """Test AttributeError is raised if dimensions of array != 2"""
    with pytest.raises(AttributeError):
        plotting.plot_all_layers(array)


@pytest.mark.mpl_image_compare(baseline_dir="img/plot_all_layers/")
@pytest.mark.parametrize(
    ("array_fixture", "img_name", "layer", "format"),
    [
        pytest.param(
            "pyramid_array_sliced",
            "pyramid_heights",
            3,
            ".png",
            id="pyramid heights, layer 3 as png",
        ),
        pytest.param(
            "pyramid_segment_label_5",
            "pyramid_label",
            3,
            ".png",
            id="pyramid labelled, layer 3 as png",
        ),
        pytest.param(
            "square_array_sliced",
            "square_heights",
            3,
            ".png",
            id="square heights, layer 3 as png",
        ),
        pytest.param(
            "square_array_sliced_mask",
            "square_label",
            3,
            ".png",
            id="square labelled, layer 3 as png",
        ),
        pytest.param(
            "sample1_spm_sliced",
            "sample1_spm_sliced",
            3,
            ".png",
            id="sample1 sliced heights, layer 3 as png",
        ),
        pytest.param(
            "sample1_spm_sliced_mask",
            "sample1_spm_sliced_mask",
            3,
            ".png",
            id="sample1 sliced mask, layer 3 as png",
        ),
        pytest.param(
            "sample1_spm_sliced_segment",
            "sample1_spm_sliced_segment",
            3,
            ".png",
            id="sample1 sliced segment, layer 3 as png",
        ),
        pytest.param(
            "sample2_spm_sliced",
            "sample2_spm_sliced",
            3,
            ".png",
            id="sample2 sliced heights, layer 3 as png",
        ),
        pytest.param(
            "sample2_spm_sliced_mask",
            "sample2_spm_sliced_mask",
            3,
            ".png",
            id="sample2 sliced mask, layer 3 as png",
        ),
        pytest.param(
            "sample2_spm_sliced_segment",
            "sample2_spm_sliced_segment",
            3,
            ".png",
            id="sample2 sliced segment, layer 3 as png",
        ),
    ],
)
def test_plot_all_layers(
    array_fixture: str,
    img_name: str,
    layer: int,
    format: str,  # pylint: disable=redefined-builtin
    tmp_path: Path,
    request,
) -> plt.Figure:
    """Test for ``plot_layer()``."""
    array = request.getfixturevalue(array_fixture)
    plots = plotting.plot_all_layers(
        array=array,
        img_name=img_name,
        outdir=tmp_path,
        format=format,
    )
    assert isinstance(plots, dict)
    assert len(plots) == array.shape[2]
    fig, _ = plots[layer]
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="img/plot_pores_by_layer/")
@pytest.mark.parametrize(
    (
        "sliced_labels_fixture",
        "scaling_fixture",
        "img_name",
        "format",
        "objects_per_layer",
    ),
    [
        pytest.param(
            "pyramid_array_sliced_mask_segment",
            1,
            "pyramid",
            ".png",
            np.asarray([1, 1, 1, 1, 1]),
            id="pyramid heights",
        ),
        pytest.param(
            "sample1_spm_sliced_segment",
            1,
            "sample1",
            ".png",
            np.asarray([1, 43, 31, 63, 1]),
            id="sample1",
        ),
        pytest.param(
            "sample2_spm_sliced_segment",
            1,
            "sample2",
            ".png",
            np.asarray([1, 76, 84, 56, 1]),
            id="sample2",
        ),
        pytest.param(
            np.asarray([0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]),
            1,
            "sample2",
            ".png",
            None,
            id="basic",
        ),
    ],
)
def test_plot_pores_by_layer(
    sliced_labels_fixture: str | list[int],
    scaling_fixture: int | str,
    img_name: str,
    format: str,  # pylint: disable=redefined-builtin
    objects_per_layer: list[int] | None,
    tmp_path: Path,
    request,
) -> plt.Figure:
    """Test for ``plot_pores_by_layer()``."""
    if objects_per_layer is not None:
        labelled_array = request.getfixturevalue(sliced_labels_fixture)
        spacing = (
            request.getfixturevalue(scaling_fixture)
            if isinstance(scaling_fixture, str)
            else scaling_fixture
        )
        sliced_region_properties = slicer.region_properties_by_slices(
            labelled_array, spacing
        )
        pores_per_layer = statistics.count_pores(
            sliced_region_properties=sliced_region_properties
        )
        np.testing.assert_array_equal(pores_per_layer, objects_per_layer)
    else:
        pores_per_layer = sliced_labels_fixture  # type: ignore[assignment]
    fig, _ = plotting.plot_pores_by_layer(
        pores_per_layer=pores_per_layer,
        outdir=tmp_path,
        img_name=img_name,
        format=format,
    )
    return fig


@pytest.mark.mpl_image_compare(baseline_dir="img/plot_area_by_layer/")
@pytest.mark.parametrize(
    (
        "sliced_labels_fixture",
        "scaling_fixture",
        "img_name",
        "format",
    ),
    [
        pytest.param(
            [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            1,
            "basic",
            "png",
            id="basic",
        ),
        pytest.param(
            "pyramid_array_sliced_mask_segment",
            1,
            "pyramid",
            ".png",
            id="pyramid heights",
        ),
        pytest.param(
            "sample1_spm_sliced_segment",
            1,
            "sample1",
            ".png",
            id="sample1",
        ),
        pytest.param(
            "sample2_spm_sliced_segment",
            1,
            "sample2",
            ".png",
            id="sample2",
        ),
    ],
)
def test_plot_area_by_layer(
    sliced_labels_fixture: str | list[float],
    scaling_fixture: int | str,
    img_name: str,
    format: str,  # pylint: disable=redefined-builtin
    tmp_path: Path,
    request,
) -> plt.Figure:
    """Test for ``plot_area_by_layer()``."""
    if isinstance(sliced_labels_fixture, str):
        labelled_array = request.getfixturevalue(sliced_labels_fixture)
        spacing = (
            request.getfixturevalue(scaling_fixture)
            if isinstance(scaling_fixture, str)
            else scaling_fixture
        )
        sliced_region_properties = slicer.region_properties_by_slices(
            labelled_array, spacing
        )
        pore_area_per_layer = statistics.area_pores(
            sliced_region_properties=sliced_region_properties
        )
    else:
        pore_area_per_layer: list[float] = sliced_labels_fixture
    fig, _ = plotting.plot_area_by_layer(
        area_per_layer=pore_area_per_layer,
        outdir=tmp_path,
        img_name=img_name,
        format=format,
    )
    return fig

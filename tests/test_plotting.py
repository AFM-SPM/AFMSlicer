"""Tests of the plotting module."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest

from afmslicer import plotting

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

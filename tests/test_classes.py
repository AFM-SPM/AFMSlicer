"""Tests of the classes module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments,protected-access

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
RESOURCES_SLICER = RESOURCES / "slicer"


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
            "pyramid_height_array_5",
            "pyramid_array_sliced_mask_segment",
            "pyramid_segment_label_5",
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
            "pyramid_height_array_2",
            "pyramid_array_mask_2",
            "pyramid_array_mask_stacked_2",
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
            # "sample1_spm_sliced",
            # "sample1_spm_sliced_mask",
            # "sample1_spm_sliced_segment",
            "afmslicer_sample1_sliced",
            "afmslicer_sample1_sliced_mask",
            "afmslicer_sample1_sliced_segment",
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
            # "sample2_spm_sliced",
            # "sample2_spm_sliced_mask",
            # "sample2_spm_sliced_segment",
            "afmslicer_sample2_sliced",
            "afmslicer_sample2_sliced_mask",
            "afmslicer_sample2_sliced_segment",
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
    print(f"\n{afmslicer_object=}\n")
    assert afmslicer_object.filename == filename
    assert afmslicer_object.img_path == img_path
    assert afmslicer_object.slices == slices
    assert afmslicer_object.min_height == min_height
    assert afmslicer_object.max_height == max_height
    np.testing.assert_array_almost_equal(afmslicer_object.layers, layers)
    # if fixture in ("afmslicer_sample1", "afmslicer_sample2"):
    #     print(f"\nSAVING FILES!!!!\n")
    #     np.savez_compressed(
    #         RESOURCES_SLICER / f"{fixture}_sliced.npz",
    #         afmslicer_object.sliced_array,
    #     )
    #     np.savez_compressed(
    #         RESOURCES_SLICER / f"{fixture}_sliced_mask.npz",
    #         afmslicer_object.sliced_mask,
    #     )
    #     np.savez_compressed(
    #         RESOURCES_SLICER / f"{fixture}_sliced_segment.npz",
    #         afmslicer_object.sliced_segments,
    #     )
    assert afmslicer_object.sliced_array.shape == sliced_array.shape
    np.testing.assert_array_equal(afmslicer_object.sliced_array, sliced_array)
    assert afmslicer_object.sliced_mask.shape == sliced_mask.shape
    np.testing.assert_array_equal(afmslicer_object.sliced_mask, sliced_mask)
    assert afmslicer_object.pixel_to_nm_scaling == pixel_to_nm_scaling
    assert afmslicer_object.sliced_segments.shape == sliced_segments.shape
    np.testing.assert_array_equal(afmslicer_object.sliced_segments, sliced_segments)

"""Tests of the Filter module."""

from __future__ import annotations

import numpy as np
import pytest

# The regressions test framework syrupy (https://syrupy-project.github.io/syrupy/) does not support
# comparison of numpy arrays or pandas dataframes directly (https://github.com/syrupy-project/syrupy/issues/887).
# Instead arrays are converted to strings and compared to snapshot


@pytest.mark.parametrize(
    ("slicing_filter_fixture", "quantile"),
    [
        pytest.param("slicing_filter_random", 0.5, id="random 0.5"),
        pytest.param("slicing_filter_random", 0.25, id="random 0.25"),
        pytest.param("slicing_filter_random", 0.75, id="random 0.75"),
    ],
)
def test_median_flatten(
    slicing_filter_fixture: str, quantile: float, request, snapshot
) -> None:
    """Test the ``Slicing_Filter.median_flatten()`` method inherited from TopoStats."""
    slicing_filter = request.getfixturevalue(slicing_filter_fixture)
    median_flatten = slicing_filter.median_flatten(
        image=slicing_filter.images["pixels"],
        mask=None,
        row_alignment_quantile=quantile,
    )
    assert np.array2string(median_flatten, precision=9) == snapshot


@pytest.mark.parametrize(
    ("slicing_filter_fixture"), [pytest.param("slicing_filter_random", id="random")]
)
def test_tilt_removal(slicing_filter_fixture: str, request, snapshot) -> None:
    """Test the ``Slicing_Filter.tilt_removal()`` method inherited from TopoStats."""
    slicing_filter = request.getfixturevalue(slicing_filter_fixture)
    tilt_removal = slicing_filter.remove_tilt(
        image=slicing_filter.images["pixels"], mask=None
    )
    assert np.array2string(tilt_removal, precision=9) == snapshot


@pytest.mark.parametrize(
    ("slicing_filter_fixture"), [pytest.param("slicing_filter_random", id="random")]
)
def test_nonlinear_polynomial_removal(
    slicing_filter_fixture: str, request, snapshot
) -> None:
    """Test the ``Slicing_Filter.nonlinear_polynomial_removal()`` method inherited from TopoStats."""
    slicing_filter = request.getfixturevalue(slicing_filter_fixture)
    nonlinear_polynomial_removal = slicing_filter.remove_nonlinear_polynomial(
        image=slicing_filter.images["pixels"], mask=None
    )
    assert np.array2string(nonlinear_polynomial_removal, precision=9) == snapshot


@pytest.mark.parametrize(
    ("slicing_filter_fixture"), [pytest.param("slicing_filter_random", id="random")]
)
def test_average_background(slicing_filter_fixture: str, request, snapshot) -> None:
    """Test the ``Slicing_Filter.average_background()`` method inherited from TopoStats."""
    slicing_filter = request.getfixturevalue(slicing_filter_fixture)
    average_background = slicing_filter.average_background(
        image=slicing_filter.images["pixels"], mask=None
    )
    assert np.array2string(average_background, precision=9) == snapshot


@pytest.mark.parametrize(
    ("slicing_filter_fixture", "gaussian_size", "gaussian_mode"),
    [
        pytest.param(
            "slicing_filter_random",
            1.0121397464510862,
            "nearest",
            id="random, default gaussian",
        )
    ],
)
def test_gaussian_filter(
    slicing_filter_fixture: str,
    gaussian_size: float,
    gaussian_mode: str,
    request,
    snapshot,
) -> None:
    """Test the ``Slicing_Filter.gaussian_filter()`` method inherited from TopoStats."""
    slicing_filter = request.getfixturevalue(slicing_filter_fixture)
    slicing_filter.gaussian_size = gaussian_size
    slicing_filter.gaussian_mode = gaussian_mode
    gaussian_filter = slicing_filter.gaussian_filter(
        image=slicing_filter.images["pixels"],
    )
    assert np.array2string(gaussian_filter, precision=9) == snapshot


@pytest.mark.parametrize(
    ("slicing_filter_fixture", "gaussian_size", "gaussian_mode"),
    [
        pytest.param(
            "slicing_filter_random",
            1.0121397464510862,
            "nearest",
            id="random, default gaussian",
        ),
        pytest.param(
            "slicing_filter_random",
            2.0242794929,
            "nearest",
            id="random, doubled gaussian",
        ),
    ],
)
def test_filter_image(
    slicing_filter_fixture: str,
    gaussian_size: float,
    gaussian_mode: str,
    request,
    snapshot,
) -> None:
    """Test the ``Slicing_Filter.filter_image()``."""
    slicing_filter = request.getfixturevalue(slicing_filter_fixture)
    slicing_filter.gaussian_size = gaussian_size
    slicing_filter.gaussian_mode = gaussian_mode
    slicing_filter.filter_image()
    assert (
        np.array2string(slicing_filter.images["gaussian_filtered"], precision=9)
        == snapshot
    )

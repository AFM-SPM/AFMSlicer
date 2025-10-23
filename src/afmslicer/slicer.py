"""Slice two-dimensional arrays to three-dimensional stacked masks."""

# from loguru import logger
from __future__ import annotations

import numpy as np
import numpy.typing as npt

# from pydantic import BaseModel

# class AFMSlice(BaseModel):
#     heights: npt.NDArray[np.float64]
#     min: float
#     max: float
#     layers: npt.NDArray[np.float64]
#     sliced_array: npt.NDArray[np.float64]
#     sliced_mask: npt.NDArray[np.bool]


def mask_slices(
    sliced_array: npt.NDArray,
    slices: int | None = None,
    min_height: np.float64 | None = None,
    max_height: np.float64 | None = None,
) -> npt.NDArray[bool]:
    """
    Convert a three-dimensional sliced array into masks based.

    A three-dimensional array is converted to masked layers where each layer indicates whether a position in the
    two-dimensional cross-section is above the threshold for that layer. Thresholds are determined from the data itself
    if not explicitly provided.

    Parameters
    ----------
    sliced_array : npt.NDArray
        Three-dimensional numpy array.
    slices : int, optional
        Number of slices to mask. Determined directly from data if not provided (i.e. depth of three-dimensional array).
    min_height : np.float64, optional
        Minimum height. Determined directly from data if not provided.
    max_height : np.float64, optional
        Maximum height. Determined directly from data if not provided.

    Returns
    -------
    npt.NDArray[bool]
        Three-dimensional array of masks.
    """
    slices = sliced_array.shape[2] if slices is None else slices
    min_height = np.min(sliced_array) if min_height is None else min_height
    max_height = np.max(sliced_array) if max_height is None else max_height
    sliced_mask = sliced_array.copy()
    for layer, height in enumerate(np.linspace(min_height, max_height, slices)):
        sliced_mask[:, :, layer] = np.where(sliced_array[:, :, layer] > height, 1, 0)
    # We want to capture the maximum...
    sliced_mask[:, :, slices - 1] = np.where(
        sliced_array[:, :, slices - 1] >= max_height, 1, 0
    )
    return sliced_mask


def slicer(heights: npt.NDArray[np.float64], slices: int) -> npt.NDArray[np.float64]:
    """
    Convert a two-dimensional numpy array to a three-dimensional stacked array of masks.

    The minimum and maximum heights are obtained directly from the array itself and a user specified number of slices
    are generated.

    Parameters
    ----------
    heights : npt.NDArray[np.float64]
        Two-dimensional numpy array of heights.
    slices : int
        Number of slices to make.

    Returns
    -------
    npt.NDArray[np.float64]
        Expanded numpy array with the original two-dimensional array copied ``slices`` in the third dimension.

    Examples
    --------
    >>> import numpy as np
    >>> simple = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> slicer(heights=simple, slices=2)
    array([[[1, 1],
        [2, 2],
        [3, 3]],

       [[4, 4],
        [5, 5],
        [6, 6]],

       [[7, 7],
        [8, 8],
        [9, 9]]])
    """
    return np.repeat(
        heights[
            :,
            :,
            np.newaxis,
        ],
        slices,
        axis=2,
    )


def show_layers(array: npt.NDArray) -> None:
    """
    Helper function for debugging which shows individual layers of a three-dimensional numpy array.

    Parameters
    ----------
    array : npt.NDArray
        Three-dimensional numpy array.
    """
    assert len(array.shape) == 3, f"Array is not 3D : {array.shape=}"
    for layer in np.arange(0, array.shape[-1]):
        print(f"\nLayer {layer} \n{array[...,layer]=}\n")  # noqa: T201

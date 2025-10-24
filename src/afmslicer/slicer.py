"""Module for slicing  two-dimensional arrays to three-dimensional arrays of stacked masks."""

# from loguru import logger
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from topostats.classes import TopoStats


@dataclass(
    repr=True,
    eq=True,
    config=ConfigDict(arbitrary_types_allowed=True),
    validate_on_init=True,
)
class AFMSlicer(TopoStats):
    """
    Class for AFMSlicer data and attributes.

    The class inherits ``TopoStats`` class.

    Attributes
    ----------
    image : npt.NDArray[np.float64]
        Two-dimensional array of heights.
    min_height : float, optional
        Minimum height. Determined from the data if not provided.
    max_height : float, optional
        Maximum height. Determined from the data if not provided.
    slices : int, optional
        The number of slices taken through the image between the ``min_height`` and ``max_height``.
    layers : npt.NDArray[np.float64], optional
        The boundaries of heights for sliced layers to be taken through the original image.
    sliced_array : npt.NDArray[np.float64], optional
        Three dimensional array of ``slices`` of the original ``image``.
    sliced_mask : npt.NDArray[bool], optional
        A three dimensional array of ``slices`` where each layer is a mask for the heights given in ``layers``.
    """

    image: npt.NDArray[np.float64]
    min_height: float | None = None
    max_height: float | None = None
    slices: int = 255
    layers: npt.NDArray[np.float64] | None = None
    sliced_array: npt.NDArray[np.float64] | None = None
    sliced_mask: npt.NDArray[np.bool] | None = None

    def __post_init__(self) -> None:
        """
        Set attributes for the ``AFMSlice`` class on instantiation.
        """
        self.min_height = (
            np.min(self.image) if self.min_height is None else self.min_height
        )
        self.max_height = (
            np.max(self.image) if self.max_height is None else self.max_height
        )
        self.layers = np.linspace(self.min_height, self.max_height, self.slices)
        self.sliced_array = slicer(heights=self.image, slices=self.slices)
        self.sliced_mask = mask_slices(
            sliced_array=self.sliced_array,
            slices=self.slices,
            layers=self.layers,
            min_height=self.min_height,
            max_height=self.max_height,
        )


def mask_slices(
    sliced_array: npt.NDArray,
    slices: int | None = 255,
    layers: npt.NDArray | None = None,
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
        Three-dimensional numpy array of image heights, each layer should be a copy of the original.
    slices : int, optional
        Number of slices to mask. Determined directly from data if not provided (i.e. depth of three-dimensional array).
    layers : npt.NDArray, optional
        Array of height thresholds for each slice. Determined directly from data if not provided.
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
    layers = np.linspace(min_height, max_height, slices) if layers is None else layers
    sliced_mask = sliced_array.copy()
    for layer, height in enumerate(layers):
        sliced_mask[:, :, layer] = np.where(sliced_array[:, :, layer] > height, 1, 0)
    # We want to capture the maximum...
    sliced_mask[:, :, slices - 1] = np.where(
        sliced_array[:, :, slices - 1] >= max_height, 1, 0
    )
    return sliced_mask


def slicer(heights: npt.NDArray[np.float64], slices: int) -> npt.NDArray[np.float64]:
    """
    Convert a two-dimensional array to a three-dimensional stacked array with copies of the original in each layer.

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

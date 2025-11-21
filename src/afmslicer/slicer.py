"""Module for slicing  two-dimensional arrays to three-dimensional arrays of stacked masks."""

# from loguru import logger
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from skimage.measure import label, regionprops  # pylint: disable=no-name-in-module
from skimage.segmentation import clear_border, watershed


def mask_slices(
    stacked_array: npt.NDArray,
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
    stacked_array : npt.NDArray
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
    slices = stacked_array.shape[2] if slices is None else slices
    min_height = np.min(stacked_array) if min_height is None else min_height
    max_height = np.max(stacked_array) if max_height is None else max_height
    layers = np.linspace(min_height, max_height, slices) if layers is None else layers
    sliced_mask = stacked_array.copy()
    for layer, height in enumerate(layers):
        sliced_mask[:, :, layer] = np.where(stacked_array[:, :, layer] > height, 1, 0)
    # We want to capture the maximum...
    sliced_mask[:, :, slices - 1] = np.where(
        stacked_array[:, :, slices - 1] >= max_height, 1, 0
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


def segment(
    array: npt.NDArray,
    method: str | None = "label",
    tidy_border: bool = True,
    **kwargs: dict[str, Any] | None,
) -> npt.NDArray:
    """
    Segment an array.

    Parameters
    ----------
    array : npt.NDArray
        Two-dimensional numpy array to segment.
    method : str, optional
        Segmentation method, supports the ``label`` (default) and ``watershed`` methods implemented by Scikit Image.
    tidy_border : bool
        Whether to remove objects that straddle the border of the image.
    **kwargs : dict[str, Any], optional
        Additional arguments to pass for segmentation.

    Returns
    -------
    npt.NDArray
        Labelled array of objects.
    """
    if method is None:
        method = "label"
        # logger.info("No segmentation method specified, defaulting to 'label'.")
    if tidy_border:
        array = clear_border(array)
        # logger.info("")
    segmenter = _get_segments(method)
    return segmenter(array, **kwargs)


def _get_segments(
    method: str = "",
) -> Callable[[npt.NDArray[np.bool]], npt.NDArray[np.int32]]:
    """
    Creator component which determines which threshold method to use.

    Parameters
    ----------
    method : str
        Segmentation method to use, currently supports ``label`` (default) and ``watershed``.

    Returns
    -------
    Callable
        Returns function appropriate for the required segmentation method.

    Raises
    ------
    ValueError
        Unsupported methods result in ``ValueError``.
    """
    if method == "watershed":
        return _watershed
    if method == "label":
        return _label
    raise ValueError()


def _watershed(array: npt.NDArray[np.int32], **kwargs) -> npt.NDArray[np.int32]:
    """
    Segment array using ``watershed`` method.

    Uses the `watershed
    <https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed>`__ method from
    Scikit Image to segment the image.

    Parameters
    ----------
    array : npt.NDArray[np.int32]
        Boolean array.
    **kwargs : dict[str, Any], optional
        Additional arguments to pass to ``watershed``.

    Returns
    -------
    npt.NDArray[np.int32]
        Labelled image.
    """
    return watershed(array, **kwargs).astype(np.int32)


def _label(array: npt.NDArray[np.int32], **kwargs) -> npt.NDArray[np.int32]:
    """
    Segment array using ``label`` method.

    Uses the `label <https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.label>`__
    method from Scikit Image to segment the image.

    Parameters
    ----------
    array : npt.NDArray[np.int32]
        Boolean array.
    **kwargs : dict[str, Any], optional
        Additional arguments to pass to ``label``.

    Returns
    -------
    npt.NDArray[np.int32]
        Labelled image.
    """
    return label(array, **kwargs).astype(np.int32)


def segment_slices(
    array: npt.NDArray[np.bool], method: str | None = "label", tidy_border: bool = False
) -> npt.NDArray[np.bool]:
    """
    Segment individual layers of a three-dimensional numpy array.

    Parameters
    ----------
    array : npt.NDArray[np.bool]
        Three-dimensional boolean array to be segmented.
    method : str
        Sgementation method to use. Currne options are ``label`` (default) and ``watershed``.
    tidy_border : bool
        Whether to tidy the border.

    Returns
    -------
    npt.NDArray[np.bool]
        Three-dimensional array of labelled layers.
    """
    for layer in np.arange(array.shape[2]):
        array[:, :, layer] = segment(array[:, :, layer], method, tidy_border)
    return array


def calculate_region_properties(array: npt.NDArray[np.int32], spacing: float) -> Any:
    """
    Calculate the region properties on a segmented array.

    The arrays can be either individual slices from a three-dimensional image or the full three-dimensional array
    itself. If the later then the resulting attribute ``area`` will be a "volume". By including the ``spacing``
    argument, which should be the ``pixel_to_nm_scaling`` attribute of the ``AFMSlicer`` object the area/volume is
    in the actual units measured rather than pixels.

    Parameters
    ----------
    array : npt.NDArray
        Array of labelled regions.
    spacing : float
        Pixel to nm scaling.

    Returns
    -------
    list[RegionProperties]
        A list of ``RegionProperties``.
    """
    return regionprops(array.astype(np.int8), spacing=spacing)


def region_properties_by_slices(
    array: npt.NDArray[np.int32], spacing: float
) -> list[Any]:
    """
    Calculate region properties for each layer in a three-dimensinoal sliced array.

    Parameters
    ----------
    array : npt.NDArray[np.int32]
        Three-dimensional sliced and labelled array.
    spacing : float
        Pixel to nanometer scaling.

    Returns
    -------
    dict[int, Any]
        Dictionary of ``regionprops`` calculated using skimage.
    """

    slice_properties = []
    for layer in range(array.shape[2]):
        slice_properties.append(
            regionprops(array[:, :, layer].astype(np.int32), spacing=spacing)
        )
    return slice_properties

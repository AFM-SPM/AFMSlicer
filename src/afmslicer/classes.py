"""Classes for the AFMSlicer package."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from topostats.classes import TopoStats

from afmslicer import slicer


@dataclass(
    repr=True,
    eq=True,
    config=ConfigDict(arbitrary_types_allowed=True, validate_assignment=True),
    validate_on_init=True,
)
class AFMSlicer(TopoStats):  # type: ignore[misc]
    """
    Class for AFMSlicer data and attributes.

    The class inherits ``TopoStats`` class.

    Attributes
    ----------
    image : npt.NDArray[np.float64]
        Two-dimensional array of heights.
    layers : npt.NDArray[np.float64], optional
        The boundaries of heights for sliced layers to be taken through the original image.
    sliced_array : npt.NDArray[np.float64], optional
        Three dimensional array of ``slices`` of the original ``image``.
    sliced_mask : npt.NDArray[bool], optional
        A three dimensional array of ``slices`` where each layer is a mask for the heights given in ``layers``.
    segment_method : str, optional
        Method for segmenting individual layers.
    min_height : float, optional
        Minimum height. Determined from the data if not provided.
    max_height : float, optional
        Maximum height. Determined from the data if not provided.
    slices : int, optional
        The number of slices taken through the image between the ``min_height`` and ``max_height``.
    """

    # We may need to set a default_factory see
    # https://stackoverflow.com/questions/70306311/pydantic-initialize-numpy-ndarray
    # https://docs.pydantic.dev/latest/concepts/dataclasses/
    layers: npt.NDArray[np.float64] | None = None
    sliced_array: npt.NDArray[np.float64] | None = None
    sliced_mask: npt.NDArray[np.bool] | None = None
    sliced_segments: npt.NDArray[np.int32] | None = None
    segment_method: str | None = None
    min_height: float | None = None
    max_height: float | None = None
    slices: int | None = None

    def __post_init__(self) -> None:
        """
        Set attributes for the ``AFMSlice`` class on instantiation.
        """
        self.slices = 255 if self.slices is None else self.slices
        self.min_height = (
            np.min(self.image) if self.min_height is None else self.min_height
        )
        self.max_height = (
            np.max(self.image) if self.max_height is None else self.max_height
        )
        self.layers = (
            np.linspace(self.min_height, self.max_height, self.slices)
            if self.layers is None
            else self.layers
        )
        self.sliced_array = slicer.slicer(heights=self.image, slices=self.slices)
        self.sliced_mask = slicer.mask_slices(
            stacked_array=self.sliced_array,
            slices=self.slices,
            layers=self.layers,
            min_height=self.min_height,
            max_height=self.max_height,
        )
        self.sliced_segments = slicer.segment_slices(
            self.sliced_mask, method=self.segment_method
        )

"""Classes for the AFMSlicer package."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from topostats.classes import TopoStats

from afmslicer import plotting, slicer, statistics

# pylint: disable=too-many-instance-attributes


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
    sliced_segments : npt.NDArray[np.int32]
    sliced_region_properties : list[list[Any]]
        A list of ``region_props`` for each object in each layer. There is one list for each layer.
    segment_method : str, optional
        Method for segmenting individual layers.
    min_height : float, optional
        Minimum height. Determined from the data if not provided.
    max_height : float, optional
        Maximum height. Determined from the data if not provided.
    slices : int, optional
        The number of slices taken through the image between the ``min_height`` and ``max_height``.
    fig_objects_per_layer : tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axes objects from plotting the layer vs the number of objects.
    fig_log_objects_per_layer : tuple[plt.Figure, plt.Axes]
        Matplotlib figure and axes objects from plotting the layer vs the log number of objects.
    """

    # We may need to set a default_factory see
    # https://stackoverflow.com/questions/70306311/pydantic-initialize-numpy-ndarray
    # https://docs.pydantic.dev/latest/concepts/dataclasses/
    layers: npt.NDArray[np.float64] | None = None
    sliced_array: npt.NDArray[np.float64] | None = None
    sliced_mask: npt.NDArray[np.bool] | None = None
    sliced_segments: npt.NDArray[np.int32] | None = None
    sliced_region_properties: list[Any] | None = None
    pores_per_layer: list[int] | None = None
    segment_method: str | None = None
    min_height: float | None = None
    max_height: float | None = None
    slices: int | None = None
    fig_objects_per_layer: tuple[plt.Figure, plt.Axes] | None = None
    fig_log_objects_per_layer: tuple[plt.Figure, plt.Axes] | None = None

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
        # Slice the array (i.e. duplicate it `slices` times)
        self.sliced_array = slicer.slicer(heights=self.image, slices=self.slices)
        # Mask each layer
        self.sliced_mask = slicer.mask_slices(
            stacked_array=self.sliced_array,
            slices=self.slices,
            layers=self.layers,
            min_height=self.min_height,
            max_height=self.max_height,
        )
        # Detect segments within each slice
        self.sliced_segments = slicer.segment_slices(
            self.sliced_mask, method=self.segment_method
        )
        # Calculate region properties
        self.sliced_region_properties = slicer.region_properties_by_slices(
            array=self.sliced_segments, spacing=self.pixel_to_nm_scaling
        )
        self.pores_per_layer = statistics.count_pores(
            sliced_region_properties=self.sliced_region_properties
        )
        # Plot all segmented layers
        plotting.plot_all_layers(
            array=self.sliced_segments,
            img_name=self.filename,
            outdir=self.config["output_dir"],
            format=self.config["slicing"]["format"],
        )
        # Plot pores per layer
        self.fig_objects_per_layer = plotting.plot_pores_by_layer(
            pores_per_layer=self.pores_per_layer,
            img_name=self.filename,
            outdir=self.config["output_dir"],
            format=self.config["slicing"]["format"],
            log=False,
        )
        # Plot pores per layer (log scale)
        self.fig_log_objects_per_layer = plotting.plot_pores_by_layer(
            pores_per_layer=self.pores_per_layer,
            img_name=self.filename,
            outdir=self.config["output_dir"],
            format=self.config["slicing"]["format"],
            log=True,
        )
        # Optionally calculate additional statistics
        # Areas
        if self.config["slicing"]["area"]:
            self.sliced_area = statistics.area_pores(
                sliced_region_properties=self.sliced_region_properties
            )
        # Centroid
        if self.config["slicing"]["centroid"]:
            self.sliced_area = statistics.centroid_pores(
                sliced_region_properties=self.sliced_region_properties
            )
        # Feret Maximum
        if self.config["slicing"]["feret_maximum"]:
            self.sliced_area = statistics.feret_diameter_maximum_pores(
                sliced_region_properties=self.sliced_region_properties
            )

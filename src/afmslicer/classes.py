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
        A three dimensional array of ``slices`` where each layer has been segmented.
    sliced_segments_clean : npt.NDArray[np.int32]
        A three dimensional array of ``slices`` where small objects have been masked. The threshold for masking comes
        from the configuration file under ``slicing.minimum_size`` and is ``8000`` nanometres squared by default.
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
    fig_objects_per_layer : tuple[plt.Figure, plt.Axes], optional
        Matplotlib figure and axes objects from plotting the layer vs the number of objects.
    fig_log_objects_per_layer : tuple[plt.Figure, plt.Axes], optional
        Matplotlib figure and axes objects from plotting the layer vs the log number of objects.
    fig_area_per_layer : tuple[plt.Figure, plt.Axes], optional
        Matplotlib figure and axes objects from plotting the layer vs the total area of objects.
    fig_log_area_per_layer : tuple[plt.Figure, plt.Axes], optional
        Matplotlib figure and axes objects from plotting the layer vs the log of the total area of objects.
    area_by_layer : : list[list[float]], optional
        List of the area of each object within a given layer.
    centroid_by_layer: list[list[tuple[float, float]]], optional
        List of the centroid (as a tuple) of each object within a given layer.
    feret_maximum_by_layer: list[list[float]], optional
        List of the maximum feret distance of each object within a given layer.
    """

    # We may need to set a default_factory see
    # https://stackoverflow.com/questions/70306311/pydantic-initialize-numpy-ndarray
    # https://docs.pydantic.dev/latest/concepts/dataclasses/
    layers: npt.NDArray[np.float64] | None = None
    sliced_array: npt.NDArray[np.float64] | None = None
    sliced_mask: npt.NDArray[np.bool] | None = None
    sliced_segments: npt.NDArray[np.int32] | None = None
    sliced_segments_clean: npt.NDArray[np.int32] | None = None
    sliced_region_properties: list[Any] | None = None
    sliced_clean_region_properties: list[Any] | None = None
    pores_per_layer: list[int] | None = None
    segment_method: str | None = None
    min_height: float | None = None
    max_height: float | None = None
    slices: int | None = None
    fig_objects_per_layer: tuple[plt.Figure, plt.Axes] | None = None
    fig_log_objects_per_layer: tuple[plt.Figure, plt.Axes] | None = None
    fig_area_per_layer: tuple[plt.Figure, plt.Axes] | None = None
    fig_log_area_per_layer: tuple[plt.Figure, plt.Axes] | None = None
    area_by_layer: list[list[float]] | None = None
    centroid_by_layer: list[list[tuple[float, float]]] | None = None
    feret_maximum_by_layer: list[list[float]] | None = None

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

    def slice_image(self) -> None:
        """
        Slice the image.
        """
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
        # ns-rse 2025-12-10 : Consider first pass of areas using skimage.measure.moments() to quickly get _just_ areas
        # so that small objects can be excluded, only then calculate regionprops on remaining items. Could possibly be
        # extended to only do regionprops on the layers of interest after the Full-Width Half Max has been determeind.
        # Calculate region properties
        self.sliced_region_properties = slicer.region_properties_by_slices(
            array=self.sliced_segments, spacing=self.pixel_to_nm_scaling
        )
        # Remove small objects
        self.sliced_segments_clean = slicer.mask_small_artefacts_all_layers(
            labelled_array=self.sliced_segments,
            properties=self.sliced_region_properties,
            minimum_size=self.config["slicing"]["minimum_size"],
        )
        # ns-rse 2025-12-10 : Need to redo sliced_region_properties using "clean" version with small regions masked
        # Calculate region properties on clean slices after removal of small objects
        self.sliced_clean_region_properties = slicer.region_properties_by_slices(
            array=self.sliced_segments_clean, spacing=self.pixel_to_nm_scaling
        )
        # Count the number of pores per layer
        self.pores_per_layer = statistics.count_pores(
            sliced_region_properties=self.sliced_region_properties
        )
        # Plot all segmented layers
        plotting.plot_all_layers(
            array=self.sliced_segments_clean,
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
            self.area_by_layer = statistics.area_pores(
                sliced_region_properties=self.sliced_region_properties
            )
            # Plot area per layer
            self.fig_area_per_layer = plotting.plot_area_by_layer(
                area_per_layer=self.area_by_layer,
                img_name=self.filename,
                outdir=self.config["output_dir"],
                format=self.config["slicing"]["format"],
                log=False,
            )
            # Plot log area per layer
            self.fig_log_area_per_layer = plotting.plot_area_by_layer(
                area_per_layer=self.area_by_layer,
                img_name=self.filename,
                outdir=self.config["output_dir"],
                format=self.config["slicing"]["format"],
                log=True,
            )
        # Centroid
        if self.config["slicing"]["centroid"]:
            self.centroid_by_layer = statistics.centroid_pores(
                sliced_region_properties=self.sliced_region_properties
            )
        # Feret Maximum
        if self.config["slicing"]["feret_maximum"]:
            self.feret_maximum_by_layer = statistics.feret_diameter_maximum_pores(
                sliced_region_properties=self.sliced_region_properties
            )

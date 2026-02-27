"""Calculate statistics on grain volumes and aggregate and summarise the data."""

from __future__ import annotations

from typing import Any

# import polars as pl
import numpy as np
import numpy.typing as npt
from scipy.signal import find_peaks, peak_widths
from scipy.stats import norm


def count_pores(sliced_region_properties: list[Any]) -> npt.NDArray[np.int32]:
    """
    Count the number of ``region_properties``, each of which represents a pore for all layers.

    Parameters
    ----------
    sliced_region_properties : list[Any]
        A list of region properties found on each layer.

    Returns
    -------
    npt.NDArray[np.int32]
        A list of the number of region properties detected in each layer.
    """
    return np.asarray([len(region_props) for region_props in sliced_region_properties])


def area_pores(sliced_region_properties: list[list[Any]]) -> list[list[float]]:
    """
    Extract area of objects in each layer.

    Parameters
    ----------
    sliced_region_properties : list[Any]
        A list (one for each layer) of lists which contain the ``region_props`` for each object within that layer.

    Returns
    -------
    list[list[float]]
        A list with the same length as the number of layers, each item is a list of the area of objects within that
        layer.
    """
    return [[props.area for props in layer] for layer in sliced_region_properties]


def sum_area_by_layer(
    areas: list[list[float] | int | float],
    min_size: float | None = None,
) -> list[float]:
    """
    Sum the area of pores on each layer.

    Parameters
    ----------
    areas : list[list[float]]
        A list of areas of pores on each layer.
    min_size : float, optional
        Minimum size to include in calculation.

    Returns
    -------
    list[float]
        A list with the total area per slice.
    """
    # Sum the area per layer, we do this in a for loop rather than dictionary comprehension for instances when there is
    # a single object in a layer which will not therefore be iterrable and raise an error with sum().
    total_area_per_layer = []
    if min_size:
        _areas = [
            [pore_area for pore_area in layer if pore_area > min_size]
            for layer in areas  # type: ignore[union-attr]
        ]
    else:
        _areas = areas
    for layer in _areas:
        try:
            total_area_per_layer.append(sum(layer))
        except TypeError:
            if isinstance(layer, (int, float)):  # type: ignore[unreachable]
                total_area_per_layer.append(layer)
    return total_area_per_layer


def centroid_pores(
    sliced_region_properties: list[list[Any]],
) -> list[list[tuple[float, float]]]:
    """
    Extract centroid of objects in each layer.

    Parameters
    ----------
    sliced_region_properties : list[Any]
        A list (one for each layer) of lists which contain the ``region_props`` for each object within that layer.

    Returns
    -------
    list[list[float]]
        A list with the same length as the number of layers, each item is a list of the centroid of objects within that
        layer.
    """
    return [[props.centroid for props in layer] for layer in sliced_region_properties]


def feret_diameter_maximum_pores(
    sliced_region_properties: list[list[Any]],
) -> list[list[float]]:
    """
    Extract the maximum feret diameter of objects in each layer.

    Parameters
    ----------
    sliced_region_properties : list[Any]
        A list (one for each layer) of lists which contain the ``region_props`` for each object within that layer.

    Returns
    -------
    list[list[float]]
        A list with the same length as the number of layers, each item is a list of the maximum feret diameter of
        objects within that layer.
    """
    return [
        [props.feret_diameter_max for props in layer]
        for layer in sliced_region_properties
    ]


def calculate_pdf(array: list[float], xmin, xmax) -> dict[str, npt.NDArray]:
    """
    Calculate the scaled probability density function for an array.

    Parameters
    ----------
    array : list[float]
        Array of data points to be summarised.
    xmin : int | float
        Minimum value.
    xmax : int | float
        Maximum value.

    Returns
    -------
    dict[str, npt.NDArray]
        Dictionary of x and y values for the PDF.
    """
    x_values = np.arange(0, len(array))
    mean = np.average(x_values, weights=array)
    std = np.sqrt(np.average((x_values - mean) ** 2, weights=array))
    x_pdf = np.linspace(xmin, xmax, len(array))
    # Scale the PDF to match the total counts and "bin width" (i.e. layers) for plotting
    y_pdf = norm.pdf(x_pdf, loc=mean, scale=std) * np.sum(array) * (x_pdf[1] - x_pdf[0])
    return {"x": x_pdf, "y": y_pdf}


def full_width_half_max(pdf: npt.NDArray) -> list[int]:
    """
    Calculate the full-width half max.

    We are interested in the layers that cover the full-width half-max of the number of pores in an image. And therefore
    extract the indices the calculated PDF (``y_pdf``)

    Parameters
    ----------
    pdf : npt.NDArray
        Array probability density function for which peak and full-width half-max are to be calculated.

    Returns
    -------
    dict[str, int]
        Dictionary of the lower and upper layers for the full-width half-max range.
    """
    peaks, _ = find_peaks(pdf)
    if len(peaks) > 0:
        _peak_widths = peak_widths(pdf, peaks, rel_height=0.5)
        # Round these as we want indexes not absolute values
        return [np.round(_peak_widths[2])[0], np.round(_peak_widths[3])[0]]
    msg = "No peaks found in distribution, can not calculate full-width half-max."
    raise ValueError(msg)

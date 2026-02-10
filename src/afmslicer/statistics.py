"""Calculate statistics on grain volumes and aggregate and summarise the data."""

from __future__ import annotations

from typing import Any

# import polars as pl
import numpy as np
import numpy.typing as npt


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
    areas: list[list[float] | int | float], min_size: float | None = None
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
            for layer in areas
        ]
    else:
        _areas = areas
    for layer in _areas:
        try:
            total_area_per_layer.append(sum(layer))
        except TypeError:
            if isinstance(layer, (int, float)):
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


def fit_gaussian(array: npt.NDArray[int | np.float64]) -> tuple[float, float]:
    """
    Calculate the weighted mean layer and standard deviation from the data.

    It is assumed the number of pores follow a Gaussian distribution moving through the layers. As we have the number of
    pores for each layer we weight the layers by the number of pores to get the centrality of the fitted gaussian
    distribution and the variance in layers around this.

    Parameters
    ----------
    array : npt.NDArray[np.int | np.float64]
        Array for which Gaussian curve is to be fitted.

    Returns
    -------
    tuple[float, float]
        Returns the weighted layer mean and standard deviation of the data.
    """
    x_values = np.arange(1, len(array) + 1)
    mean = np.average(x_values, weights=array)
    std = np.sqrt(np.average((x_values - mean) ** 2, weights=array))
    return (mean, std)


# def aggregate_arrays(
#     arrays: dict[str, npt.NDArray[np.float64]],
# ) -> npt.NDArray[np.float64]:
#     print(f"\n{list(arrays.values())=}\n")
#     return np.concatenate(list(arrays.values()))

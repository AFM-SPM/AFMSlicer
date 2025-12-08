"""Calculate statistics on grain volumes and aggregate and summarise the data."""

from __future__ import annotations

from typing import Any

# import polars as pl


def count_pores(sliced_region_properties: list[Any]) -> list[int]:
    """
    Count the number of ``region_properties``, each of which represents a pore for all layers.

    Parameters
    ----------
    sliced_region_properties : list[Any]
        A list of region properties found on each layer.

    Returns
    -------
    list[int]
        A list of the number of region properties detected in each layer.
    """
    return [len(region_props) for region_props in sliced_region_properties]


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


# def aggregate_arrays(
#     arrays: dict[str, npt.NDArray[np.float64]],
# ) -> npt.NDArray[np.float64]:
#     print(f"\n{list(arrays.values())=}\n")
#     return np.concatenate(list(arrays.values()))

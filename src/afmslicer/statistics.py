"""Calculate statistics on grain volumes and aggregate and summarise the data."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
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
    areas: list[list[float]],
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


def volume_by_layer(
    areas: list[list[int | float] | int | float],
    scale: float | None = None,
) -> npt.NDArray[np.float64]:
    """
    Calculate the volume of objects in each layer.

    Parameters
    ----------
    areas :  list[list[int | float] | int | float]
        List of areas of pores on each layer.
    scale : float, optional
        Scaling for calculating volume. This may be the pixel to nanometer scaling if slicing has been performed
        relative to this factor, or if an arbitrary number of slices have been taken it should be the range of heights
        divided by the number of slices.

    Returns
    -------
    npt.NDArray[np.float64]
        A numpy array of volumes for each object in each slice.
    """
    scale = 1 if scale is None else scale
    return np.asarray(areas) * scale


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


def create_statistics_dictionary(
    sliced_region_properties: list[list[Any]],
    feret_maximum: bool = False,
    centroid: bool = False,
) -> dict[int, dict[int, Any]]:
    """
    Extract statistics of objects from each layer.

    Parameters
    ----------
    sliced_region_properties : list[list[Any]]
        List of lists of region properties, the top level is layer, the nesting within it is each of the objects within
        the layer.
    feret_maximum : bool
        Whether to extract the maximum feret distance.
    centroid : bool
        Whether to extract the centroid coordinates of the region.

    Returns
    -------
    dict[int, dict[int, Any]]
        Dictionary of statistics, top-level is the layer/slice through the image, and nested within are statistics for
        each pore.
    """
    statistics = {}
    for layer, slice_properties in enumerate(sliced_region_properties):
        statistics[layer] = {}
        for pore, props in enumerate(slice_properties):
            statistics[layer][pore] = {"area": props.area}
            if feret_maximum:
                statistics[layer][pore]["feret_diameter_max"] = props.feret_diameter_max
            if centroid:
                statistics[layer][pore]["centroid"] = props.centroid
    return statistics


def calculate_pdf(
    array: list[float], xmin: float | None = None, xmax: float | None = None
) -> dict[str, npt.NDArray]:
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
    xmin = 0 if xmin is None else xmin
    xmax = len(array) if xmax is None else xmax
    x_values = np.arange(0, len(array))
    mean = np.average(x_values, weights=array)
    std = np.sqrt(np.average((x_values - mean) ** 2, weights=array))
    x_pdf = np.linspace(xmin, xmax, len(array))
    # Scale the PDF to match the total counts and "bin width" (i.e. layers) for plotting
    y_pdf = norm.pdf(x_pdf, loc=mean, scale=std) * np.sum(array) * (x_pdf[1] - x_pdf[0])
    return {"x": x_pdf, "y": y_pdf}


def full_width_half_max(pdf: npt.NDArray[np.float32]) -> tuple[int, int]:
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
    tuple[int, int]
        Dictionary of the lower and upper layers for the full-width half-max range.
    """
    peaks, _ = find_peaks(pdf)
    if len(peaks) > 0:
        _peak_widths = peak_widths(pdf, peaks, rel_height=0.5)
        # Round these as we want indexes not absolute values
        return (int(np.round(_peak_widths[2])[0]), int(np.round(_peak_widths[3])[0]))
    msg = "No peaks found in distribution, can not calculate full-width half-max."
    raise ValueError(msg)


def classify_pore_size(
    df: pd.DataFrame,
    area_thresholds: dict[str, int],
    pore_colors: list[str],
    area_val: str = "area",
) -> pd.DataFrame:
    """
    Classify pore sizes in a dataframe into colors.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with areas to be classified.
    area_thresholds : dict[str, int]
        Dictionary of thresholds, there should be three values for low, medium and high thresholds resulting in four
        categories.
    pore_colors : list[str]
        Colors to use for the four categories.
    area_val : str
        Column name containing the area data, default is ``area`` and is unlikely to need changing.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional column with text categorisation of pore area.
    """
    if len(pore_colors) != 4:
        msg = f"'pore_colors' should have four values : {pore_colors=}"
        raise ValueError(msg)
    if len(area_thresholds) != 3:
        msg = f"'area_thresholds' should have three values : {area_thresholds=}"
        raise ValueError(msg)
    if len(set(area_thresholds.keys()) - set(["low", "medium", "high"])):  # noqa: C405
        msg = f"Keys to 'area_thresholds are not 'low', 'medium' and 'high' : {area_thresholds.keys()=}"
        raise ValueError(msg)
    df["pore_color"] = df[area_val].case_when(
        [
            # NB - We do not need the lower boundary for 'medium' and 'high' since once a condition is metadata
            #      the subsequent conditions are ignored
            (df[area_val] < area_thresholds["low"], pore_colors[0]),
            (df[area_val] < area_thresholds["medium"], pore_colors[1]),
            (df[area_val] < area_thresholds["high"], pore_colors[2]),
            (df[area_val] >= area_thresholds["high"], pore_colors[3]),
        ]
    )
    # Force type as `str`, seems case_when() doesn't enforce new string type in Pandas 3.0.1
    # https://pandas.pydata.org/docs/user_guide/migration-3-strings.html
    df["pore_color"] = df["pore_color"].astype("str")
    df.index.names = ["image", "layer", "pore"]
    return df.reset_index()


def summarise_pores(df: pd.DataFrame, pore_colors: list[str]) -> pd.DataFrame:
    """
    Summarise pore types by image, layer and color.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe to aggregate, typically will be ``AFMSlicer.statistics``. Must have columns ``image``,
        ``layer``, ``pore_color`` and ``counter``.
    pore_colors : list[str]
        List of pore color columns that should be present in the resulting dataframe.

    Returns
    -------
    pd.DataFrame
        Aggregated data frame of counts of ``pore_color`` by ``image``/``layer`` with counts of each ``pore_color``.
    """
    # Aggregate data counting the number pore types by image and layer
    color_count_df = df[["image", "layer", "pore_color", "pore"]].pivot_table(
        index=["image", "layer"], columns="pore_color", aggfunc="count", fill_value=0
    )
    color_count_df = color_count_df.droplevel(level=0, axis=1)
    color_count_df.columns.name = None
    for pore_color in pore_colors:
        color_count_df = _add_missing_column(df=color_count_df, pore_color=pore_color)
    color_count_df["total"] = color_count_df.sum(axis=1)
    return color_count_df.reset_index()


def _add_missing_column(df: pd.DataFrame, pore_color: str) -> pd.DataFrame:
    """
    Add a missing column to the dataframe.

    There should be a column for each ``pore_color`` defined in the configuration, if none are observed across all
    images/layers then the reshaped data frame is missing the column. We therefore add a column where all counts are
    ``0`` if it is not present.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to check column is present.
    pore_color : str
        Name of column to check and add if missing.

    Returns
    -------
    pd.DataFrame
        Dataframe with column added, all values set to ``0``.
    """
    if pore_color not in df.columns:
        df[pore_color] = 0
        return df
    return df


def cumulative_area(areas: npt.NDArray[np.float64]) -> pd.DataFrame:
    """Sort and calculate the cumulative sum of a single layer of areas.

    Parameters
    ----------
    areas : npt.NDArray[np.float64]
        An array of areas for a single slice.

    Returns
    -------
    pd.DataFrame
        Pandas dataframe of sorted area, cumulative sum of the area and the cumulative fraction.

    Examples
    --------

    >>> import numpy as np
    >>> from afmslicer import statistics
    >>>
    >>> areas = np.asarray([10, 1, 9, 2, 8, 3, 7, 4, 6, 5]),
    >>> statistics.cumulative_area(areas)
    """
    sorted_areas = np.sort(areas)
    cumulative_sum = np.cumsum(sorted_areas)
    cumulative_fraction = cumulative_sum / cumulative_sum[-1]
    return pd.DataFrame(
        {
            "area_sorted": sorted_areas.tolist(),
            "cumulative_sum": cumulative_sum.tolist(),
            "cumulative_fraction": cumulative_fraction.tolist(),
        }
    )


def cumulative_areas(areas: list[npt.NDArray[np.float64]]) -> list[pd.DataFrame]:
    """Calculate cumulative area for objects across slices.

    Parameters
    ----------
    areas : list[npt.NDArray[np.float64]]
        List of numpy arrays of the area of objects in layers.

    Returns
    -------
    list[pd.DataFrame]
        A list of dataframes with the areas sorted by size, the cumulative area and the cumulative area fraction.

    Examples
    --------
    >>> import numpy as np
    >>> from afmslicer import statistics
    >>>
    >>> areas = [
                np.asarray([10, 1, 9, 2, 8, 3, 7, 4, 6, 5]),
                np.asarray(
                    [
                        4.23384259,
                        8.50731915,
                        5.59014498,
                        7.24935116,
                        7.57840579,
                        8.98832143,
                        1.51457266,
                        8.49551991,
                        2.12878874,
                        3.22178091,
                    ]
                ),
            ]
    >>> statistics.cumulative_areas(areas)
    """
    return [cumulative_area(area) for area in areas]


def concatenate_areas(areas: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate a list of dataframes appending the layer id.

    Used for concatenating the cumulative area across layers.

    Parameters
    ----------
    areas : list[pd.DataFrame]
        List of dataframes to be concatenated.

    Returns
    -------
    pd.DataFrame
        A single dataframe with the layer added to identify which layer data pertains to.

    Examples
    --------
    >>> import pandas as pd
    >>> from afmslicer import statistics
    >>>
    >>> cumulative_areas = [
                pd.DataFrame(
                    {
                        "area_sorted": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        "cumulative_sum": [1, 3, 6, 10, 15, 21, 28, 36, 45, 55],
                        "cumulative_fraction": [
                            0.018182,
                            0.054545,
                            0.109091,
                            0.181818,
                            0.272727,
                            0.381818,
                            0.509091,
                            0.654545,
                            0.818182,
                            1.000000,
                        ],
                    }
                ),
                pd.DataFrame(
                    {
                        "area_sorted": [
                            1.514573,
                            2.128789,
                            3.221781,
                            4.233843,
                            5.590145,
                            7.249351,
                            7.578406,
                            8.495520,
                            8.507319,
                            8.988321,
                        ],
                        "cumulative_sum": [
                            1.514573,
                            3.643361,
                            6.865142,
                            11.098985,
                            16.689130,
                            23.938481,
                            31.516887,
                            40.012407,
                            48.519726,
                            57.508047,
                        ],
                        "cumulative_fraction": [
                            0.026336708175331593,
                            0.06335394035771619,
                            0.11937707207826649,
                            0.19299881350935774,
                            0.29020512185250114,
                            0.4162631519515137,
                            0.5480430704702286,
                            0.6957705678538069,
                            0.8437032406962971,
                            1.0,
                        ],
                    }
                ),
            ]
    >>> statistics.concatenate_areas(cumulative_areas)
    """
    for layer, area in enumerate(areas):
        area["layer"] = layer
    return pd.concat(areas).reset_index(drop=True)


def find_nearest(
    areas: pd.DataFrame,
    fraction: float = 0.5,
    cum_fraction_col: str = "cumulative_fraction",
    cum_area_col: str = "cumulative_sum",
) -> dict[str, float]:
    """Extract the area and proportion closes to the specified fraction.

    Parameters
    ----------
    areas : pd.DataFrame
        Pandas dataframe of the sorted area, cumulative area and cumulative fraction of the cumulative area.
    fraction : float
        Proportion of interest, default is `0.5` and typically won't need changing.
    cum_fraction_col : str
        Column name for the cumulative fraction. Default is `cumulative_fraction` and typically won't need changing.
    cum_area_col : str
        Column name for the cumulative area. Default is `cumulative_area` and typically won't need changing.

    Returns
    -------
    dict[str, float]
        A dictionary of the cumulative area and fraction.

    Examples
    --------
    >>> import pandas as pd
    >>> from afmslicer import statistics
    >>>
    >>> cumulative_areas = pd.DataFrame(
                {
                    "area_sorted": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    "cumulative_sum": [1, 3, 6, 10, 15, 21, 28, 36, 45, 55],
                    "cumulative_fraction": [
                        0.018182,
                        0.054545,
                        0.109091,
                        0.181818,
                        0.272727,
                        0.381818,
                        0.509091,
                        0.654545,
                        0.818182,
                        1.000000,
                    ],
                }
            )
    >>> statistics.find_nearest(cumulative_areas, fraction=0.5)
    >>> statistics.find_nearest(cumulative_areas, fraction=0.2)
    """
    index = (np.abs(areas[cum_fraction_col].to_numpy() - fraction)).argmin()
    return {
        "fraction": fraction,
        "area": areas[cum_area_col][index],
        "cumulative_fraction": areas[cum_fraction_col][index],
    }


def hcfa(areas: list[pd.DataFrame], fraction: float = 0.5) -> pd.DataFrame:
    """
    Half Cumulative Fraction for the total Area (HCFA) for all layers.

    Parameters
    ----------
    areas : list[pd.DataFrame]
        List of pandas dictionaries with the `fraction`, `area` and `cumulative_fraction` for the given `fraction`.
    fraction : float
        The fraction at which to extract areas.

    Returns
    -------
    pd.DataFrame
        Pandas dataframe of the `layer`, `fraction`, `area` and `cumulative_fraction`.

    Examples
    --------
    >>> import pandas as pd
    >>> from afmslicer import statistics
    >>>
    >>> areas = [
                pd.DataFrame(
                    {
                        "area_sorted": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        "cumulative_sum": [1, 3, 6, 10, 15, 21, 28, 36, 45, 55],
                        "cumulative_fraction": [
                            0.018182,
                            0.054545,
                            0.109091,
                            0.181818,
                            0.272727,
                            0.381818,
                            0.509091,
                            0.654545,
                            0.818182,
                            1.000000,
                        ],
                    }
                ),
                pd.DataFrame(
                    {
                        "area_sorted": [
                            1.514573,
                            2.128789,
                            3.221781,
                            4.233843,
                            5.590145,
                            7.249351,
                            7.578406,
                            8.495520,
                            8.507319,
                            8.988321,
                        ],
                        "cumulative_sum": [
                            1.514573,
                            3.643361,
                            6.865142,
                            11.098985,
                            16.689130,
                            23.938481,
                            31.516887,
                            40.012407,
                            48.519726,
                            57.508047,
                        ],
                        "cumulative_fraction": [
                            0.026336708175331593,
                            0.06335394035771619,
                            0.11937707207826649,
                            0.19299881350935774,
                            0.29020512185250114,
                            0.4162631519515137,
                            0.5480430704702286,
                            0.6957705678538069,
                            0.8437032406962971,
                            1.0,
                        ],
                    }
                ),
            ]
    >>> statistics.hcfa(areas=areas, fraction=0.5)
    """
    half_area_fractions = [
        find_nearest(areas=area, fraction=fraction) for area in areas
    ]
    return pd.DataFrame(half_area_fractions).reset_index(names="layer")

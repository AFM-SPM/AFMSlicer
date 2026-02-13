"""Plot processed images."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.typing as mpt
import numpy as np
import numpy.typing as npt

# import numpy as np
from loguru import logger

from afmslicer import statistics


def plot_layer(  # pylint: disable=too-many-positional-arguments
    array: npt.NDArray,
    img_name: str | None = None,
    layer: int | None = None,
    outdir: str | Path | None = None,
    format: str | None = None,  # pylint: disable=redefined-builtin
    cmap: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a two-dimensional numpy array.

    Parameters
    ----------
    array : npt.NDArray
        Two-dimensional numpy array to be plotted, may be labelled or unlabelled data.
    img_name : str, optional
        The image name from which the data is derived.
    layer : int, optional
        The layer that is being plotted.
    outdir : str | Path, optional
        Target output directory, will be created if it does not exist.
    format : str, optional
        Image format for creating image. Default is ``png``.
    cmap : str, optional
        Colormap to plot as, defaults to ``black`` for binary images, ``viridis`` for heights. Labelled images are
        plotted as is.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Matplotlib ``Figure`` and ``Axes``.
    """
    if len(array.shape) != 2:
        msg = f"The input array should be two-dimensional but has {len(array.shape)}."
        raise AttributeError(msg)
    if cmap is None:
        cmap = "binary" if array.max() == 1 else "viridis"
        # ns-rse 2025-11-25 : Sort out datatypes of arrays so that np.int32 are plotted with the reverse jet colormap
        # elif array.dtype == np.int32:
        #     cmap = "jet_r"
    fig, ax = plt.subplots(1, 1)
    ax.imshow(array, cmap=cmap)
    plt.close()
    if outdir is not None:
        assert img_name is not None, (
            "If saving output you must supply an `img_name` parameter."
        )
        assert layer is not None, (
            "If saving output you must supply an `layer` parameter."
        )
        format = "png" if format is None else format.replace(".", "")
        outdir = outdir if isinstance(outdir, Path) else Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = f"{img_name}_{layer}.{format}"
        plt.imsave(fname=outdir / outfile, arr=array, format=format, cmap=cmap)
        logger.info(f"Image saved to : {outdir!s}/{outfile}")
        plt.close()
    return (fig, ax)


def plot_all_layers(
    array: npt.NDArray,
    img_name: str | None = None,
    outdir: str | Path | None = None,
    format: str | None = None,  # pylint: disable=redefined-builtin
    cmap: str | mpt.ColorType | None = None,
) -> dict[int, tuple[plt.Figure, plt.Axes]]:
    """
    Plot a three-dimensional numpy array.

    This function calls ``plot_layer()`` for each "slice" of a three-dimensional array.

    Parameters
    ----------
    array : npt.NDArray
        Two-dimensional numpy array to be plotted, may be labelled or unlabelled data.
    img_name : str, optional
        The image name from which the data is derived.
    outdir : str | Path, optional
        Target output directory, will be created if it does not exist.
    format : str, optional
        Image format for creating image. Default is ``png``.
    cmap : str | mpt.ColorType, optional
        Colormap to plot as, defaults to ``black`` for binary images, ``viridis`` for heights. Labelled images are
        plotted as is.

    Returns
    -------
    dict[int, tuple[plt.Figure, plt.Axes]]
        Dictionary of Matplotlib ``Figure`` and ``Axes`` indexed by layer.
    """
    if len(array.shape) != 3:
        msg = f"The input array should be three-dimensional but has {len(array.shape)}."
        raise AttributeError(msg)
    plots = {}
    for layer in range(array.shape[2]):
        plots[layer] = plot_layer(
            array[:, :, layer], img_name, layer, outdir, format, cmap
        )
    return plots


def plot_pores_by_layer(  # pylint: disable=too-many-positional-arguments,too-many-locals
    pores_per_layer: npt.NDArray[np.int32 | np.float64],
    img_name: str | None = None,
    outdir: str | Path | None = None,
    format: str | None = None,  # pylint: disable=redefined-builtin
    log: bool | None = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the layer v number of pores within it.

    Parameters
    ----------
    pores_per_layer : list[int]
        A list of the number of pores in each layer.
    img_name : str, optional
        Image name.
    outdir : str | Path, optional
        Output directory, if no ``None`` the image is saved there as ``<img_name>_pores_per_layer_[log].<format>``.
    format : str, optional
        Output file format as a string, defaults to ``png`` if not specified.
    log : bool, optional
        Whether to plot with the logarithm (base10) of the number of pores.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Returns a tuple of Matplotlib ``fig`` and ``ax``.
    """
    format = "png" if format is None else format.replace(".", "")
    if log:
        pores_per_layer = np.log10(pores_per_layer)
        ylabel = "log(n)"
        outfile = f"{img_name}_pores_per_layer_log.{format}"
    else:
        ylabel = "n"
        outfile = f"{img_name}_pores_per_layer.{format}"
    fig, ax = plt.subplots(1, 1)
    ax.plot(pores_per_layer)
    # Calculate the Gaussian and overlay
    if not log:
        xmin, xmax = plt.xlim()
        pdf = statistics.calculate_pdf(array=pores_per_layer, xmin=xmin, xmax=xmax)
        ax.plot(pdf["x"], pdf["y"], "k-", linewidth=1, label="Gaussian PDF (Scaled)")
        full_width_half_max_index = statistics.full_width_half_max(pdf=pdf["y"])
        ymin, ymax = plt.ylim()
        ax.vlines(full_width_half_max_index, ymin=ymin, ymax=ymax)
    ax.set_title("Pores per layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    if outdir is not None:
        assert img_name is not None, (
            "If saving output you must supply an `img_name` parameter."
        )
        outdir = outdir if isinstance(outdir, Path) else Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(fname=outdir / outfile, format=format)
        logger.info(f"Image saved to : {outdir!s}/{outfile}")
    return (fig, ax)


def plot_area_by_layer(  # pylint: disable=too-many-positional-arguments
    area_per_layer: list[list[float]],
    img_name: str | None = None,
    min_size: float | None = None,
    outdir: str | Path | None = None,
    format: str | None = None,  # pylint: disable=redefined-builtin
    log: bool | None = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the layer v total area of pores within it.

    Parameters
    ----------
    area_per_layer : list[list[float]]
        A list of the total area of pores within each layer.
    img_name : str, optional
        Image name.
    min_size : float
        Minimum size of objects to include when summing the area by layer.
    outdir : str | Path, optional
        Output directory, if no ``None`` the image is saved there as ``<img_name>_pores_per_layer_[log].<format>``.
    format : str, optional
        Output file format as a string, defaults to ``png`` if not specified.
    log : bool
        Whether to plot with the logarithm (base10) of the number of pores.

    Returns
    -------
    tuple[plt.Figure, plt.Axes]
        Returns a tuple of Matplotlib ``fig`` and ``ax``.
    """
    format = "png" if format is None else format.replace(".", "")
    total_area_per_layer = statistics.sum_area_by_layer(
        areas=area_per_layer,
        min_size=min_size,
    )
    if log:
        total_area_per_layer = np.log10(total_area_per_layer)
        ylabel = "log(Area)"
        outfile = f"{img_name}_area_per_layer_log.{format}"
    else:
        ylabel = "Area (nm^2)"
        outfile = f"{img_name}_area_per_layer.{format}"
    fig, ax = plt.subplots(1, 1)
    ax.plot(total_area_per_layer)
    if not log:
        xmin, xmax = plt.xlim()
        pdf = statistics.calculate_pdf(array=total_area_per_layer, xmin=xmin, xmax=xmax)
        ax.plot(pdf["x"], pdf["y"], "k-", linewidth=1, label="Gaussian PDF (Scaled)")
    ax.set_title("Area per layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    if outdir is not None:
        assert img_name is not None, (
            "If saving output you must supply an `img_name` parameter."
        )
        outdir = outdir if isinstance(outdir, Path) else Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        plt.savefig(fname=outdir / outfile, format=format)
        logger.info(f"Image saved to : {outdir!s}/{outfile}")
    return (fig, ax)

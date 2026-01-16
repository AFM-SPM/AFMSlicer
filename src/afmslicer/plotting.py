"""Plot processed images."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.typing as mpt
import numpy as np
import numpy.typing as npt

# import numpy as np
from loguru import logger
from scipy.stats import norm


def plot_layer(  # pylint: disable=too-many-positional-arguments
    array: npt.NDArray,
    img_name: str | None = None,
    layer: int | None = None,
    outdir: str | Path | None = None,
    format: str | None = None,  # pylint: disable=redefined-builtin
    cmap: str | mpt.ColorType | None = None,
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
    cmap : str | mpt.ColorType, optional
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
    mean: float | None = None,
    std: float | None = None,
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
    mean : float, optional
        Mean of ``pores_per_layer``.
    std : float, optional
        Standard deviation of ``pores_per_layer``.
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
        if mean is None or std is None:
            x_values = np.arange(1, len(pores_per_layer) + 1)
            mean = np.average(x_values, weights=pores_per_layer)
            std = np.sqrt(np.average((x_values - mean) ** 2, weights=pores_per_layer))
        xmin, xmax = plt.xlim()
        x_pdf = np.linspace(xmin, xmax, len(pores_per_layer))
        # Scale the PDF to match the total counts and "bin width" (i.e. layers) for plotting
        pdf_scaled = (
            norm.pdf(x_pdf, loc=mean, scale=std)
            * np.sum(pores_per_layer)
            * (x_pdf[1] - x_pdf[0])
        )
        ax.plot(x_pdf, pdf_scaled, "k-", linewidth=1, label="Gaussian PDF (Scaled)")
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


def plot_area_by_layer(
    area_per_layer: list[list[float]],
    img_name: str | None = None,
    outdir: str | Path | None = None,
    format: str | None = None,  # pylint: disable=redefined-builtin
    log: bool | None = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot the layer v total area of pores within it.

    Parameters
    ----------
    area_per_layer : list[list[float]]
        A list of lists of the area of pores within each layer.
    img_name : str, optional
        Image name.
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
    total_area_per_layer = [sum(layer) for layer in area_per_layer]
    if log:
        total_area_per_layer = np.log10(total_area_per_layer)
        ylabel = "log(Area)"
        outfile = f"{img_name}_area_per_layer_log.{format}"
    else:
        ylabel = "Area"
        outfile = f"{img_name}_area_per_layer.{format}"
    fig, ax = plt.subplots(1, 1)
    ax.plot(total_area_per_layer)
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

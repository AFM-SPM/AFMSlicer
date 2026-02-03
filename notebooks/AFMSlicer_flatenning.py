from __future__ import annotations

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium", auto_download=["html"])

with app.setup:
    # Initialization code that runs before all other cells
    pass


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # AFMSlicer Flatenning

    This notebook seekes to reproduce the steps pre-processing steps of flattening Atomic Force Microscopy (AFM) images that are traditionally undertaken using [Gwyddion](https://gwyddion.net) using the `Filters` module from [TopoStats](https://afm-spm.github.io/TopoStats).

    The steps in processing are detailed along with examples in these [slides](https://docs.google.com/presentation/d/1bOMfPmaRMs5TPFFGrxXpamf-GJswlHd4/) and are summarised as...

    1. Median difference
    2. Scar Removal
    3. Polynomial flattening, typically with 1 or 2 degrees[^1]
    4. Gaussian blurring (**Optional**)

    [^1]: Some images require higher degrees of polynomilas for flattening and different numbers of degrees for the horizontal and vertical planes.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Setup

    - Import libraries
    - Setup directory paths
    - Get a list of `.spm` files.
    """)
    return


@app.cell(hide_code=True)
def _():
    import pickle as pkl
    import re
    from datetime import datetime
    from pathlib import Path
    from typing import Any

    import matplotlib.pyplot as plt
    import numpy as np
    import numpy.typing as npt
    from topostats.classes import TopoStats
    from topostats.filters import Filters
    from topostats.io import LoadScans, read_yaml
    from topostats.scars import remove_scars

    from afmslicer.classes import AFMSlicer

    BASE_DIR = Path().cwd()
    PKG_BASE_DIR = BASE_DIR.parent
    PKG_SRC_DIR = PKG_BASE_DIR / "src" / "afmslicer"
    DATA_DIR = BASE_DIR / "data"
    SPM_DIR = DATA_DIR / "spm"
    NPY_DIR = DATA_DIR / "npy"
    OUT_DIR = BASE_DIR / "output" / "flattening"

    config = read_yaml(PKG_SRC_DIR / "default_config.yaml")

    spm_files = SPM_DIR.glob("**/*.spm")
    # Found that loading the files was hanging, suspicion of the crop file being the culprit so exclude it for now.
    spm_files_filtered = [
        spm_file
        for spm_file in spm_files
        if not re.search(pattern=r"crop", string=str(spm_file))
    ]
    load_scans = LoadScans(spm_files_filtered, config=config)
    load_scans.get_data()
    # len(load_scans.img_dict)
    # list(spm_files_filtered)
    return (
        AFMSlicer,
        Any,
        Filters,
        LoadScans,
        NPY_DIR,
        OUT_DIR,
        PKG_SRC_DIR,
        Path,
        TopoStats,
        config,
        datetime,
        np,
        npt,
        pkl,
        plt,
        read_yaml,
        remove_scars,
    )


@app.cell(hide_code=True)
def _(npt, plt):
    def compare_arrays(
        arr1: npt.NDArray,
        arr2: npt.NDArray,
        arr1_name: str,
        arr2_name: str,
        cmap: str = "viridis",
    ) -> None:
        """
        Compare two arrays and print the mean and standard deviation difference.

        Subtract ``arr1 - arr2`` and calculate the mean and standard deviation of the differences.

        Parameters
        ----------
        arr1 : npt.NDArray
            Numpy array.
        arr2 : npt.NDArray
            Numpy array.
        arr1_name : str
            Description of ``arr1``.
        arr2_name : str
            Description of ``arr2``.#
        cmap : str
            Colormap to plot, defaults to ``viridis`` where blue is low and yellow is high.
        """
        diff = diff_arrays(arr1, arr2)
        plot_array(diff, cmap, arr1_name, arr2_name)
        return diff

    def plot_array(array: npt.NDArray, cmap: str = "viridis", arr1_name: str | None = None, arr2_name: str | None = None) -> None:
        fig, ax = plt.subplots()
        ax1 = ax.imshow(array, cmap=cmap)
        fig.colorbar(ax1, ax=ax)
        ax.set_title(f"Mean (std) difference between {arr1_name} - {arr2_name} : {array.mean():.4f} ({array.std():.4f})")
        plt.show()

    def plot_array_with_colorbar_title(
        array: npt.NDArray, fig, ax, cmap: str = "viridis", shrink: float = 0.75
    ) -> None:
        ax1 = ax.imshow(array, cmap=cmap)
        fig.colorbar(ax1, ax=ax, shrink=shrink)
        ax.set_title(
            f"{array.mean():.3f} ({array.std():.3f})", fontdict={"fontsize": 9}
        )
        return fig, ax

    def diff_arrays(arr1: npt.NDArray, arr2: npt.NDArray) -> None:
        """
        Calculate the difference between two arrays.

        Subtract ``arr1 - arr2`` and calculate the mean and standard deviation of the differences.

        Parameters
        ----------
        arr1 : npt.NDArray
            Numpy array.
        arr2 : npt.NDArray
            Numpy array.
        """
        return arr1 - arr2
    return (
        compare_arrays,
        diff_arrays,
        plot_array,
        plot_array_with_colorbar_title,
    )


@app.cell(hide_code=True)
def _(
    Any,
    Filters,
    TopoStats,
    default_config,
    diff_arrays,
    npt,
    plot_array_with_colorbar_title,
    plt,
    remove_scars,
):
    def afmslicer_filter(
        topostats_object: TopoStats,
        filter_config: dict[str, Any],
        cmap: str = "viridis",
        gaussian_blur: float = default_config["filter"]["gaussian_size"],
    ) -> tuple[TopoStats, Filters]:
        """
        Filter/flatten images using the four steps for AFMSlicer.

        - Median flatten
        - Scar removal
        - Nonlinear polynomial removal
        - Gaussian blurring

        Parameters
        ----------
        topostats_object : TopoStats
            ``TopoStats`` object with at least ``image_original`` attribute and ``filename``.
        filter_config : dict[str, Any]
            Dictionary of filtering options.

        Returns
        -------
        tuple[TopoStats, Filters]
            Returns the ``TopoStats`` object with cleaned ``image`` attribute and the ``Filters`` class.
        """
        if gaussian_blur is not None:
            filter_config["gaussian_size"] = gaussian_blur
        filter = Filters(topostats_object=topostats_object, **filter_config)
        # Median Flatten
        filter.images["initial_median_flatten"] = filter.median_flatten(
            filter.images["pixels"],
            mask=None,
            row_alignment_quantile=filter.row_alignment_quantile,
        )
        # Removes scars
        filter.images["initial_scar_removal"], _ = remove_scars(
            filter.images["initial_median_flatten"],
            filename=filter.filename,
            **filter.remove_scars_config,
        )
        # Polynomial removal
        filter.images["initial_nonlinear_polynomial_removal"] = (
            filter.remove_nonlinear_polynomial(
                filter.images["initial_scar_removal"], mask=None
            )
        )
        # Gaussian Filter
        filter.images["gaussian_filtered"] = filter.gaussian_filter(
            image=filter.images["initial_nonlinear_polynomial_removal"],
        )
        # Plot the images
        plot_comparison(topostats_object=topostats_object, filter=filter, cmap=cmap)

        return filter

    def _plot_array(array: npt.NDArray, fig, ax, cmap: str = "viridis") -> None:
        ax1 = ax.imshow(array, cmap=cmap)
        fig.colorbar(ax1, ax=ax)
        ax.set_title(
            f"{array.mean():.3f} ({array.std():.3f})", fontdict={"fontsize": 9}
        )
        return fig, ax

    def plot_comparison(
        topostats_object: TopoStats, filter: Filters, cmap: str = "viridis"
    ):
        fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(8, 12))
        plt.style.use("grayscale")
        # Original
        ax1 = plt.subplot(5, 3, 1)
        fig, ax1 = plot_array_with_colorbar_title(
            array=topostats_object.image_original, fig=fig, ax=ax1, cmap=cmap
        )
        # plt.title("Original")

        # Median Flattened
        ax4 = plt.subplot(5, 3, 4)
        fig, ax4 = plot_array_with_colorbar_title(
            array=filter.images["initial_median_flatten"], fig=fig, ax=ax4, cmap="gray"
        )
        # plt.title("Median Flattened")
        ax5 = plt.subplot(5, 3, 5)
        fig, ax5 = plot_array_with_colorbar_title(
            array=diff_arrays(
                topostats_object.image_original, filter.images["initial_median_flatten"]
            ),
            fig=fig,
            ax=ax5,
            cmap="viridis",
        )
        ax6 = plt.subplot(5, 3, 6)
        fig, ax6 = plot_array_with_colorbar_title(
            array=diff_arrays(
                topostats_object.image_original, filter.images["initial_median_flatten"]
            ),
            fig=fig,
            ax=ax6,
            cmap="viridis",
        )
        # plt.title("Original - Median Flattened")

        # Scar removal
        ax7 = plt.subplot(5, 3, 7)
        # ax7.imshow(filter.images["initial_scar_removal"])
        fig, ax7 = plot_array_with_colorbar_title(
            array=filter.images["initial_scar_removal"], fig=fig, ax=ax7, cmap="gray"
        )
        # plt.title("Scar Removal")
        ax8 = plt.subplot(5, 3, 8)
        fig, ax8 = plot_array_with_colorbar_title(
            array=diff_arrays(
                topostats_object.image_original, filter.images["initial_scar_removal"]
            ),
            fig=fig,
            ax=ax8,
            cmap="viridis",
        )
        # plt.title("Original - Scar Removal")
        ax9 = plt.subplot(5, 3, 9)
        fig, ax9 = plot_array_with_colorbar_title(
            array=diff_arrays(
                filter.images["initial_median_flatten"],
                filter.images["initial_scar_removal"],
            ),
            fig=fig,
            ax=ax9,
            cmap="viridis",
        )
        # plt.title("Median Flattened - Scar Removal")

        # Polynomial removal
        ax10 = plt.subplot(5, 3, 10)
        # ax10.imshow(filter.images["initial_nonlinear_polynomial_removal"])
        fig, ax10 = plot_array_with_colorbar_title(
            array=filter.images["initial_nonlinear_polynomial_removal"],
            fig=fig,
            ax=ax10,
            cmap="gray",
        )
        # plt.title("Nonlinear Polynomial")
        ax11 = plt.subplot(5, 3, 11)
        fig, ax11 = plot_array_with_colorbar_title(
            array=diff_arrays(
                topostats_object.image_original,
                filter.images["initial_nonlinear_polynomial_removal"],
            ),
            fig=fig,
            ax=ax11,
            cmap="viridis",
        )
        # plt.title("Original - Nonlinear Polynomial")
        ax12 = plt.subplot(5, 3, 12)
        fig, ax12 = plot_array_with_colorbar_title(
            array=diff_arrays(
                filter.images["initial_scar_removal"],
                filter.images["initial_nonlinear_polynomial_removal"],
            ),
            fig=fig,
            ax=ax12,
            cmap="viridis",
        )
        # plt.title("Scar Removal - Nonlinear Polynomial")

        # Gaussian
        ax13 = plt.subplot(5, 3, 13)
        # ax13.imshow(filter.images["gaussian_filtered"])
        fig, ax13 = plot_array_with_colorbar_title(
            array=filter.images["gaussian_filtered"], fig=fig, ax=ax13, cmap="gray"
        )
        # plt.title("Gaussian Filtered")
        ax14 = plt.subplot(5, 3, 14)
        fig, ax14 = plot_array_with_colorbar_title(
            array=diff_arrays(
                topostats_object.image_original, filter.images["gaussian_filtered"]
            ),
            fig=fig,
            ax=ax14,
            cmap="viridis",
        )
        # plt.title("Original - Gaussian Filtered")
        ax15 = plt.subplot(5, 3, 15)
        fig, ax15 = plot_array_with_colorbar_title(
            array=diff_arrays(
                filter.images["initial_nonlinear_polynomial_removal"],
                filter.images["gaussian_filtered"],
            ),
            fig=fig,
            ax=ax15,
            cmap="viridis",
        )
        # plt.title("Nonlinear Polynomial - Gaussian Filtered")

        cols = [
            f"Image\n{topostats_object.image_original.mean():.3f} ({topostats_object.image_original.std():.3f})",
            "Diff Original",
            "Diff Previous",
        ]
        rows = [
            "Original",
            "Median Flattened",
            "Scar Removal",
            "Nonlinear Polynomial",
            "Gaussian Filtered",
        ]
        for ax, col in zip(axes[0], cols, strict=False):
            ax.set_title(col)
        for ax, row in zip(axes[:, 0], rows, strict=False):
            ax.set_ylabel(row, rotation=90, size="medium")
        axes[0, 1].axis("off")
        axes[0, 2].axis("off")

        fig.tight_layout()
        plt.show()
        # topostats_object.image = filter.images["gaussian_filtered"].copy()

    # case1_filtered_test = afmslicer_filter(topostats_object=case1, filter_config=filters_config)
    return (afmslicer_filter,)


@app.cell(hide_code=True)
def _(
    Any,
    Filters,
    TopoStats,
    diff_arrays,
    plot_array_with_colorbar_title,
    plt,
    remove_scars,
):
    def topostats_filter(
        topostats_object: TopoStats,
        filter_config: dict[str, Any],
        cmap: str = "viridis",
    ) -> Filters:
        filter = Filters(topostats_object=topostats_object, **filter_config)
        filter.images["initial_median_flatten"] = filter.median_flatten(
            filter.images["pixels"],
            mask=None,
            row_alignment_quantile=filter.row_alignment_quantile,
        )
        filter.images["initial_tilt_removal"] = filter.remove_tilt(
            filter.images["initial_median_flatten"], mask=None
        )
        filter.images["initial_quadratic_removal"] = filter.remove_quadratic(
            filter.images["initial_tilt_removal"], mask=None
        )
        filter.images["initial_nonlinear_polynomial_removal"] = (
            filter.remove_nonlinear_polynomial(
                filter.images["initial_quadratic_removal"], mask=None
            )
        )

        # Remove scars
        filter.images["initial_scar_removal"], _ = remove_scars(
            filter.images["initial_nonlinear_polynomial_removal"],
            filename=filter.filename,
            **filter.remove_scars_config,
        )

        # Gaussian
        filter.images["gaussian_filtered"] = filter.gaussian_filter(
            image=filter.images["initial_scar_removal"]
        )

        # Plot
        plot_topostats_comparison(
            topostats_object=topostats_object, filter=filter, cmap=cmap
        )
        return filter

    def plot_topostats_comparison(
        topostats_object: TopoStats, filter: Filters, cmap: dict[str, str] = {"image": "gray", "diff": "viridis"}  # noqa: B006
    ) -> None:
        fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(8, 12))
        plt.style.use("grayscale")
        # Original
        cmap = "gray" if cmap is None else cmap["image"]
        ax1 = plt.subplot(7, 3, 1)
        fig, ax1 = plot_array_with_colorbar_title(
            array=topostats_object.image_original, fig=fig, ax=ax1, cmap=cmap
        )
        # plt.title("Original")

        # Median Flattened
        cmap = "gray" if cmap is None else cmap["image"]
        ax4 = plt.subplot(7, 3, 4)
        fig, ax4 = plot_array_with_colorbar_title(
            array=filter.images["initial_median_flatten"], fig=fig, ax=ax4, cmap=cmap
        )
        # plt.title("Median Flattened")
        cmap = "viridis" if cmap is None else cmap["diff"]
        ax5 = plt.subplot(7, 3, 5)
        fig, ax5 = plot_array_with_colorbar_title(
            array=diff_arrays(
                topostats_object.image_original, filter.images["initial_median_flatten"]
            ),
            fig=fig,
            ax=ax5,
            cmap=cmap,
        )
        cmap = "viridis" if cmap is None else cmap["diff"]
        ax6 = plt.subplot(7, 3, 6)
        fig, ax6 = plot_array_with_colorbar_title(
            array=diff_arrays(
                topostats_object.image_original, filter.images["initial_median_flatten"]
            ),
            fig=fig,
            ax=ax6,
            cmap=cmap,
        )
        # plt.title("Original - Median Flattened")

        # Tilt Removal
        cmap = "gray" if cmap is None else cmap["image"]
        ax7 = plt.subplot(7, 3, 7)
        fig, ax7 = plot_array_with_colorbar_title(
            array=filter.images["initial_tilt_removal"], fig=fig, ax=ax7, cmap=cmap
        )
        # plt.title("Tilt Removal")
        cmap = "viridis" if cmap is None else cmap["diff"]
        ax8 = plt.subplot(7, 3, 8)
        fig, ax8 = plot_array_with_colorbar_title(
            array=diff_arrays(
                topostats_object.image_original, filter.images["initial_tilt_removal"]
            ),
            fig=fig,
            ax=ax8,
            cmap=cmap,
        )
        # plt.title("Original - Tilt Removal")
        cmap = "viridis" if cmap is None else cmap["diff"]
        ax9 = plt.subplot(7, 3, 9)
        fig, ax9 = plot_array_with_colorbar_title(
            array=diff_arrays(
                filter.images["initial_median_flatten"],
                filter.images["initial_tilt_removal"],
            ),
            fig=fig,
            ax=ax9,
            cmap=cmap,
        )
        # plt.title("Median Flattened - Tilt Removal")

        # Quadratic removal
        cmap = "gray" if cmap is None else cmap["image"]
        ax10 = plt.subplot(7, 3, 10)
        fig, ax10 = plot_array_with_colorbar_title(
            array=filter.images["initial_quadratic_removal"],
            fig=fig,
            ax=ax10,
            cmap=cmap,
        )
        # plt.title("Quadratic")
        cmap = "viridis" if cmap is None else cmap["diff"]
        ax11 = plt.subplot(7, 3, 11)
        fig, ax11 = plot_array_with_colorbar_title(
            array=diff_arrays(
                topostats_object.image_original,
                filter.images["initial_quadratic_removal"],
            ),
            fig=fig,
            ax=ax11,
            cmap=cmap,
        )
        # plt.title("Original - Quadratic")
        cmap = "viridis" if cmap is None else cmap["diff"]
        ax12 = plt.subplot(7, 3, 12)
        fig, ax12 = plot_array_with_colorbar_title(
            array=diff_arrays(
                filter.images["initial_tilt_removal"],
                filter.images["initial_quadratic_removal"],
            ),
            fig=fig,
            ax=ax12,
            cmap=cmap,
        )
        # plt.title("Tilt Removal - Quadratic")

        # Nonlinear Polynomial
        cmap = "gray" if cmap is None else cmap["image"]
        ax13 = plt.subplot(7, 3, 13)
        fig, ax13 = plot_array_with_colorbar_title(
            array=filter.images["initial_nonlinear_polynomial_removal"],
            fig=fig,
            ax=ax13,
            cmap=cmap,
        )
        # plt.title("Nonlinear Polynomial")
        cmap = "viridis" if cmap is None else cmap["diff"]
        ax14 = plt.subplot(7, 3, 14)
        fig, ax14 = plot_array_with_colorbar_title(
            array=diff_arrays(
                topostats_object.image_original,
                filter.images["initial_nonlinear_polynomial_removal"],
            ),
            fig=fig,
            ax=ax14,
            cmap=cmap,
        )
        # plt.title("Original - Nonlinear Polynomial")
        cmap = "viridis" if cmap is None else cmap["diff"]
        ax15 = plt.subplot(7, 3, 15)
        fig, ax15 = plot_array_with_colorbar_title(
            array=diff_arrays(
                filter.images["initial_quadratic_removal"],
                filter.images["initial_nonlinear_polynomial_removal"],
            ),
            fig=fig,
            ax=ax15,
            cmap=cmap,
        )
        # plt.title("Quadratic - Nonlinear Polynomial")

        # Scar removal
        cmap = "gray" if cmap is None else cmap["image"]
        ax16 = plt.subplot(7, 3, 16)
        fig, ax16 = plot_array_with_colorbar_title(
            array=filter.images["initial_scar_removal"], 
            fig=fig, ax=ax16,
            cmap=cmap
        )
        # plt.title("Scar Removal")
        cmap = "viridis" if cmap is None else cmap["diff"]
        ax17 = plt.subplot(7, 3, 17)
        fig, ax17 = plot_array_with_colorbar_title(
            array=diff_arrays(
                topostats_object.image_original, filter.images["initial_scar_removal"]
            ),
            fig=fig,
            ax=ax17,
            cmap=cmap,
        )
        # plt.title("Original - Scar Removal")
        cmap = "viridis" if cmap is None else cmap["diff"]
        ax18 = plt.subplot(7, 3, 18)
        fig, ax18 = plot_array_with_colorbar_title(
            array=diff_arrays(
                filter.images["initial_nonlinear_polynomial_removal"],
                filter.images["initial_scar_removal"],
            ),
            fig=fig,
            ax=ax18,
            cmap=cmap,
        )
        # plt.title("Nonlinear Polynomial - Scar Removal")

        # Gaussian
        cmap = "gray" if cmap is None else cmap["image"]
        ax19 = plt.subplot(7, 3, 19)
        fig, ax19 = plot_array_with_colorbar_title(
            array=filter.images["gaussian_filtered"],
            fig=fig,
            ax=ax19,
            cmap=cmap
        )
        # plt.title("Gaussian Filtered")
        cmap = "viridis" if cmap is None else cmap["diff"]
        ax20 = plt.subplot(7, 3, 20)
        fig, ax20 = plot_array_with_colorbar_title(
            array=diff_arrays(
                topostats_object.image_original, filter.images["gaussian_filtered"]
            ),
            fig=fig,
            ax=ax20,
            cmap=cmap,
        )
        # plt.title("Original - Gaussian Filtered")
        cmap = "viridis" if cmap is None else cmap["diff"]
        ax21 = plt.subplot(7, 3, 21)
        fig, ax21 = plot_array_with_colorbar_title(
            array=diff_arrays(
                filter.images["initial_scar_removal"],
                filter.images["gaussian_filtered"],
            ),
            fig=fig,
            ax=ax21,
            cmap=cmap,
        )
        # plt.title("Scar Removal - Gaussian Filtered")

        cols = [
            f"Image\n{topostats_object.image_original.mean():.3f} ({topostats_object.image_original.std():.3f})",
            "Diff Original",
            "Diff Previous",
        ]
        rows = [
            "Original",
            "Median",
            "Tilt",
            "Quadratic",
            "Nonlinear",
            "Scar",
            "Gaussian",
        ]
        for ax, col in zip(axes[0], cols, strict=False):
            ax.set_title(col)
        for ax, row in zip(axes[:, 0], rows, strict=False):
            ax.set_ylabel(row, rotation=90, size="medium")
        axes[0, 1].axis("off")
        axes[0, 2].axis("off")

        fig.tight_layout()
        plt.show()

    # case1_topostats_filtered_test = topostats_filter(topostats_object=case1, filter_config=filters_config)
    return (topostats_filter,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Configuration File

    TopoStats uses [YAML](https://yaml.org) configuration files to set the parameters for processing a batch of images. AFMSlicer inherits this functionality from TopoStats and a "default" file is included in the package under `src/afmslicer/default_config.yaml`.

    We load the configuration file and show the values (inherited from TopoStats) that are the default.
    """)
    return


@app.cell
def _(PKG_SRC_DIR, read_yaml):
    default_config = read_yaml(PKG_SRC_DIR / "default_config.yaml")
    # default_config
    return (default_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For this work the section that is of interest is the `filter` configuration. We can look at just that section. The object `default_config` is a Python dictionary and so we can use the "key" name `filter` to list just the contents stored there. It shows that the...

    - `row_alignment_quantile` is set to `0.5` (i.e. the median).
    - `threshold_method` is set to `std_dev` (standard deviation)
    - The `threshold_std_dev.below` value is `10.0` so objects ten standard deviations below the mean are detected.
    - The `threshold_std_dev.below` value is `1.0` so objects one standard deviations above the mean are detected.
    - `gaussian_size`, the amount of gaussian blurring to apply is set to `1.0121397464510862`
    - `gaussian_method` is set to `nearest` (see SciPy [`scipy.ndimage.gaussian_filter`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html#scipy.ndimage.gaussian_filter) for description and other options).
    - `remove_scars.run` is `False` so scar removal is currently disabled.
    """)
    return


@app.cell
def _(default_config):
    default_config["filter"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We want to enable scar removal and so we modify the value in the dictionary.
    """)
    return


@app.cell
def _(default_config):
    default_config["filter"]["remove_scars"]["run"] = True
    default_config["filter"]["remove_scars"]
    # default_config["filter"]["gaussian_size"] = 3
    run_scar_removal = default_config["filter"]["remove_scars"]
    try:
        run_scar_removal.pop("run")
    except:  # noqa: E722
        pass
    filters_config = default_config["filter"]
    try:
        filters_config.pop("run")
    except:  # noqa: E722
        pass
    return filters_config, run_scar_removal


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Case 1 - External Cell Wall (Easy to Flatten)

    This sample image (`case2.spm`)
    """)
    return


@app.cell
def _(LoadScans, Path, config):
    case1_scans = LoadScans(
        [Path("/home/neil/work/git/hub/ns-rse/AFMSlicer/notebooks/data/spm/case1.spm")],
        config=config,
    )
    case1_scans.get_data()
    case1 = case1_scans.img_dict["case1"]
    # case1 = load_scans.img_dict["case1"]
    return (case1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Original Image

    This is the image as loaded, plotted using the `grayscale` colour scale.
    """)
    return


@app.cell
def _(case1, plot_array):
    plot_array(array=case1.image_original, cmap="gray")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Flattening

    We now step through the different stages of flattening the image using TopoStats' `Filter` class.

    We instantiate (create) the `case1_filter` object which is an instance of the `Filters` class with the parameters `case1` (our `TopoStats` class object that holds the loaded image) and provide the other arguments using the `**kwargs` option of Python which expands the dictionary, which uses the parameters for the `Filters` class as keys, with the values we want to provide as values.
    """)
    return


@app.cell
def _(Filters, case1, filters_config):
    case1_filter = Filters(topostats_object=case1, **filters_config)
    return (case1_filter,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Median Flattening

    The first step is to perform "median flattening". The `Filters` class has an attribute `images` which is a dictionary that is meant to hold the results of each step of processing so that users can look at these if required. We therefore store the result of median flattening in the `images["initial_median_flatten"]` key of the dictionary.
    """)
    return


@app.cell
def _(case1_filter, plot_array):
    case1_filter.images["initial_median_flatten"] = case1_filter.median_flatten(
        image=case1_filter.topostats_object.image_original
    )
    plot_array(array=case1_filter.images["initial_median_flatten"], cmap="gray")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This doesn't look too different from the original, perhaps a little "lighter". Lets check and see what differences there are across all cells. We can do this by subtracting the median flattened array (`case1_median_flattened`) from the original (`case1.image_original`) and calculate some statistics on it.
    """)
    return


@app.cell
def _(case1, case1_filter, compare_arrays):
    diff_orign_median = compare_arrays(
        arr1=case1.image_original,
        arr2=case1_filter.images["initial_median_flatten"],
        arr1_name="Original",
        arr2_name="Median flattened",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Scar Removal

    There do not appear to be any scars in this image. However, because TopoStats, and in turn AFMSlicer, are batch processing tools we don't have a choice of whether to run scar removal on any given image, it is enabled in the configuration file and all images are processed in the same way.

    #### Differences to TopoStats

    There are two processing steps which are undertaken in the normal TopoStats workflow which are not part of the described AFMSlicer workflow.

    1. [`Filters.remove_tilt()`](https://afm-spm.github.io/TopoStats/main/autoapi/topostats/filters/index.html#topostats.filters.Filters.remove_tilt) which..

    > Remove the planar tilt from an image (linear in 2D spaces).
    >
    > Uses a linear fit of the medians of the rows and columns to determine the linear slants in x and y directions and then subtracts the fit from the columns.

    2. [`Filters.remove_quadratic()`](https://afm-spm.github.io/TopoStats/main/autoapi/topostats/filters/index.html#topostats.filters.Filters.remove_quadratic)

    > Remove the quadratic bowing that can be seen in some large-scale AFM images.
    >
    > Use a simple quadratic fit on the medians of the columns of the image and then subtracts the calculated quadratic from the columns.

    It would be simple to include this tilt and quadratic removal in the AFMSlicer image processing if required and we apply this and compare the differences below.
    """)
    return


@app.cell
def _(case1_filter, plot_array, remove_scars, run_scar_removal):
    # remove_scars() returns the image with scars removed and a mask array which we don't need hence assigning it to '_'
    case1_filter.images["initial_scar_removal"], _ = remove_scars(
        img=case1_filter.images["initial_median_flatten"],
        filename=case1_filter.filename,
        **run_scar_removal,
    )
    plot_array(array=case1_filter.images["initial_scar_removal"], cmap="gray")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can check to see what difference scar removal has made to the image by subtracting arrays and summarising the mean. Unsurprisingly, in this instance as there are no scars to remove there is no difference in the images.
    """)
    return


@app.cell
def _(case1, case1_filter, compare_arrays):
    diff_orig_scar = compare_arrays(
        arr1=case1.image_original,
        arr2=case1_filter.images["initial_scar_removal"],
        arr1_name="Original",
        arr2_name="Initial Scar Removal",
    )

    diff_median_scar = compare_arrays(
        arr1=case1_filter.images["initial_median_flatten"],
        arr2=case1_filter.images["initial_scar_removal"],
        arr1_name="Initial Median Flattened",
        arr2_name="Initial Scar Removal",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Polynomial Flattening

    Next we flatten the image with polynomial flattening using the [`Filters.remove_nonlinear_polynomial()`](https://afm-spm.github.io/TopoStats/main/autoapi/topostats/filters/index.html#topostats.filters.Filters.remove_nonlinear_polynomial) method.

    > Fit and remove a “saddle” shaped nonlinear polynomial from the image.
    >
    > “Saddles” with the form a + b * x * y - c * x - d * y from the supplied image. AFM images sometimes contain a “saddle” shape trend to their background, and so to remove them we fit a nonlinear polynomial of x and y and then subtract the fit from the image.
    >
    > If these trends are not removed, then the image will not flatten properly and will leave opposite diagonal corners raised or lowered.

    #### Differences from TopoStats

    In the TopoStats workflow the polynomial flattening is undertaken **before** the removal of scars. For this particular image it likely doesn't make any difference as there are no scars to remove.
    """)
    return


@app.cell
def _(case1_filter, plot_array):
    case1_filter.images["initial_nonlinear_polynomial_removal"] = (
        case1_filter.remove_nonlinear_polynomial(
            case1_filter.images["initial_scar_removal"], mask=None
        )
    )
    plot_array(
        array=case1_filter.images["initial_nonlinear_polynomial_removal"], cmap="gray"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can again check what differences this has made to the heights within the image by comparing to the original and the array after median flattening and scar removal. We can see that the nonlinear polynomial has removed tilt across the diagonal acis of the image.
    """)
    return


@app.cell
def _(case1, case1_filter, compare_arrays):
    diff_orig_polynomial = compare_arrays(
        arr1=case1.image_original,
        arr2=case1_filter.images["initial_nonlinear_polynomial_removal"],
        arr1_name="Original",
        arr2_name="Initial Scar Removal",
    )

    diff_scar_polynomial = compare_arrays(
        arr1=case1_filter.images["initial_scar_removal"],
        arr2=case1_filter.images["initial_nonlinear_polynomial_removal"],
        arr1_name="Initial Scar Removal",
        arr2_name="Initial Nonlinear Polynomial Removal",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Gaussian Blur

    We now apply a Gaussian blur to the image using [`Filters.gaussian_filter()`](https://afm-spm.github.io/TopoStats/main/autoapi/topostats/filters/index.html#topostats.filters.Filters.gaussian_filter)
    """)
    return


@app.cell
def _(case1_filter, plot_array):
    case1_filter.images["gaussian_filtered"] = case1_filter.gaussian_filter(
        image=case1_filter.images["initial_nonlinear_polynomial_removal"]
    )
    plot_array(array=case1_filter.images["gaussian_filtered"], cmap="gray")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Again check the differences in heights by subtracting arrays...
    """)
    return


@app.cell
def _(case1, case1_filter, compare_arrays):
    diff_orig_gauss = compare_arrays(
        arr1=case1.image_original,
        arr2=case1_filter.images["gaussian_filtered"],
        arr1_name="Original",
        arr2_name="Gaussian Filtered",
    )

    diff_polynomial_gauss = compare_arrays(
        arr1=case1_filter.images["initial_nonlinear_polynomial_removal"],
        arr2=case1_filter.images["gaussian_filtered"],
        arr1_name="Initial Nonlinear Polynomial Removal",
        arr2_name="Initial Scar Removal",
    )
    return


@app.cell
def _(afmslicer_filter, case1, filters_config):
    case1_filtered = afmslicer_filter(
        topostats_object=case1, filter_config=filters_config
    )
    return (case1_filtered,)


@app.cell
def _(case1_filtered, plot_array):
    plot_array(case1_filtered.images["gaussian_filtered"], cmap="gray")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### TopoStats Filtering

    As mentioned there are some differences from the [described]() AFMSlicer filtering and those implemented in TopoStats.

    TopoStats performs the following steps...

    1. Median flattening
    2. Tilt removasl (not undertaken above)
    3. Quadratic Removal (not undertaken above)
    4. Remove nonlinear polynomial _before_ scar removal
    5. Scar removal
    6. Average background

    Because TopoStats is concerned with identifying "grains" (DNA and protein molecules) within an image it then goes on to perform thresholding using the specified method which removes background noise and artefacts which are masked so that only those items that are within the specified range are imaged and processed. Steps 1-6 are then repeated after a mask has been derived.

    We can apply these additional steps and sequence of flattening to the cell wall images and see how the images compare to the above.
    """)
    return


@app.cell
def _(case1, filters_config, topostats_filter):
    case1_topostats_filtered = topostats_filter(
        topostats_object=case1, filter_config=filters_config
    )
    return (case1_topostats_filtered,)


@app.cell
def _(case1_topostats_filtered, plot_array):
    plot_array(case1_topostats_filtered.images["gaussian_filtered"], cmap="gray")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### AFMSlicer v TopoStats

    We can now compare the image that has been flattened by the AFMSlicer and TopoStats methods.

    **NB** The arguments use different images from the `.images()` dictionary because the two methods process in different orders. For AFMSlicer the polynomial removal is the last step, for TopoStats the last step is scar removal.

    The image below suggests that there is a systematic difference between the AFMSlicer steps and TopoStats in that the left and right hand edges and the middle vertical section are higher after TopoStats steps as with the `viridis` colourmap blue is the lowest value and yellow and the plot shows the difference of subtracting the `TopoStats` values from `AFMSlicer` gives at each pixel.
    """)
    return


@app.cell
def _(case1_filtered, case1_topostats_filtered, compare_arrays):
    case1_diff_slicer_topostats = compare_arrays(
        arr1=case1_filtered.images["initial_nonlinear_polynomial_removal"],
        arr2=case1_topostats_filtered.images["initial_scar_removal"],
        arr1_name="AFMSlicer",
        arr2_name="TopoStats",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparison to Gwyddion Image

    Ideally we would compare this image to that processed with Gwyddion.

    I have taken the `.spm` files and loaded them into Gwyddion (v2.70 2025-12-28) and followed the instructions in the [slides](https://docs.google.com/presentation/d/1bOMfPmaRMs5TPFFGrxXpamf-GJswlHd4/edit) using the same settings and additionally applying a Gaussian blur and saving them as `.npy` Numpy arrays which we can then load and compare.
    """)
    return


@app.cell
def _(NPY_DIR, np, plot_array):
    case1_gwy = np.load(file=NPY_DIR / "case1.npy")
    plot_array(array=case1_gwy, cmap="gray")
    return (case1_gwy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### AFMSlicer - Gwyddion

    Here we compare the difference between the AFMSlicer steps performed in this notebook with the Gwyddion processed image.
    """)
    return


@app.cell
def _(case1_filtered, case1_gwy, compare_arrays):
    diff_afmslicer_gwy = compare_arrays(
        arr1=case1_filtered.images["gaussian_filtered"],
        arr2=case1_gwy,
        arr1_name="AFMSlicer",
        arr2_name="Gwyddion",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### TopoStats - Gwyddion

    Here we compare the difference between the TopoStats steps performed in this notebook with the Gwyddion processed image.
    """)
    return


@app.cell
def _(case1_gwy, case1_topostats_filtered, compare_arrays):
    diff_topostats_gwy = compare_arrays(
        arr1=case1_topostats_filtered.images["gaussian_filtered"],
        arr2=case1_gwy,
        arr1_name="AFMSlicer",
        arr2_name="Gwyddion",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Case 2 - External Cell Wall (Diificult to Flatten)
    """)
    return


@app.cell
def _(LoadScans, Path, config):
    case2_scans = LoadScans(
        [Path("/home/neil/work/git/hub/ns-rse/AFMSlicer/notebooks/data/spm/case2.spm")],
        config=config,
    )
    case2_scans.get_data()
    case2 = case2_scans.img_dict["case2"]
    return (case2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Original Image

    This is the image as loaded, plotted using the `grayscale` colour scale.
    """)
    return


@app.cell
def _(case2, plot_array):
    plot_array(array=case2.image_original, cmap="gray")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### AFMSlicer Flattening

    Process with the four AFMSlicer steps
    """)
    return


@app.cell
def _(afmslicer_filter, case2, filters_config):
    case2_filtered = afmslicer_filter(
        topostats_object=case2, filter_config=filters_config
    )
    return (case2_filtered,)


@app.cell
def _(case2_filtered, plot_array):
    plot_array(case2_filtered.images["gaussian_filtered"], cmap="gray")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### TopoStats Flattening

    Process with the TopoStats steps
    """)
    return


@app.cell
def _(case2, filters_config, topostats_filter):
    case2_topostats_filtered = topostats_filter(
        topostats_object=case2, filter_config=filters_config
    )
    return (case2_topostats_filtered,)


@app.cell
def _(case2_topostats_filtered, plot_array):
    plot_array(case2_topostats_filtered.images["gaussian_filtered"], cmap="gray")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### AFMSlicer v TopoStats

    As with `case1` the TopoStats steps result in higher values on the left and right hand side, a consequence of the Quadratic flattening having been applied. The tilt removal has made a very small difference but only slight and the Nonlinear polynomial flattening has had less of an impact as a consequence of having included Quadratic and Tilt flattening.
    """)
    return


@app.cell
def _(case2_filtered, case2_topostats_filtered, compare_arrays):
    case2_diff_slicer_topostats = compare_arrays(
        arr1=case2_filtered.images["initial_nonlinear_polynomial_removal"],
        arr2=case2_topostats_filtered.images["initial_scar_removal"],
        arr1_name="AFMSlicer",
        arr2_name="TopoStats",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparison to Gwyddion Image

    Ideally we would compare this image to that processed with Gwyddion.

    I have taken the `.spm` files and loaded them into Gwyddion (v2.70 2025-12-28) and followed the instructions in the [slides](https://docs.google.com/presentation/d/1bOMfPmaRMs5TPFFGrxXpamf-GJswlHd4/edit) using the same settings and additionally applying a Gaussian blur and saving them as `.npy` Numpy arrays which we can then load and compare.
    """)
    return


@app.cell
def _(NPY_DIR, np, plot_array):
    case2_gwy = np.load(file=NPY_DIR / "case2.npy")
    plot_array(array=case2_gwy, cmap="gray")
    return (case2_gwy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### AFMSlicer - Gwyddion

    Here we compare the difference between the AFMSlicer steps performed in this notebook with the Gwyddion processed image.
    """)
    return


@app.cell
def _(case2_filtered, case2_gwy, compare_arrays):
    diff_case2_afmslicer_gwy = compare_arrays(
        arr1=case2_filtered.images["gaussian_filtered"],
        arr2=case2_gwy,
        arr1_name="AFMSlicer",
        arr2_name="Gwyddion",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### TopoStats - Gwyddion

    Here we compare the difference between the TopoStats steps performed in this notebook with the Gwyddion processed image.
    """)
    return


@app.cell
def _(case2_gwy, case2_topostats_filtered, compare_arrays):
    diff_case2_topostats_gwy = compare_arrays(
        arr1=case2_topostats_filtered.images["gaussian_filtered"],
        arr2=case2_gwy,
        arr1_name="AFMSlicer",
        arr2_name="Gwyddion",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Case 3 - Internal Cell Wall (Easy to Flatten)
    """)
    return


@app.cell
def _(LoadScans, Path, config):
    case3_scans = LoadScans(
        [Path("/home/neil/work/git/hub/ns-rse/AFMSlicer/notebooks/data/spm/case3.spm")],
        config=config,
    )
    case3_scans.get_data()
    case3 = case3_scans.img_dict["case3"]
    return (case3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Original Image

    This is the image as loaded, plotted using the `grayscale` colour scale.
    """)
    return


@app.cell
def _(case3, plt):
    plt.style.use(style="grayscale")
    plt.imshow(case3.image_original)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### AFMSlicer Flattening

    Process with the four AFMSlicer steps
    """)
    return


@app.cell
def _(afmslicer_filter, case3, filters_config):
    case3_filtered = afmslicer_filter(
        topostats_object=case3, filter_config=filters_config
    )
    return (case3_filtered,)


@app.cell
def _(case3_filtered, plot_array):
    plot_array(case3_filtered.images["gaussian_filtered"], cmap="gray")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### TopoStats Flattening

    Process with the TopoStats steps
    """)
    return


@app.cell
def _(case3, filters_config, topostats_filter):
    case3_topostats_filtered = topostats_filter(
        topostats_object=case3, filter_config=filters_config
    )
    return (case3_topostats_filtered,)


@app.cell
def _(case3_topostats_filtered, plot_array):
    plot_array(case3_topostats_filtered.images["gaussian_filtered"], cmap="gray")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### AFMSlicer v TopoStats
    """)
    return


@app.cell
def _(case3_filtered, case3_topostats_filtered, compare_arrays):
    case3_diff_slicer_topostats = compare_arrays(
        arr1=case3_filtered.images["initial_nonlinear_polynomial_removal"],
        arr2=case3_topostats_filtered.images["initial_scar_removal"],
        arr1_name="AFMSlicer",
        arr2_name="TopoStats",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparison to Gwyddion Image

    Ideally we would compare this image to that processed with Gwyddion.

    I have taken the `.spm` files and loaded them into Gwyddion (v2.70 2025-12-28) and followed the instructions in the [slides](https://docs.google.com/presentation/d/1bOMfPmaRMs5TPFFGrxXpamf-GJswlHd4/edit) using the same settings and additionally applying a Gaussian blur and saving them as `.npy` Numpy arrays which we can then load and compare.
    """)
    return


@app.cell
def _(NPY_DIR, np, plot_array):
    case3_gwy = np.load(file=NPY_DIR / "case3.npy")
    plot_array(array=case3_gwy, cmap="gray")
    return (case3_gwy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### AFMSlicer - Gwyddion

    Here we compare the difference between the AFMSlicer steps performed in this notebook with the Gwyddion processed image.
    """)
    return


@app.cell
def _(case3_filtered, case3_gwy, compare_arrays):
    diff_case3_topostats_gwy = compare_arrays(
        arr1=case3_filtered.images["gaussian_filtered"],
        arr2=case3_gwy,
        arr1_name="AFMSlicer",
        arr2_name="Gwyddion",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### TopoStats - Gwyddion

    Here we compare the difference between the TopoStats steps performed in this notebook with the Gwyddion processed image.
    """)
    return


@app.cell
def _(case3_gwy, case3_topostats_filtered, compare_arrays):
    diff_case3_afmslicer_gwy = compare_arrays(
        arr1=case3_topostats_filtered.images["gaussian_filtered"],
        arr2=case3_gwy,
        arr1_name="AFMSlicer",
        arr2_name="Gwyddion",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Case 4 - Internal Cell Wall (Difficult to Flatten)
    """)
    return


@app.cell
def _(LoadScans, Path, config):
    case4_scans = LoadScans(
        [Path("/home/neil/work/git/hub/ns-rse/AFMSlicer/notebooks/data/spm/case4.spm")],
        config=config,
    )
    case4_scans.get_data()
    case4 = case4_scans.img_dict["case4"]
    return (case4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Original Image

    This is the image as loaded, plotted using the `grayscale` colour scale.
    """)
    return


@app.cell
def _(case4, plt):
    plt.style.use(style="grayscale")
    plt.imshow(case4.image_original)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### AFMSlicer Flattening

    Process with the four AFMSlicer steps
    """)
    return


@app.cell
def _(afmslicer_filter, case4, filters_config):
    case4_filtered = afmslicer_filter(
        topostats_object=case4, filter_config=filters_config
    )
    return (case4_filtered,)


@app.cell
def _(case4_filtered, plot_array):
    plot_array(case4_filtered.images["gaussian_filtered"], cmap="gray")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### TopoStats Flattening

    Process with the TopoStats steps
    """)
    return


@app.cell
def _(case4, filters_config, topostats_filter):
    case4_topostats_filtered = topostats_filter(
        topostats_object=case4, filter_config=filters_config
    )
    return (case4_topostats_filtered,)


@app.cell
def _(case4_topostats_filtered, plot_array):
    plot_array(case4_topostats_filtered.images["gaussian_filtered"], cmap="gray")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### AFMSlicer v TopoStats
    """)
    return


@app.cell
def _(case4_filtered, case4_topostats_filtered, compare_arrays):
    case4_diff_slicer_topostats = compare_arrays(
        arr1=case4_filtered.images["initial_nonlinear_polynomial_removal"],
        arr2=case4_topostats_filtered.images["initial_scar_removal"],
        arr1_name="AFMSlicer",
        arr2_name="TopoStats",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Comparison to Gwyddion Image

    Ideally we would compare this image to that processed with Gwyddion.

    I have taken the `.spm` files and loaded them into Gwyddion (v2.70 2025-12-28) and followed the instructions in the [slides](https://docs.google.com/presentation/d/1bOMfPmaRMs5TPFFGrxXpamf-GJswlHd4/edit) using the same settings and additionally applying a Gaussian blur and saving them as `.npy` Numpy arrays which we can then load and compare.
    """)
    return


@app.cell
def _(NPY_DIR, np, plot_array):
    case4_gwy = np.load(file=NPY_DIR / "case4.npy")
    plot_array(array=case4_gwy, cmap="gray")
    return (case4_gwy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### AFMSlicer - Gwyddion

    Here we compare the difference between the AFMSlicer steps performed in this notebook with the Gwyddion processed image.
    """)
    return


@app.cell
def _(case4_filtered, case4_gwy, compare_arrays):
    diff_case4_afmslicer_gwy = compare_arrays(
        arr1=case4_filtered.images["gaussian_filtered"],
        arr2=case4_gwy,
        arr1_name="AFMSlicer",
        arr2_name="Gwyddion",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### TopoStats - Gwyddion

    Here we compare the difference between the TopoStats steps performed in this notebook with the Gwyddion processed image.
    """)
    return


@app.cell
def _(case4_gwy, case4_topostats_filtered, compare_arrays):
    diff_case4_topostats_gwy = compare_arrays(
        arr1=case4_topostats_filtered.images["gaussian_filtered"],
        arr2=case4_gwy,
        arr1_name="AFMSlicer",
        arr2_name="Gwyddion",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Summary

    Comparing all the images we have processed via Gwyddion, equivalent steps using Python (`AFMSlicer` row) and the default filtering performed by TopoStats (`TopoStats` row) we have the following plots shown below.

    There isn't a huge amount of difference between any of these. AFMSlicer and TopoStats pre-processing give very slightly different results to each other. There is however some difference between Neil Shephard's processing with Gwyddion and both AFMSlicer/TopoStats cleaning and Gwyddion tends to result in slightly higher data points, although this is not uniform, there are plenty of points which are lower.
    """)
    return


@app.cell
def _(
    case1,
    case1_filtered,
    case1_gwy,
    case1_topostats_filtered,
    case2,
    case2_filtered,
    case2_gwy,
    case2_topostats_filtered,
    plot_array_with_colorbar_title,
    plt,
):
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(8, 12))
    plt.style.use("grayscale")
    # Row 1 - Originals
    ax1_1 = plt.subplot(4, 2, 1)
    fig, ax1_1 = plot_array_with_colorbar_title(
        array=case1.image_original, fig=fig, ax=ax1_1, cmap="gray", shrink=0.70
    )
    ax2_1 = plt.subplot(4, 2, 2)
    fig, ax2_1 = plot_array_with_colorbar_title(
        array=case2.image_original, fig=fig, ax=ax2_1, cmap="gray", shrink=0.70
    )

    # Row 2 - Gwyddion
    ax3_1 = plt.subplot(4, 2, 3)
    fig, ax3_1 = plot_array_with_colorbar_title(
        array=case1_gwy, fig=fig, ax=ax3_1, cmap="gray", shrink=0.70
    )
    ax4_1 = plt.subplot(4, 2, 4)
    fig, ax4_1 = plot_array_with_colorbar_title(
        array=case2_gwy, fig=fig, ax=ax4_1, cmap="gray", shrink=0.70
    )

    # Row 3 - AFMSlicer
    ax5_1 = plt.subplot(4, 2, 5)
    fig, ax5_1 = plot_array_with_colorbar_title(
        array=case1_filtered.images["gaussian_filtered"],
        fig=fig,
        ax=ax5_1,
        cmap="gray",
        shrink=0.30,
    )
    ax6_1 = plt.subplot(4, 2, 6)
    fig, ax6_1 = plot_array_with_colorbar_title(
        array=case2_filtered.images["gaussian_filtered"],
        fig=fig,
        ax=ax6_1,
        cmap="gray",
        shrink=0.30,
    )

    # Row 4 -TopoStats
    ax7_1 = plt.subplot(4, 2, 7)
    fig, ax7_1 = plot_array_with_colorbar_title(
        array=case1_topostats_filtered.images["gaussian_filtered"],
        fig=fig,
        ax=ax7_1,
        cmap="gray",
        shrink=0.30,
    )
    ax8_1 = plt.subplot(4, 2, 8)
    fig, ax8_1 = plot_array_with_colorbar_title(
        array=case2_topostats_filtered.images["gaussian_filtered"],
        fig=fig,
        ax=ax8_1,
        cmap="gray",
        shrink=0.30,
    )

    cols = [
        f"Case 1\n{case1.image_original.mean():.3f} ({case1.image_original.std():.3f})",
        f"Case 2\n{case1.image_original.mean():.3f} ({case1.image_original.std():.3f})",
    ]
    rows = ["Original", "Gwyddion", "AFMSlicer", "TopoStats"]
    for ax, col in zip(axes[0], cols, strict=False):
        ax.set_title(col)
    for ax, row in zip(axes[:, 0], rows, strict=False):
        ax.set_ylabel(row, rotation=90, size="medium")

    fig.tight_layout()
    plt.show()
    return


@app.cell
def _(
    case3,
    case3_filtered,
    case3_gwy,
    case3_topostats_filtered,
    case4,
    case4_filtered,
    case4_gwy,
    case4_topostats_filtered,
    plot_array_with_colorbar_title,
    plt,
):
    fig2, axes2 = plt.subplots(nrows=4, ncols=2, figsize=(8, 12))
    plt.style.use("grayscale")
    # Row 1 - Originals
    ax1 = plt.subplot(4, 2, 1)
    fig2, ax1 = plot_array_with_colorbar_title(
        array=case3.image_original, fig=fig2, ax=ax1, cmap="gray", shrink=0.70
    )
    ax2 = plt.subplot(4, 2, 2)
    fig2, ax2 = plot_array_with_colorbar_title(
        array=case4.image_original, fig=fig2, ax=ax2, cmap="gray", shrink=0.70
    )

    # Row 2 - Gwyddion
    ax3 = plt.subplot(4, 2, 3)
    fig2, ax3 = plot_array_with_colorbar_title(
        array=case3_gwy, fig=fig2, ax=ax3, cmap="gray", shrink=0.70
    )
    ax4 = plt.subplot(4, 2, 4)
    fig2, ax4 = plot_array_with_colorbar_title(
        array=case4_gwy, fig=fig2, ax=ax4, cmap="gray", shrink=0.70
    )

    # Row 3 - AFMSlicer (Python)
    ax5 = plt.subplot(4, 2, 5)
    fig2, ax5 = plot_array_with_colorbar_title(
        array=case3_filtered.images["gaussian_filtered"],
        fig=fig2,
        ax=ax5,
        cmap="gray",
        shrink=0.30,
    )
    ax6 = plt.subplot(4, 2, 6)
    fig2, ax6 = plot_array_with_colorbar_title(
        array=case4_filtered.images["gaussian_filtered"],
        fig=fig2,
        ax=ax6,
        cmap="gray",
        shrink=0.30,
    )

    # Row 4 - TopoStats
    ax7 = plt.subplot(4, 2, 7)
    fig2, ax7 = plot_array_with_colorbar_title(
        array=case3_topostats_filtered.images["gaussian_filtered"],
        fig=fig2,
        ax=ax7,
        cmap="gray",
        shrink=0.30,
    )
    ax8 = plt.subplot(4, 2, 8)
    fig2, ax8 = plot_array_with_colorbar_title(
        array=case4_topostats_filtered.images["gaussian_filtered"],
        fig=fig2,
        ax=ax8,
        cmap="gray",
        shrink=0.30,
    )

    cols2 = [
        f"Case 3\n{case3.image_original.mean():.3f} ({case3.image_original.std():.3f})",
        f"Case 4\n{case4.image_original.mean():.3f} ({case4.image_original.std():.3f})",
    ]
    rows2 = ["Original", "Gwyddion", "AFMSlicer", "TopoStats"]
    for ax2, col2 in zip(axes2[0], cols2, strict=False):
        ax2.set_title(col2)
    for ax2, row2 in zip(axes2[:, 0], rows2, strict=False):
        ax2.set_ylabel(row2, rotation=90, size="medium")

    fig2.tight_layout()
    plt.show()
    return


@app.cell
def _(
    OUT_DIR,
    case1,
    case1_filtered,
    case1_gwy,
    case1_topostats_filtered,
    case2,
    case2_filtered,
    case2_gwy,
    case3,
    case3_filtered,
    case3_gwy,
    case4,
    case4_filtered,
    case4_gwy,
    datetime,
    pkl,
):
    # Build a dictionary of the images
    all_images = {
        "case1": {
            "gwy": {
                "image_original": case1.image_original,
                "image": case1_gwy,
                "filename": case1.filename,
                "img_path": case1.img_path,
                "pixel_to_nm_scaling": case1.pixel_to_nm_scaling,
            },
            "afmslicer": {
                "image_original": case1.image_original,
                "image": case1_filtered.images["gaussian_filtered"],
                "filename": case1.filename,
                "img_path": case1.img_path,
                "pixel_to_nm_scaling": case1.pixel_to_nm_scaling,
            },
            "topostats": {
                "image_original": case1.image_original,
                "image": case1_topostats_filtered.images["gaussian_filtered"],
                "filename": case1.filename,
                "img_path": case1.img_path,
                "pixel_to_nm_scaling": case1.pixel_to_nm_scaling,
            },
        },
        "case2": {
            "gwy": {
                "image_original": case2.image_original,
                "image": case2_gwy,
                "filename": case2.filename,
                "img_path": case2.img_path,
                "pixel_to_nm_scaling": case2.pixel_to_nm_scaling,
            },
            "afmslicer": {
                "image_original": case2.image_original,
                "image": case2_filtered.images["gaussian_filtered"],
                "filename": case2.filename,
                "img_path": case2.img_path,
                "pixel_to_nm_scaling": case2.pixel_to_nm_scaling,
            },
            "topostats": {
                "image_original": case2.image_original,
                "image": case1_topostats_filtered.images["gaussian_filtered"],
                "filename": case2.filename,
                "img_path": case2.img_path,
                "pixel_to_nm_scaling": case2.pixel_to_nm_scaling,
            },
        },
        "case3": {
            "gwy": {
                "image_original": case3.image_original,
                "image": case3_gwy,
                "filename": case3.filename,
                "img_path": case3.img_path,
                "pixel_to_nm_scaling": case3.pixel_to_nm_scaling,
            },
            "afmslicer": {
                "image_original": case3.image_original,
                "image": case3_filtered.images["gaussian_filtered"],
                "filename": case3.filename,
                "img_path": case3.img_path,
                "pixel_to_nm_scaling": case3.pixel_to_nm_scaling,
            },
            "topostats": {
                "image_original": case3.image_original,
                "image": case1_topostats_filtered.images["gaussian_filtered"],
                "filename": case3.filename,
                "img_path": case3.img_path,
                "pixel_to_nm_scaling": case3.pixel_to_nm_scaling,
            },
        },
        "case4": {
            "gwy": {
                "image_original": case4.image_original,
                "image": case4_gwy,
                "filename": case4.filename,
                "img_path": case4.img_path,
                "pixel_to_nm_scaling": case4.pixel_to_nm_scaling,
            },
            "afmslicer": {
                "image_original": case4.image_original,
                "image": case4_filtered.images["gaussian_filtered"],
                "filename": case4.filename,
                "img_path": case4.img_path,
                "pixel_to_nm_scaling": case4.pixel_to_nm_scaling,
            },
            "topostats": {
                "image_original": case4.image_original,
                "image": case1_topostats_filtered.images["gaussian_filtered"],
                "filename": case4.filename,
                "img_path": case4.img_path,
                "pixel_to_nm_scaling": case4.pixel_to_nm_scaling,
            },
        },
    }
    # Save the images for loading which is quicker
    outfile = (
        OUT_DIR / f"flattened_images_{datetime.today().strftime('%Y-%m-%d-%H%M')}.pkl"
    )
    with outfile.open("wb") as f:
        pkl.dump(all_images, file=f)
    return (all_images,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Slicing Differences

    These comparisons are all well and good and show that there are _some_ differences between pre-processing done in Gwyddion, attempting a minimal equivalent using TopoStats functions and using the default filtering/flattening from TopoStats, but the important question is what the quantitative impact on the AFM Slicing we are seeking to replicate are and whether this impacts our qualitiative interpretation of the results (i.e. there may be small differences but if we end up making the same conclusion about the location of the layer with the maximum area should we be concerned).

    Here we step through slicing images produced by each of the filtering steps to check this impact.
    """)
    return


@app.cell
def _(AFMSlicer, all_images, config):
    # Create a nested dictionary of AFMSlicer objects, this takes a while to run as in creating AFMSlicer objects we do all the processing (sequentially rather than in parallel)
    afmsliced_images = {}
    for case, images in all_images.items():
        print(f"Processing Case : {case[-1]}")
        afmsliced_images[case] = {}
        for image_type, data in images.items():
            print(f"    Filtered using : {image_type}")
            afmsliced_images[case][image_type] = AFMSlicer(
                image=data["image"],
                image_original=data["image_original"],
                filename=data["filename"],
                pixel_to_nm_scaling=data["pixel_to_nm_scaling"],
                img_path=data["img_path"],
                slices=config["slicing"]["slices"],
                segment_method="label",
                config=config,
            )
    return


app._unparsable_cell(
    r"""
    fig_compare_slices, axes_compare_slices = plt.subplots(
        nrows=3, ncols=2, figsize=(8, 12)
    )
    plt.style.use("grayscale")
    # Row 1 - Gwyddion
    axes_compare_slices1_1 = plt.subplot(3, 2, 1)
    fig_compare_slices, axes_compare_slices1_1 = afmsliced_images["case1"][
        "gwy"
    ].fig_compare_slices_objects_per_layer
    axes_compare_slices2_1 = plt.subplot(3, 2, 2)
    fig_compare_slices, axes_compare_slices2_1 = afmsliced_images["case2"][
        "gwy"
    ].fig_compare_slices_objects_per_layer

    # Row 2 - AFMSlicer
    axes_compare_slices3_1 = plt.subplot(3, 2, 3)
    fig_compare_slices, axes_compare_slices3_1 = afmsliced_images["case1"][
        "afmslicer"
    ].fig_compare_slices_objects_per_layer
    axes_compare_slices4_1 = plt.subplot(3, 2, 4)
    fig_compare_slices, axes_compare_slices4_1 = afmsliced_images["case2"][
        "afmslicer"
    ].fig_compare_slices_objects_per_layer

    # Row 3 - TopoStats
    axes_compare_slices5_1 = plt.subplot(3, 2, 5)
    fig_compare_slices, axes_compare_slices5_1 = afmsliced_images["case1"][
        "topostats"
    ].fig_compare_slices_objects_per_layer
    axes_compare_slices6_1 = plt.subplot(3, 2, 6)
    fig_compare_slices, axes_compare_slices6_1 = afmsliced_images["case2"][
        "topostats"
    ].fig_compare_slices_objects_per_layer

    cols_compare = [
        f"Case 1\n{case1.image_original.mean():.3f} ({case1.image_original.std():.3f})",
        f"Case 2\n{case1.image_original.mean():.3f} ({case1.image_original.std():.3f})",
    ]
    rows_compare = ["Gwyddion", "AFMSlicer", "TopoStats"]
    for ax_compare, col_compare in zip(axes_compare_slices[0], cols_compare, strict=False:
        ax.set_title(col_compare)
    for ax_compare, row_compare in zip(axes_compare_slices[:, 0], rows_compare, strict=False):
        ax.set_ylabel(row_compare, rotation=90, size="medium")

    fig_compare_slices.tight_layout()
    plt.show()
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Challenges

    #### Batch Processing

    TopoStats has primarily been developed as a batch processing tool. I do not think that it would be easy to automate two steps of pre-processing that are done because both are highly contingent on the image being processed.

    1. Cropping regions - Regions of interest are going to vary considerably. It might be possible to mask some features but at the risk of losing areas of interest.
    2. Polynomial filtering - The number of degrees seems subjective and is based on a judgement call for each image being processed.

    #### Polynomial

    In the AFMSlicer/TopoStats processing step a non-linear polynomial is fitted of the form `a + b * x * y - c * x - d * y `. I don't know enough to say how this compares to the polynomials that are fitted by Gwyddion but it relates to the above point that deciding the number of degrees on a per-image basis.
    """)
    return


if __name__ == "__main__":
    app.run()

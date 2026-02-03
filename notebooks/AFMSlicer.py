from __future__ import annotations

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")

with app.setup:
    # Initialization code that runs before all other cells
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    This is a [Marimo](https://marimo.io) notebook that shows the progress made with porting AFMSlicer functionality to Python.
    """)
    return


@app.cell
def _():
    from pathlib import Path
    from pprint import pprint

    import matplotlib.pyplot as plt
    from PIL import Image
    from topostats.filters import Filters
    from topostats.io import LoadScans, read_yaml

    from afmslicer.classes import AFMSlicer
    from afmslicer.plotting import (
        plot_layer,
        plot_pores_by_layer,
    )

    NOTEBOOK_DIR = Path.cwd()
    BASE_DIR = NOTEBOOK_DIR / "../"
    RESOURCES = BASE_DIR / "tests" / "resources"
    RESOURCES_SLICER = RESOURCES / "slicer"
    RESOURCES_SPM = RESOURCES / "spm"
    default_config = read_yaml(BASE_DIR / "src/afmslicer/default_config.yaml")
    pprint(default_config)
    return (
        AFMSlicer,
        BASE_DIR,
        Filters,
        Image,
        LoadScans,
        RESOURCES_SPM,
        default_config,
        plot_layer,
        plot_pores_by_layer,
        plt,
        pprint,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We use [TopoStats](https://afm-spm.github.io/TopoStats) to search for and load the sample images that reside under the `tests/resources/spm` directory.
    """)
    return


@app.cell
def _(LoadScans, RESOURCES_SPM, default_config):
    scan_loader = LoadScans(
        img_paths=[
            RESOURCES_SPM / "sample1.spm",
            RESOURCES_SPM / "sample2.spm",
        ],
        config=default_config,
    )
    RESOURCES_SPM / "sample1.spm"
    scan_loader.get_data()
    return (scan_loader,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Sample 1

    Process `sample1.spm` with AFMSlicer. Here we view the raw image before any processing.
    """)
    return


@app.cell
def _(plt, scan_loader):
    sample1 = scan_loader.img_dict["sample1"]
    plt.imshow(sample1.image_original)
    return (sample1,)


@app.cell
def _():
    mo.md(r"""
    Before processing with AFMSlicer we use TopoStats `Filter` class to flatten the image and remove any artefacts such as scars which may be present in the image. In this instance it doesn't make much difference visually.
    """)
    return


@app.cell
def _(Filters, default_config, plt, sample1):
    sample1_filtered = Filters(
        topostats_object=sample1,
        threshold_std_dev=default_config["filter"]["threshold_std_dev"],
        threshold_method=default_config["filter"]["threshold_method"],
        threshold_absolute=default_config["filter"]["threshold_absolute"],
        otsu_threshold_multiplier=default_config["filter"]["otsu_threshold_multiplier"],
        gaussian_size=default_config["filter"]["gaussian_size"],
        gaussian_mode=default_config["filter"]["gaussian_mode"],
        row_alignment_quantile=default_config["filter"]["row_alignment_quantile"],
        remove_scars=default_config["filter"]["remove_scars"],
    )
    plt.imshow(sample1_filtered.image)
    return (sample1_filtered,)


@app.cell
def _():
    mo.md(r"""
    We now use AFMSlicer to slice this image. There are a number of functions within the `afmslicer.slicer` module that perform each step. But in order to make the workflow easier for users we define a "class" which is an "object" in Python terms. In this case it is an extensions of the `TopoStats` class and so it inherits all the attributes from its parent class, but extends it further.

    This takes a few seconds to run as it is...

    1. Finding the minimum and maximum height in the image.
    2. Working out the heights to slice at for the value `slices` defined in the `default_config` we have loaded (this value is `255`).
    3. Creating `255` slices within the image and storing in a three-dimensional numpy array.
    4. Segmenting the image using the `segment_method` defined in the `default_config` we have loaded (this value is `watershed`).
    5. Calculating an array of attributes on each region within the image.
    """)
    return


@app.cell
def _(AFMSlicer, RESOURCES_SPM, default_config, sample1, sample1_filtered):
    sample1_afmslicer = AFMSlicer(
        image=sample1_filtered.image,
        image_original=sample1.image_original,
        filename="sample1",
        img_path=RESOURCES_SPM / "sample1.spm",
        pixel_to_nm_scaling=sample1_filtered.pixel_to_nm_scaling,
        slices=default_config["slicing"]["slices"],
        segment_method="label",
        config=default_config,
    )
    return (sample1_afmslicer,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    We can plot an individual layer using the `afmslicer.plotting.plot_layer()` function which has a number of arguments including `out_dir`. If this value is anything other than `None` then it is used as the output directory (it will be created if it doesn't exist) and the file is saved there. We can also specify the `format` we want to save the image in, here we use `png`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Masked layers

    We can plot the masked layers for a given image.
    """)
    return


@app.cell
def _(BASE_DIR, plot_layer, plt, sample1_afmslicer):
    layer = 100
    plot_layer(
        array=sample1_afmslicer.sliced_segments[:, :, layer],
        outdir=BASE_DIR / "tmp" / "output",
        img_name=sample1_afmslicer.filename,
        layer=layer,
        format="png",
    )
    plt.show()
    return


@app.cell
def _():
    #### Generate GIF of moving through layers
    return


@app.cell
def _(BASE_DIR, Image, sample1_afmslicer):
    gif = []
    for _layer in range(sample1_afmslicer.sliced_segments.shape[2]):
        gif.append(Image.fromarray(sample1_afmslicer.sliced_segments[:, :, _layer]))
    gif[0].save(
        BASE_DIR / "tmp" / "output" / "sample1.gif",
        save_all=True,
        append_images=gif[1:],
        optimize=False,
        duration=200,
        loop=0,
    )
    return


@app.cell
def _():
    mo.md(r"""
    This is ok, but we want to have similar functionality to existing AFMSlicer and plot _all_ layers. To do this we could use the `afmslicer.plotting.plot_all_layers()` function. However, the `AFMSlicer` class has a `__post_init__` method which not only slices the images and detects the objects within them but also plots them using `plot_all_layers()` and has already saved the images to the output directory (`output_dir`) specified in the `default_config` (which is by default `./output`) as Portable Network Graphic (`.png`).

    Output files have the original image name `sample1` suffixed with the layer number, e.g. `sample1_100.png`.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ### Statistics

    We want to calculate statistics on each of the objects within these layers. We are using [scikit-image](scikit-image.org/docs/stable/) and its [segmentation](scikit-image.org/docs/stable/api/skimage.segmentation.html) method to process the images and each object within a given layer has `region_props` defined. This includes the area of the region.

    The `afmslicer.slicer.calculate_region_properties()` function will do this for a single layer, but as with plotting we have a wrapper function `afmslicer.slicer.region_properties_by_slices()` which performs the calculations on _all_ layers and as with plotting this is run as part of the `__post_init__` setup of the `AFMSlicer` class (which is why the above cell takes some time to run).

    We can access the results of this though and they are stored in the `AFMSlicer.sliced_region_properties` attribute of our class.
    """)
    return


@app.cell
def _(sample1_afmslicer):
    print(sample1_afmslicer.pores_per_layer)
    return


@app.cell
def _(pprint, sample1_afmslicer):
    pprint(sample1_afmslicer.area_by_layer)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    More usefully though we want to plot the number of objects or "pores" found on each layer. To do this we can use the `plotting.plot_pores_by_layer()` function to plot the `sample1_afmslicer.pores_per_layer`
    """)
    return


@app.cell
def _(plot_pores_by_layer, sample1_afmslicer):
    plot_pores_by_layer(pores_per_layer=sample1_afmslicer.pores_per_layer)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    However, as with the statistics themselves these plots are automatically generated from the underlying data and are attributes of the `AFMSlicer` object. There are two forms, one with the layer v's the number of pores the other plotting the layer v's the logarithm (base10) of the number of pores. Both of these images are saved to the `outdir` directory automatically as Portable Network Graphics (`.png`).
    """)
    return


@app.cell
def _(sample1_afmslicer):
    sample1_afmslicer.fig_objects_per_layer
    return


@app.cell
def _(sample1_afmslicer):
    sample1_afmslicer.fig_log_objects_per_layer
    return


@app.cell
def _(sample1_afmslicer):
    sample1_afmslicer.fig_area_per_layer
    return


@app.cell
def _(sample1_afmslicer):
    sample1_afmslicer.fig_log_area_per_layer
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## `sample2.spm`

    Repeat the process for `sample2.spm`
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Original Image
    """)
    return


@app.cell
def _(plt, scan_loader):
    sample2 = scan_loader.img_dict["sample2"]
    plt.imshow(sample2.image_original)
    return (sample2,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Flattenend Image
    """)
    return


@app.cell
def _(Filters, default_config, plt, sample2):
    sample2_filtered = Filters(
        topostats_object=sample2,
        threshold_std_dev=default_config["filter"]["threshold_std_dev"],
        threshold_method=default_config["filter"]["threshold_method"],
        threshold_absolute=default_config["filter"]["threshold_absolute"],
        otsu_threshold_multiplier=default_config["filter"]["otsu_threshold_multiplier"],
        gaussian_size=default_config["filter"]["gaussian_size"],
        gaussian_mode=default_config["filter"]["gaussian_mode"],
        row_alignment_quantile=default_config["filter"]["row_alignment_quantile"],
        remove_scars=default_config["filter"]["remove_scars"],
    )
    plt.imshow(sample2_filtered.image)
    return (sample2_filtered,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Process with AFMSlicer
    """)
    return


@app.cell
def _(AFMSlicer, RESOURCES_SPM, default_config, sample2, sample2_filtered):
    sample2_afmslicer = AFMSlicer(
        image=sample2_filtered.image,
        image_original=sample2.image_original,
        filename="sample2",
        img_path=RESOURCES_SPM / "sample2.spm",
        pixel_to_nm_scaling=sample2_filtered.pixel_to_nm_scaling,
        slices=default_config["slicing"]["slices"],
        segment_method="label",
        config=default_config,
    )
    return (sample2_afmslicer,)


@app.cell
def _(BASE_DIR, plot_layer, sample2_afmslicer):
    layer2 = 50
    plot_layer(
        array=sample2_afmslicer.sliced_segments[:, :, layer2],
        outdir=BASE_DIR / "tmp" / "output",
        img_name=sample2_afmslicer.filename,
        layer=layer2,
        format="png",
    )
    return


@app.cell
def _():
    mo.md(r"""
    #### Generate GIF of moving through layers
    """)
    return


@app.cell
def _(BASE_DIR, Image, gif2, sample1_afmslicer):
    for _layer in range(sample1_afmslicer.sliced_segments.shape[2]):
        gif2.append(Image.fromarray(sample1_afmslicer.sliced_segments[:, :, _layer]))
    gif2[0].save(
        BASE_DIR / "tmp" / "output" / "sample1.gif",
        save_all=True,
        append_images=gif2[1:],
        optimize=False,
        duration=200,
        loop=0,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Objects Detected
    """)
    return


@app.cell
def _(sample2_afmslicer):
    sample2_afmslicer.fig_objects_per_layer
    return


@app.cell
def _(sample2_afmslicer):
    sample2_afmslicer.fig_log_area_per_layer
    return


if __name__ == "__main__":
    app.run()

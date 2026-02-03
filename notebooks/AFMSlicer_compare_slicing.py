from __future__ import annotations

import marimo

__generated_with = "0.19.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # Compare Slicing

    We have seen in `AFMSlicer_flattening.py` notebook how the flattening process compare but it is important to see how the different methods compare with the slicing methods.

    We load the files from this step and slice the flattened images. We can do this directly using the `afmslicer.classes.AFMSlicer` class as that
    """)


@app.cell
def _(mo):
    mo.md(r"""
    ## Setup

    Load libraries, setup paths and load the data.
    """)


@app.cell
def _():
    import pickle as pkl
    from datetime import datetime
    from pathlib import Path

    from topostats.io import read_yaml

    from afmslicer.classes import AFMSlicer

    BASE_DIR = Path().cwd()
    PKG_BASE_DIR = BASE_DIR.parent
    PKG_SRC_DIR = PKG_BASE_DIR / "src" / "afmslicer"
    DATA_DIR = BASE_DIR / "data"
    SPM_DIR = DATA_DIR / "spm"
    NPY_DIR = DATA_DIR / "npy"
    OUT_DIR = BASE_DIR / "output" / "flattening"
    RESULTS_DIR = OUT_DIR / f"{datetime.today().strftime('%Y-%m-%d-%H%M')}"
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    config = read_yaml(PKG_SRC_DIR / "default_config.yaml")
    config["slicing"]["slices"] = 40
    # Load the images
    flattened_images = OUT_DIR / "flattened.pkl"
    flattened_images = OUT_DIR / "flattened_images_2026-01-30-0929.pkl"
    # flattened_images = Path("/home/neil/work/git/hub/ns-rse/AFMSlicer/notebooks/output/flattened_images_2026-01-30-0929.pkl")
    with flattened_images.open("rb") as f:
        all_images = pkl.load(f)
    return AFMSlicer, all_images, config


@app.cell
def _(mo):
    mo.md(r"""
    ## Case 1
    """)


@app.cell
def _(AFMSlicer, all_images, config):
    config_case1_gwy = config.copy()
    # config["output_dir"] = f"output/{datetime.today().strftime('%Y-%m-%d-%H')}/case1/gwy"
    config["output_dir"] = "output/case1/gwy"
    case1_gwy = AFMSlicer(
        image=all_images["case1"]["gwy"]["image"],
        image_original=all_images["case1"]["gwy"]["image_original"],
        filename=all_images["case1"]["gwy"]["filename"],
        pixel_to_nm_scaling=all_images["case1"]["gwy"]["pixel_to_nm_scaling"],
        img_path=all_images["case1"]["gwy"]["img_path"],
        slices=config["slicing"]["slices"],
        segment_method="label",
        config=config_case1_gwy,
    )
    case1_gwy.fig_objects_per_layer


@app.cell
def _(AFMSlicer, all_images, config):
    config_case1_afmslicer = config.copy()
    # config["output_dir"] = f"output/{datetime.today().strftime('%Y-%m-%d-%H')}/case1/afmslicer"
    config["output_dir"] = "output/case1/afmslicer"
    case1_afmslicer = AFMSlicer(
        image=all_images["case1"]["afmslicer"]["image"],
        image_original=all_images["case1"]["afmslicer"]["image_original"],
        filename=all_images["case1"]["afmslicer"]["filename"],
        pixel_to_nm_scaling=all_images["case1"]["afmslicer"]["pixel_to_nm_scaling"],
        img_path=all_images["case1"]["afmslicer"]["img_path"],
        slices=config["slicing"]["slices"],
        segment_method="label",
        config=config_case1_afmslicer,
    )
    case1_afmslicer.fig_objects_per_layer


@app.cell
def _(AFMSlicer, all_images, config):
    config_case1_topostats = config.copy()
    # config["output_dir"] = f"output/{datetime.today().strftime('%Y-%m-%d-%H')}/case1/topostats"
    config["output_dir"] = "output/case1/topostats"
    case1_topostats = AFMSlicer(
        image=all_images["case1"]["topostats"]["image"],
        image_original=all_images["case1"]["topostats"]["image_original"],
        filename=all_images["case1"]["topostats"]["filename"],
        pixel_to_nm_scaling=all_images["case1"]["topostats"]["pixel_to_nm_scaling"],
        img_path=all_images["case1"]["topostats"]["img_path"],
        slices=config["slicing"]["slices"],
        segment_method="label",
        config=config_case1_topostats,
    )
    case1_topostats.fig_objects_per_layer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Case 2
    """)


@app.cell
def _(AFMSlicer, all_images, config):
    config_case2_gwy = config.copy()
    # config["output_dir"] = f"output/{datetime.today().strftime('%Y-%m-%d-%H')}/case2/gwy"
    config["output_dir"] = "output/case2/gwy"
    case2_gwy = AFMSlicer(
        image=all_images["case2"]["gwy"]["image"],
        image_original=all_images["case2"]["gwy"]["image_original"],
        filename=all_images["case2"]["gwy"]["filename"],
        pixel_to_nm_scaling=all_images["case2"]["gwy"]["pixel_to_nm_scaling"],
        img_path=all_images["case2"]["gwy"]["img_path"],
        slices=config["slicing"]["slices"],
        segment_method="label",
        config=config_case2_gwy,
    )
    case2_gwy.fig_objects_per_layer


@app.cell
def _(AFMSlicer, all_images, config):
    config_case2_afmslicer = config.copy()
    # config["output_dir"] = f"output/{datetime.today().strftime('%Y-%m-%d-%H')}/case2/afmslicer"
    config["output_dir"] = "output/case2/afmslicer"
    case2_afmslicer = AFMSlicer(
        image=all_images["case2"]["afmslicer"]["image"],
        image_original=all_images["case2"]["afmslicer"]["image_original"],
        filename=all_images["case2"]["afmslicer"]["filename"],
        pixel_to_nm_scaling=all_images["case2"]["afmslicer"]["pixel_to_nm_scaling"],
        img_path=all_images["case2"]["topostats"]["img_path"],
        slices=config["slicing"]["slices"],
        segment_method="label",
        config=config_case2_afmslicer,
    )
    case2_afmslicer.fig_objects_per_layer


@app.cell
def _(AFMSlicer, all_images, config):
    config_case2_topostats = config.copy()
    # config["output_dir"] = f"output/{datetime.today().strftime('%Y-%m-%d-%H')}/case2/topostats"
    config["output_dir"] = "output/case2/topostats"
    case2_topostats = AFMSlicer(
        image=all_images["case2"]["topostats"]["image"],
        image_original=all_images["case2"]["topostats"]["image_original"],
        filename=all_images["case2"]["topostats"]["filename"],
        pixel_to_nm_scaling=all_images["case2"]["topostats"]["pixel_to_nm_scaling"],
        img_path=all_images["case2"]["topostats"]["img_path"],
        slices=config["slicing"]["slices"],
        segment_method="label",
        config=config_case2_topostats,
    )
    case2_topostats.fig_objects_per_layer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Case 3
    """)


@app.cell
def _(AFMSlicer, all_images, config):
    config_case3_gwy = config.copy()
    # config["output_dir"] = f"output/{datetime.today().strftime('%Y-%m-%d-%H')}/case3/gwy"
    config["output_dir"] = "output/case3/gwy"
    case3_gwy = AFMSlicer(
        image=all_images["case3"]["gwy"]["image"],
        image_original=all_images["case3"]["gwy"]["image_original"],
        filename=all_images["case3"]["gwy"]["filename"],
        pixel_to_nm_scaling=all_images["case3"]["gwy"]["pixel_to_nm_scaling"],
        img_path=all_images["case3"]["gwy"]["img_path"],
        slices=config["slicing"]["slices"],
        segment_method="label",
        config=config_case3_gwy,
    )
    case3_gwy.fig_objects_per_layer


@app.cell
def _(AFMSlicer, all_images, config):
    config_case3_afmslicer = config.copy()
    # config["output_dir"] = f"output/{datetime.today().strftime('%Y-%m-%d-%H')}/case3/afmslicer"
    config["output_dir"] = "output/case3/afmslicer"
    case3_afmslicer = AFMSlicer(
        image=all_images["case3"]["afmslicer"]["image"],
        image_original=all_images["case3"]["afmslicer"]["image_original"],
        filename=all_images["case3"]["afmslicer"]["filename"],
        pixel_to_nm_scaling=all_images["case3"]["afmslicer"]["pixel_to_nm_scaling"],
        img_path=all_images["case3"]["topostats"]["img_path"],
        slices=config["slicing"]["slices"],
        segment_method="label",
        config=config_case3_afmslicer,
    )
    case3_afmslicer.fig_objects_per_layer


@app.cell
def _(AFMSlicer, all_images, config):
    config_case3_topostats = config.copy()
    # config["output_dir"] = f"output/{datetime.today().strftime('%Y-%m-%d-%H')/case3/topostats"
    config["output_dir"] = "output/case3/topostats"
    case3_topostats = AFMSlicer(
        image=all_images["case3"]["topostats"]["image"],
        image_original=all_images["case3"]["topostats"]["image_original"],
        filename=all_images["case3"]["topostats"]["filename"],
        pixel_to_nm_scaling=all_images["case3"]["topostats"]["pixel_to_nm_scaling"],
        img_path=all_images["case3"]["topostats"]["img_path"],
        slices=config["slicing"]["slices"],
        segment_method="label",
        config=config_case3_topostats,
    )
    case3_topostats.fig_objects_per_layer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Case 4
    """)


@app.cell
def _(AFMSlicer, all_images, config):
    config_case4_gwy = config.copy()
    # config["output_dir"] = f"output/{datetime.today().strftime('%Y-%m-%d-%H')/case4/gwy"
    config["output_dir"] = "output/case4/gwy"
    case4_gwy = AFMSlicer(
        image=all_images["case4"]["gwy"]["image"],
        image_original=all_images["case4"]["gwy"]["image_original"],
        filename=all_images["case4"]["gwy"]["filename"],
        pixel_to_nm_scaling=all_images["case4"]["gwy"]["pixel_to_nm_scaling"],
        img_path=all_images["case4"]["gwy"]["img_path"],
        slices=config["slicing"]["slices"],
        segment_method="label",
        config=config_case4_gwy,
    )
    case4_gwy.fig_objects_per_layer


@app.cell
def _(AFMSlicer, all_images, config):
    config_case4_afmslicer = config.copy()
    # config["output_dir"] = f"output/{datetime.today().strftime('%Y-%m-%d-%H')/case4/afmslicer"
    config["output_dir"] = "output/case4/afmslicer"
    case4_afmslicer = AFMSlicer(
        image=all_images["case4"]["afmslicer"]["image"],
        image_original=all_images["case4"]["afmslicer"]["image_original"],
        filename=all_images["case4"]["afmslicer"]["filename"],
        pixel_to_nm_scaling=all_images["case4"]["afmslicer"]["pixel_to_nm_scaling"],
        img_path=all_images["case4"]["topostats"]["img_path"],
        slices=config["slicing"]["slices"],
        segment_method="label",
        config=config_case4_afmslicer,
    )
    case4_afmslicer.fig_objects_per_layer


@app.cell
def _(AFMSlicer, all_images, config):
    config_case4_topostats = config.copy()
    # config["output_dir"] = f"output/{datetime.today().strftime('%Y-%m-%d-%H')/case4/topostats"
    config["output_dir"] = "output/case4/topostats"
    case4_topostats = AFMSlicer(
        image=all_images["case4"]["topostats"]["image"],
        image_original=all_images["case4"]["topostats"]["image_original"],
        filename=all_images["case4"]["topostats"]["filename"],
        pixel_to_nm_scaling=all_images["case4"]["topostats"]["pixel_to_nm_scaling"],
        img_path=all_images["case4"]["topostats"]["img_path"],
        slices=config["slicing"]["slices"],
        segment_method="label",
        config=config_case4_topostats,
    )
    case4_topostats.fig_objects_per_layer


@app.cell
def _(mo):
    mo.md(r"""
    ## All in one
    """)


@app.cell
def _(AFMSlicer, all_images, config):
    # Currently this loop fails as `case1` `afmslicer` takes ~13minutes then crashes :-/
    # Create a nested dictionary of AFMSlicer objects, this takes a while to run as in creating AFMSlicer objects we do all the processing (sequentially rather than in parallel)
    afmsliced_images = {}
    for case, images in all_images.items():
        print(f"Processing Case : {case[-1]}")
        afmsliced_images[case] = {}
        for image_type, data in images.items():
            print(f"    Filtered using : {image_type}")
            config["output"] = f"output/{case}/{image_type}"
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


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

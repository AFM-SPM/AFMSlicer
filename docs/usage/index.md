# Usage

There are two different ways of using AFMSlicer to process images, the Command Line Interface (CLI) which allows for
batch processing of multiple images in parallel or a Graphical User Interface (GUI) built in Napari. This page describes
how to use the command line version of AFMSlicer, for details of using the Napari GUI see the [Napari](napari.md) page.

## Data

You should place your raw AFM images as exported from the scanning machine in a directory to be processed where the
Virtual Environment will be activated. If you have installed AFMSlicer from GitHub this will be the `AFMSlicer` you have
cloned the repository to. If you have installed from PyPI this will be whatever directory you created to setup a Virtual
Environment in.
<!-- markdownlint-disable MD046 -->
!!! note

    This isn't actually essential as you can either use the `afmslicer --base-dir` flag to specify an alternative
    directory to scan for images or modify ~base_dir: ./~ option in a [custom configuration][configuration.md].
<!-- markdownlint-enable MD046 -->

To keep data organised it is recommended to place it in the `data/raw` directory, optionally data from specific days or
experiments may be grouped into sub-directories within these. AFMSlicer will automatically find all files with the user
specified extension (which defaults to `.spm`).

<!-- markdownlint-disable MD046 -->
``` shell
.
└──   ./data
    └──   ./data/raw
        ├──   ./data/raw/e_coli
        │   ├──   ./data/raw/e_coli/20260303
        │   ├──   ./data/raw/e_coli/20260304
        │   └──   ./data/raw/e_coli/20260306
        └──   ./data/raw/staph
            ├──   ./data/raw/staph/20260320
            ├──   ./data/raw/staph/20260321
            ├──   ./data/raw/staph/20260323
            └──   ./data/raw/staph/20260326
```
<!-- markdownlint-enable MD046 -->

<!-- markdownlint-disable MD046 -->
!!! warning

    If you edit your `.spm` file(s) in other software (e.g. Gwyddion) and then save them they may not load successfully
    with AFMSlicer.
<!-- markdownlint-enable MD046 -->

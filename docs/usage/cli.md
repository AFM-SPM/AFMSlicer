# AFMSlicer at the Command Line

Once you have [installed](../installation.md) AFMSlicer within a virtual environment you can start using the Command Line
Interface. The command is called `afmslicer` and it has a useful `--help` flag which lists the available options.

``` shell
afmslicer --help
usage: afmslicer [-h] [-v] [-c CONFIG_FILE] [-b BASE_DIR] [-o OUTPUT_DIR] [-l LOG_LEVEL] [-j CORES] [-f FILE_EXT] [--channel CHANNEL] {process,filter,slicer,create-config} ...

Run various processing steps for slicing AFM images of bacterial cell walls.

options:
  -h, --help            show this help message and exit
  -v, --version         Report the current version of AFMSlicer that is installed
  -c CONFIG_FILE, --config-file CONFIG_FILE
                        Path to a YAML configuration file.
  -b BASE_DIR, --base-dir BASE_DIR
                        Base directory to scan for images.
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output directory to write results to.
  -l LOG_LEVEL, --log-level LOG_LEVEL
                        Logging level to use, default is 'info' for verbose output use 'debug'.
  -j CORES, --cores CORES
                        Number of CPU cores to use when processing.
  -f FILE_EXT, --file-ext FILE_EXT
                        File extension to scan for.
  --channel CHANNEL     Channel to extract.

program:
  Available processing options are:

  {process,filter,slicer,create-config}
    process             Process all images, slicing and summarising statistics across slices and plotting results.
    filter              Load and filter images, saving as .afmslicer files for subsequent processing.
    slicer              Load and slice images, saving as '.topostats' files for subsequent processing.
    create-config       Create a configuration file using the defaults.
```

Command line options should be self-explanatory.

## Processing

The most common step is to batch process each image from end-to-end. This involves...

- identifying all files with the given extension (default is `.spm`).
- loading each file and extracting the `channel` (default is `height`).
- filtering the image which involves flattening and removing tilt and optionally removing scars.
- slicing the image.
- segmenting the image to identify cell structures.
- calculating statistics on each segment and generating images.
- aggregating statistics and writing output to disk.

AFMSlicer comes with a default configuration file that will be loaded, to use these defaults to process images in the
`data/` directory run

``` shell
afmslicer --base-dir data/ process
```

You should see output similar to the following.

``` shell
[Thu, 26 Mar 2026 11:14:21] [INFO    ] [topostats] The YAML configuration file is valid.
2026-03-26 11:14:21.463 | INFO     | afmslicer.run_modules:_log_setup:61 - Configuration file loaded from      : None
2026-03-26 11:14:21.463 | INFO     | afmslicer.run_modules:_log_setup:62 - Scanning for images in              : tests
2026-03-26 11:14:21.463 | INFO     | afmslicer.run_modules:_log_setup:63 - Output directory                    : output
2026-03-26 11:14:21.463 | INFO     | afmslicer.run_modules:_log_setup:64 - Looking for images with extension   : .spm
2026-03-26 11:14:21.463 | INFO     | afmslicer.run_modules:_log_setup:65 - Images with extension .spm in tests : 2
2026-03-26 11:14:21.463 | INFO     | afmslicer.run_modules:_log_setup:68 - Slices per image                    : 50
[Thu, 26 Mar 2026 11:14:21] [INFO    ] [topostats] Extracting image from tests/resources/spm/sample2.spm
2026-03-26 11:14:21.464 | INFO     | AFMReader.spm:load_spm:89 - Loading image from : tests/resources/spm/sample2.spm
2026-03-26 11:14:21.465 | INFO     | AFMReader.spm:load_spm:94 - [sample2] : Loaded image from : tests/resources/spm/sample2.spm
2026-03-26 11:14:21.499 | INFO     | AFMReader.spm:load_spm:96 - [sample2] : Extracted channel Height
2026-03-26 11:14:21.502 | INFO     | AFMReader.spm:spm_pixel_to_nm_scaling:53 - [sample2] : Pixel to nm scaling : 0.625
[Thu, 26 Mar 2026 11:14:21] [INFO    ] [topostats] Extracting image from tests/resources/spm/sample1.spm
2026-03-26 11:14:21.503 | INFO     | AFMReader.spm:load_spm:89 - Loading image from : tests/resources/spm/sample1.spm
2026-03-26 11:14:21.504 | INFO     | AFMReader.spm:load_spm:94 - [sample1] : Loaded image from : tests/resources/spm/sample1.spm
2026-03-26 11:14:21.527 | INFO     | AFMReader.spm:load_spm:96 - [sample1] : Extracted channel Height
2026-03-26 11:14:21.528 | INFO     | AFMReader.spm:spm_pixel_to_nm_scaling:53 - [sample1] : Pixel to nm scaling : 39.0625
Processing images from tests, results are under output:   0%|                      | 0/2 [00:00<?, ?i2026-03-26 11:14:21.994 | INFO     | afmslicer.processing:filter_scan:103 - [sample1] : Filtering complete 😻
2026-03-26 11:14:21.996 | INFO     | afmslicer.classes:__post_init__:158 - [sample1] : AFMSlicer object created. 🔪
2026-03-26 11:14:22.203 | INFO     | afmslicer.processing:filter_scan:103 - [sample2] : Filtering complete 😻
2026-03-26 11:14:22.204 | INFO     | afmslicer.classes:__post_init__:158 - [sample2] : AFMSlicer object created. 🔪
2026-03-26 11:14:23.182 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_0.png
2026-03-26 11:14:23.219 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_1.png
2026-03-26 11:14:23.256 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_2.png
2026-03-26 11:14:23.293 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_3.png
2026-03-26 11:14:23.330 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_4.png
2026-03-26 11:14:23.375 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_5.png
2026-03-26 11:14:23.412 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_6.png
2026-03-26 11:14:23.449 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_7.png
2026-03-26 11:14:23.492 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_8.png
2026-03-26 11:14:23.534 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_9.png
2026-03-26 11:14:23.577 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_10.png
2026-03-26 11:14:23.616 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_11.png
2026-03-26 11:14:23.654 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_12.png
2026-03-26 11:14:23.695 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_13.png
2026-03-26 11:14:23.736 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_14.png
2026-03-26 11:14:23.775 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_15.png
2026-03-26 11:14:23.814 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_16.png
2026-03-26 11:14:23.847 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_17.png
2026-03-26 11:14:23.879 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_18.png
2026-03-26 11:14:23.912 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_19.png
2026-03-26 11:14:23.946 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_20.png
2026-03-26 11:14:23.983 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_21.png
2026-03-26 11:14:24.026 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_22.png
2026-03-26 11:14:24.063 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_23.png
2026-03-26 11:14:24.088 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_0.png
2026-03-26 11:14:24.104 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_24.png
2026-03-26 11:14:24.135 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_1.png
2026-03-26 11:14:24.145 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_25.png
2026-03-26 11:14:24.191 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_2.png
2026-03-26 11:14:24.192 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_26.png
2026-03-26 11:14:24.233 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_27.png
2026-03-26 11:14:24.242 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_3.png
2026-03-26 11:14:24.272 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_28.png
2026-03-26 11:14:24.296 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_4.png
2026-03-26 11:14:24.320 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_29.png
2026-03-26 11:14:24.345 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_5.png
2026-03-26 11:14:24.362 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_30.png
2026-03-26 11:14:24.395 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_6.png
2026-03-26 11:14:24.408 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_31.png
2026-03-26 11:14:24.442 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_7.png
2026-03-26 11:14:24.448 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_32.png
2026-03-26 11:14:24.489 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_33.png
2026-03-26 11:14:24.497 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_8.png
2026-03-26 11:14:24.528 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_34.png
2026-03-26 11:14:24.545 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_9.png
2026-03-26 11:14:24.573 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_35.png
2026-03-26 11:14:24.596 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_10.png
2026-03-26 11:14:24.614 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_36.png
2026-03-26 11:14:24.645 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_11.png
2026-03-26 11:14:24.655 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_37.png
2026-03-26 11:14:24.698 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_38.png
2026-03-26 11:14:24.702 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_12.png
2026-03-26 11:14:24.738 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_39.png
2026-03-26 11:14:24.747 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_13.png
2026-03-26 11:14:24.801 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_14.png
2026-03-26 11:14:24.849 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_15.png
2026-03-26 11:14:24.885 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_40.png
2026-03-26 11:14:24.904 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_16.png
2026-03-26 11:14:24.922 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_41.png
2026-03-26 11:14:24.956 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_17.png
2026-03-26 11:14:24.956 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_42.png
2026-03-26 11:14:24.990 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_43.png
2026-03-26 11:14:25.010 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_18.png
2026-03-26 11:14:25.030 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_44.png
2026-03-26 11:14:25.065 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_45.png
2026-03-26 11:14:25.067 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_19.png
2026-03-26 11:14:25.100 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_46.png
2026-03-26 11:14:25.119 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_20.png
2026-03-26 11:14:25.140 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_47.png
2026-03-26 11:14:25.176 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_48.png
2026-03-26 11:14:25.177 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_21.png
2026-03-26 11:14:25.215 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample1_49.png
2026-03-26 11:14:25.236 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_22.png
2026-03-26 11:14:25.284 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_23.png
2026-03-26 11:14:25.331 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_24.png
2026-03-26 11:14:25.384 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_25.png
2026-03-26 11:14:25.438 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_26.png
2026-03-26 11:14:25.485 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_27.png
2026-03-26 11:14:25.536 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_28.png
2026-03-26 11:14:25.596 | INFO     | afmslicer.plotting:generate_gif:278 - GIF saved to : output/sample1.gif
2026-03-26 11:14:25.599 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_29.png
2026-03-26 11:14:25.649 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_30.png
2026-03-26 11:14:25.674 | INFO     | afmslicer.plotting:plot_pores_by_layer:177 - Image saved to : output/sample1_pores_per_layer.png
2026-03-26 11:14:25.703 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_31.png
2026-03-26 11:14:25.747 | INFO     | afmslicer.plotting:plot_pores_by_layer:177 - Image saved to : output/sample1_pores_per_layer_log.png
2026-03-26 11:14:25.751 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_32.png
2026-03-26 11:14:25.802 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_33.png
2026-03-26 11:14:25.839 | INFO     | afmslicer.plotting:plot_area_by_layer:240 - Image saved to : output/sample1_area_per_layer.png
2026-03-26 11:14:25.857 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_34.png
2026-03-26 11:14:25.904 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_35.png
2026-03-26 11:14:25.906 | INFO     | afmslicer.plotting:plot_area_by_layer:240 - Image saved to : output/sample1_area_per_layer_log.png
2026-03-26 11:14:25.923 | INFO     | afmslicer.processing:slicer_scan:156 - [sample1] Slicing complete 😻
2026-03-26 11:14:25.955 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_36.png
2026-03-26 11:14:26.001 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_37.png
2026-03-26 11:14:26.043 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_38.png
2026-03-26 11:14:26.085 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_39.png
2026-03-26 11:14:26.135 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_40.png
2026-03-26 11:14:26.191 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_41.png
2026-03-26 11:14:26.256 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_42.png
2026-03-26 11:14:26.370 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_43.png
2026-03-26 11:14:26.414 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_44.png
2026-03-26 11:14:26.464 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_45.png
2026-03-26 11:14:26.513 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_46.png
2026-03-26 11:14:26.562 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_47.png
2026-03-26 11:14:26.603 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_48.png
2026-03-26 11:14:26.646 | INFO     | afmslicer.plotting:plot_layer:74 - Image saved to : output/sample2_49.png
2026-03-26 11:14:27.100 | INFO     | afmslicer.plotting:generate_gif:278 - GIF saved to : output/sample2.gif
Processing images from tests, results are under output:  50%|███████████           | 1/2 [00:05<00:05,  5.59s2026-03-26 11:14:27.149 | INFO     | afmslicer.run_modules:process:189 - [sample1] Processing completed.
2026-03-26 11:14:27.167 | INFO     | afmslicer.plotting:plot_pores_by_layer:177 - Image saved to : output/sample2_pores_per_layer.png
2026-03-26 11:14:27.230 | INFO     | afmslicer.plotting:plot_pores_by_layer:177 - Image saved to : output/sample2_pores_per_layer_log.png
2026-03-26 11:14:27.300 | INFO     | afmslicer.plotting:plot_area_by_layer:240 - Image saved to : output/sample2_area_per_layer.png
2026-03-26 11:14:27.353 | INFO     | afmslicer.plotting:plot_area_by_layer:240 - Image saved to : output/sample2_area_per_layer_log.png
2026-03-26 11:14:27.357 | INFO     | afmslicer.processing:slicer_scan:156 - [sample2] Slicing complete 😻
Processing images from tests, results are under output: 100%|███████████████████████| 2/2 [00:07<00:00,  3.36s2026-03-26 11:14:28.947 | INFO     | afmslicer.run_modules:process:189 - [sample2] Processing completed.
Processing images from tests, results are under output: 100%|███████████████████████| 2/2 [00:07<00:00,  3.70s
2026-03-26 11:14:29.008 | INFO     | afmslicer.run_modules:completion_message:305 -

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


   _____       _____    __    __    ______    __        __     _____     _____    __ __
  /\___/\    /\_____\  /_/\  /\_\  / ____/\  /\_\      /\_\   /\ __/\  /\_____\  /_/\__/\
 / / _ \ \  ( (  ___/  ) ) \/ ( (  ) ) __\/ ( ( (      \/_/   ) )__\/ ( (_____/  ) ) ) ) )
 \ \(_)/ /   \ \ \_   /_/ \  / \_\  \ \ \    \ \_\      /\_\ / / /     \ \__\   /_/ /_/_/
 / / _ \ \   / / /_\  \ \ \\// / /  _\ \ \   / / /__   / / / \ \ \_    / /__/_  \ \ \ \ \
( (_( )_) ) / /____/   )_) )( (_(  )____) ) ( (_____( ( (_(   ) )__/\ ( (_____\  )_) ) \ \
 \/_/ \_\/  \/_/       \_\/  \/_/  \____\/   \/_____/  \/_/   \/___\/  \/_____/  \_\/ \_\/


2026-03-26 11:14:29.008 | INFO     | afmslicer.run_modules:completion_message:309 -

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COMPLETE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  AFMSlicer Version           : 0.1
  AFMSlicer Commit            : cd24434e2.d20251114
  Base Directory              : tests
  File Extension              : .spm
  Files Found                 : 2
  Successfully Processed^1    : 2 (100.0%)
  All statistics              : output/statistics.csv
  Configuration               : output/config.yaml

  Email                       : afmslicer@sheffield.ac.uk
  Documentation               : https://afm-spm.github.io/afmslicer/
  Source Code                 : https://github.com/AFM-SPM/AFMSlicer/
  Bug Reports/Feature Request : https://github.com/AFM-SPM/AFMSlicer/issues/new/choose
  Citation File Format        : https://github.com/AFM-SPM/AFMSlicer/blob/main/CITATION.cff

  ^1 Successful processing of an image is slicing an image, calculating and plotting
     statistics. If these have been disabled the percentage will be 0.

  If you encounter bugs/issues or have feature requests please report them at the above URL
  or email us.

  If you have found AFMSlicer useful please consider citing it. A Citation File Format is
  linked above and available from the Source Code page.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

## Output

Output is saved, by default to the `output/` directory. You will find a `.png` image for each slice of each input file
as well as a `.gif` for each image. Results are saved to `.csv` files and there are two files ``.

### `all_statistics.csv`

This file has five columns

- `image` - the filename (minus extension) the row refers to.
- `layer` - the slice/layer the row refers to.
- `pore` - the numbered pore within the given layer.
- `area` - the area of the pore.
- `pore_color` - classification of the pore size.

``` shell
─────┬─────────────────────────────────────
     │ File: output/all_statistics.csv
─────┼─────────────────────────────────────
   1 │ image,layer,pore,area,pore_color
   2 │ sample1,0,0,399998474.12109375,blue
   3 │ sample1,1,0,396571350.09765625,blue
   4 │ sample1,2,0,390835571.2890625,blue
   5 │ sample1,3,0,349429321.2890625,blue
   6 │ sample1,3,1,17481994.62890625,blue
   7 │ sample1,3,2,576782.2265625,blue
   8 │ sample1,3,3,152587.890625,blue
   9 │ sample1,4,0,323004150.390625,blue
  10 │ sample1,4,1,16612243.65234375,blue
  11 | ...
```

### `color_count.csv`

This file has seven columns.

- `image` - the filename (minus extension) the row refers to.
- `layer` - the slice/layer the row refers to.
- `blue` - number of pores within sample/layer classified as `blue`.
- `green` - number of pores within sample/layer classified as `green`.
- `magenta` - number of pores within sample/layer classified as `magenta`.
- `yellow` - number of pores within sample/layer classified as `yellow`.
- `total` - number of pores within sample/layer.

``` shell
─────┬────────────────────────────────────────────
     │ File: output/color_count.csv
─────┼────────────────────────────────────────────
   1 │ image,layer,blue,green,magenta,yellow,total
   2 │ sample1,0,1,0,0,0,1
   3 │ sample1,1,1,0,0,0,1
   4 │ sample1,2,1,0,0,0,1
   5 │ sample1,3,4,0,0,0,4
   6 │ sample1,4,7,0,0,0,7
   7 │ sample1,5,6,0,0,0,6
   8 │ sample1,6,12,0,0,0,12
   9 │ sample1,7,23,0,0,0,23
  10 │ sample1,8,27,0,0,0,27
  11 │ ...
```

### Plots

For each input file a number of summary plots are also generated  in the output directory. These show the number of
pores per layer with a Gaussian distribution overlaid and the full-width half max range marked on the graph. This
determines the range of layers for which the area per layer and the log-area per layer are then plotted.

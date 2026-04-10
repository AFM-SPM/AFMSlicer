# Configuration

## Custom Configuration

The default configuration may not represent the best set of options for processing your images. To facilitate this it is
possible to use `afmslicer create-config` to generate a copy of the default configuration files and edit the parameters.

``` shell
afmslicer create-config --help
usage: afmslicer create-config [-h] [-f FILENAME] [-o OUTPUT_DIR] [-c CONFIG]

Create a configuration file using the defaults.

options:
  -h, --help            show this help message and exit
  -f FILENAME, --filename FILENAME
                        Name of YAML file to save configuration to (default 'config.yaml').
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Path to where the YAML file should be saved (default './' the current directory).
  -m MODULE, --module MODULE
                        The AFM module to use, currently `afmslicer` (default).
  -c CONFIG, --config CONFIG
                        Configuration to use, currently only 'default' is supported.
```

To create your own configuration file within the current directory called `e_coli_config_20260326.yaml` you would run

``` shell
afmslicer create-config --filename e_coli_config_20260306.yaml
[Thu, 26 Mar 2026 11:35:22] [INFO    ] [topostats] A sample configuration has been written to : e_coli_20260306.yaml
```

This is an ASCII text file in [YAML][yaml] format which can be opened in a text editor for editing.

<!-- markdownlint-disable MD046 -->
!!! note

    Microsoft Word and other word processors are not suitable for editing ASCII text files.
<!-- markdownlint-enable MD046 -->

An example of the file that is created is shown below. Each line has an explanation of what the parameter controls and
possible values where appropriate.

``` yaml
base_dir: ./ # Directory in which to search for data files
output_dir: ./output # Directory to output results to
log_level: info # Verbosity of output. Options: warning, error, info, debug
cores: 2 # Number of CPU cores to utilise for processing multiple files simultaneously.
file_ext: .spm # File extension of the data files.
loading:
  channel: Height # Channel to pull data from in the data files.
  extract: raw # Array to extract when loading .topostats files.
filter:
  run: true # Options : true, false
  row_alignment_quantile: 0.5 # lower values may improve flattening of larger features
  gaussian_size: 4.0 # Gaussian blur intensity in px
  gaussian_mode: nearest # Mode for Gaussian blurring. Options : nearest, reflect, constant, mirror, wrap
  # Scar remvoal parameters. Be careful with editing these as making the algorithm too sensitive may
  # result in ruining legitimate data.
  remove_scars:
    run: false
    removal_iterations: 2 # Number of times to run scar removal.
    threshold_low: 0.250 # lower values make scar removal more sensitive
    threshold_high: 0.666 # lower values make scar removal more sensitive
    max_scar_width: 4 # Maximum thickness of scars in pixels.
    min_scar_length: 16 # Minimum length of scars in pixels.
slicing:
  slices: 50 # Number of slices to create through the image between the min and max height
  segment_method: label # Method for segmenting images. Options : label, watershed
  area: true # Whether to calculate the area of pores on each slice, pretty much always needed!
  minimum_size: 20000 # Minimum size in nanometres squared of objects to retain, <= minimum_area are masked & excluded
  centroid: false # Whether to calculate the centroid of pores on each slice.
  feret_maximum: false # Whether to calculate the maximum feret distance of pores on each slice
  area_thresholds:
    low: 20
    medium: 500
    high: 1500
  pore_colors:
    - yellow
    - green
    - magenta
    - blue
plotting:
  format: png # Format for saving images as
  gif_duration: 100 # duration in microseconds between frames in GIF
  gif_loop: 0 # Whether to loop the GIF (0 = False, 1 = True)

```

If you wanted to change...

- the default `base_dir` to always be `data/`
- the number of slices/layers from `50` to `180`
- the area threshold boundaries

...you would edit the values under...

``` yaml
base_dir: data/
...
slicing:
  slices: 180
  ...
  area_thresholds:
    low: 100
    medium: 400
    high: 1200
```

...save the file and you can then run your analysis with this modified configuration by specifying the name of your YAML
file.

``` shell
rm -rf output/      # Optionally remove the output/ directory first
afmslicer --config e_coli_config_20260306.yaml
```

[yaml]: https://yaml.org

# AFMSlicer

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]
[![Code style: black][code-style-black-badge]][code-style-black-link]
[![Code style: flake8][flake8-badge]][flake8-link]
[![codecov][codecov-badge]][codecov-link]
[![pre-commit.ci status][pre-commit-badge]][pre-commit-link]
[![fair-software.eu][fair-software-badge]][fair-software-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- [![PyPI version][pypi-version]][pypi-link] -->
<!-- [![PyPI platforms][pypi-platforms]][pypi-link] -->

AFMSlicer is a package for processing images from Atomic Force Microscopy of bacterial cell walls. From a single image
of heights a user defined number of masks that "slice" through the image at regular intervals are taken. Each slice is
segmented to identify areas and the distribution of these areas across slices, which typically approximates a Gaussian
distribution, is used to select the Full-width Half-max range of slices to subset for summarisation.

## Documentation

For the full documentation please refer to [AFMSlicer Docs][doc-gh-pages] which are also mirrored at
[ReadTheDocs][rtd-link].

## Installation

Currently AFMSlicer is only available via GitHub, although a release to the [Python Package Index (PyPI)][pypi] is
planned. There are two options for installing from GitHub.

1. Install directly from GitHub.
2. Clone and install from a local copy.

Which you use depends on your use case. If you are going to just be using AFMSlicer then Option 1 will suffice. If you
will be developing and making contributions to the code base then Option 2 is recommended. In either case it is
recommended that use use [uv][uv] to setup a virtual environment in which to install AFMSlicer.

### Install directly from GitHub

[pip][pip] has support for installing packages [directly from GitHub][pypi-github].

``` shell
mkdir afmslicer
cd afmslicer
uv venv --python 3.11
source .venv/bin/activate   # This command works on GNU/Linux and OSX but will not under Windows, see output from
previous step
uv pip install git+https://github.com/AFM-SPM/AFMSlicer.git@main
```

You can replace `@main` in the last command to any git commit hash or tag if you want to install a specific version of
the code.

### Clone and install

If developing or contributing to AFMSlicer then this is the recommended method. You will need to have [Git][git]
installed on your computer in order to clone the repository (Windows users are recommended to use [Git Bash][gitbash]).

The following installs the package along with all development dependencies that are required for working on the code base.

``` shell
git clone git@github.com:AFM-SPM/AFMSlicer.git
cd AFMSlicer
uv venv --python 3.11
source .venv/bin/activate   # This command works on GNU/Linux and OSX but will not under Windows, see output from previous step
uv pip install -e. --group dev
```

## License

This software is licensed under the [GNU GPLv3 License](COPYING.md).

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/ns-rse/AFMSlicer/workflows/CI/badge.svg
[actions-link]:             https://github.com/ns-rse/AFMSlicer/actions
[code-style-black-badge]:   https://img.shields.io/badge/code%20style-black-000000.svg
[code-style-black-link]:    https://github.com/psf/black
[codecov-badge]:            https://codecov.io/gh/ns-rse/AFMSlicer/branch/dev/graph/badge.svg
[codecov-link]:             https://codecov.io/gh/ns-rse/AFMSlicer
[doc-gh-pages]:             https://afm-spm.github.io/AFMSlicer/
[fair-software-badge]:      https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8B-yellow
[fair-software-link]:       https://fair-software.eu
[flake8-badge]:             https://img.shields.io/badge/code%20style-flake8-456789.svg
[flake8-link]:              https://github.com/psf/flake8
[git]:                      https://git-scm.com
[gitbash]:                  https://git-scm.com/install/windows
<!-- [conda-badge]:              https://img.shields.io/conda/vn/conda-forge/AFMSlicer -->
<!-- [conda-link]:               https://github.com/conda-forge/AFMSlicer-feedstock -->
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/ns-rse/AFMSlicer/discussions
[pre-commit-badge]:         https://results.pre-commit.ci/badge/github/ns-rse/AFMSlicer/main.svg
[pre-commit-link]:          https://results.pre-commit.ci/latest/github/ns-rse/AFMSlicer/main
[pip]:                      https://pip.pypa.io/en/stable/index.html
[pypi]:                     https://pypi.org/
[pypi-github]:              https://pip.pypa.io/en/stable/getting-started/#install-a-package-from-github
<!-- [pypi-link]:                https://pypi.org/project/AFMSlicer/ -->
<!-- [pypi-platforms]:           https://img.shields.io/pypi/pyversions/AFMSlicer -->
<!-- [pypi-version]:             https://img.shields.io/pypi/v/AFMSlicer -->
[rtd-badge]:                https://readthedocs.org/projects/AFMSlicer/badge/?version=latest
[rtd-link]:                 https://AFMSlicer.readthedocs.io/en/latest/?badge=latest
[uv]:                       https://docs.astral.sh/uv
<!-- prettier-ignore-end -->

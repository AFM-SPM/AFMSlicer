# Installation

This page details how to setup a [Python][python] virtual environment using
[uv][uv] to install and run AFMSlicer.

## Virtual Environment

We recommend using [uv][uv] to install and manage your virtual environments but
you do not have to. Other options such [virtualenvwrapper][venvwrapper] and
[MiniForge][miniforge] also allow you to create and manage virtual environments
but are not covered here.

### Install `uv`

[uv][uv] has excellent [instructions][uv_install] on how to install the software
on different operating systems.

### Create a Virtual Environment

To create a virtual environment you should create a directory for undertaking
your work, change directory into it and then us `uv venv --python 3.11`.

**NB** Because of a performance regression in one of [TopoStats][topostats]
dependencies, the package [Topoly][topoly_issue] TopoStats only supports Python
3.10 and 3.11. The developers are aware and are hoping this will be resolved in
the near future.

```shell
mkdir AFMSlicer
cd AFMSlicer
uv venv --python 3.11
```

You can now activate the environment and check that your are using the `python`
binary from that environment.

```shell
source .venv/bin/activate
which python
python --version
```

You should see output similar to the following although the first line will
depend on your operating system and the path where you created the `afm_slicing`
programme.

```shell
/home/user/work/AFMSlicer
Python 3.11.13
```

If you use [direnv][direnv] to activate virtual environments automatically when
you enter them you can add the following to `.envrc` to activate the environment
from within the `AFMSlicer` directory.

```shell
echo "#!/bin/bash\nsource .venv/bin/activate" > .envrc
```

## PyPI

Currently AFMSlicer has not been packaged and released to [PyPi][pypi], when
this changes instructions will be added.

## Development

AFMSlicer is in the early stages of development and as such must be installed
from the GitHub repository along with development versions of its dependency
[TopoStats][topostats]. The steps involved in installing the development version
are described below and require that you install and use [uv][uv] to set things
up.

1. Clone the [AFMSlicer repository][afmslicer_gh].
2. Clone the [TopoStats repository][topostats_gh].
3. Switch to the `ns-rse/1102-switching-to-TopoStats-class` branch on TopoStats.
4. Activate the virtual environment and synchronise it.
5. Install the development version of TopoStats under the virtual environment
   (this will pull in the [AFMReader][afmreader] dependency automatically).
6. Install AFMSlicer.

```shell
cd ~/work/
git clone git@github.com:ns-rse/AFMSlicer.git
git clone git@github.com:AFM-SPM/TopoStats.git
cd TopoStats
git checkout ns-rse/1102-switching-to-TopoStats-class
git pull
cd ../AFMSlicer
source .venv/bin/activate
uv sync
uv pip install -e ../TopoStats/.
uv pip install -e ".[dev,tests]"
```

[afmreader]: https://AFM-SPM.github.io/AFMReader/
[afmslicer_gh]: https://github.com/ns-rse/AFMSlicer
[miniforge]: https://github.com/conda-forge/miniforge
[direnv]: https://direnv.net
[pypi]: https://pypi.org/
[python]: https://python.org
[uv]: https://docs.astral.sh/uv/
[uv_install]: https://docs.astral.sh/uv/getting-started/installation/
[topostats]: https://AFM-SPM.github.io/TopoStats/
[topostats_gh]: https://github.com/AFM-SPM/TopoStats
[topoly_issue]: https://github.com/ilbsm/topoly_tutorial/issues/4
[venvwrapper]: https://virtualenvwrapper.readthedocs.io/en/latest/

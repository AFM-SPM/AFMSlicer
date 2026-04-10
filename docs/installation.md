# Installation

Usage of AFMSlicer and the Napari plugin require installation and usage of a
[Python Virtual Environment][venvprimer].
This page details how to setup a [Python][python] virtual environment using
[uv][uvdocs] and
how to install AFMSlicer and the Napari plugin within this virtual environment.

## Virtual Environment

We recommend using [uv][uvdocs] to install and manage your virtual environments but you do not have to. Other options such
[virtualenvwrapper][venvwrapper] and
[MiniForge][miniforge] also allow you to create and manage virtual environments but
are not covered here.

### Install `uv`

[uv][uvdocs] has excellent
[instructions][uvinstall] on how to install the software on different operating systems.

### Create a project folder

You need somewhere to undertake the work of processing images with AFMSlicer. There are two strategies that could be
employed here...

- Create a single directory for installing AFMSlicer virtual environment and have nested directories for each
  project/set of images.
- Create a virtual environment in each directory of images you wish to process.

The following instructions assume the former approach and that the directory is called `AFMSlicer` and is located under
the users home directory `work/`

### Create a Virtual Environment

To create a virtual environment you should create a directory for undertaking your work, change directory into it and
then us `uv venv --python 3.11`.

**NB** Because of a performance regression in one of [TopoStats][topostats] dependencies, the package
[Topoly][topolyissue], TopoStats only supports Python 3.10 and 3.11. The TopoStats developers are aware and are hoping
this will be resolved in the near future. In the mean time you have to explicitly create

=== "Linux"

    ```shell
    cd ~/work/
    mkdir AFMSlicer
    cd AFMSlicer
    uv venv --python 3.11
    ```

=== "Windows"

    ``` shell
    cd Work
    mkdir AFMSlicer
    cd AFMSlicer
    uv venv --python 3.11
    ```

### Activate Virtual Environment

You can now activate the environment and check that your are using the `python` binary from that environment.

=== "Linux/OSX"

    ```shell
    source .venv/bin/activate
    which python
    python --version
    ```
    You should see output similar to the following although the first line will depend on your operating system and the path
    where you created the `afm_slicing` programme.

    ```shell
    /home/user/work/AFMSlicer
    Python 3.11.13
    ```

<!-- markdownlint-disable MD046 -->
!!! tip "`direnv`"

    GNU/Linux and OSX users may want to consider using [dirnev][direnv] to automate activation of virtual environments on
    navigation to the `AFMSlicer` directory. Add the following to `.envrc` to activate the environment from within the
    `AFMSlicer` directory.

    ```shell
    echo "#!/bin/bash\nsource .venv/bin/activate" > .envrc
    ```
<!-- markdownlint-disable MD046 -->

=== "Windows"

    ``` shell
    .venv\Scripts\activate
    where python
    python --version
    ```
    You should see output similar to the following although the first line will depend on your operating system and the
    path where you created the `afm_slicing` programme.

    ```shell
    c:\work\AFMSlicer
    Python 3.11.13
    ```

## Installing AFMSlicer

### PyPI

<!-- markdownlint-disable MD046 -->
!!! failure

    Currently AFMSlicer has not been packaged and released to [PyPi][pypi]. For now the following will _not_ work and
    you should follow the **Development** isntructions.
<!-- markdownlint-enable MD046 -->

You can install both AFMSlicer and the Napari plugin within the virtual environment using the following command.

<!-- markdownlint-disable MD046 -->
``` shell
uv pip install napari-afmslicer
```
<!-- markdownlint-enable MD046 -->

### Development

AFMSlicer is in the early stages of development and as such must be installed from the GitHub repository along with the
development versions of its dependency [TopoStats][topostats]. The steps involved in installing the development version
are described below and require that you install and use [uv][uvdocs] to set things up.

#### Cloning Repositories

The following repositories need cloning.

- [AFMSlicer repository][afmslicergh].
- [napari-afmslicer repository][afmslicergh]
- [TopoStats][topostatsgh]

You can use the following commands to clone these.

<!-- markdownlint-disable MD046 -->
```shell
mkdir -p ~/work/git/hub
cd ~/work/git/hub
git clone https://github.com/AFM-SPM/AFMSlicer.git
git clone https://github.com/AFM-SPM/napari-afmslicer.git

```
<!-- markdownlint-enable MD046 -->

<!-- markdownlint-disable MD046 -->
!!! note "TopoStats"

    Unfortunately the TopoStats repository is large and will take a while to clone.
<!-- markdownlint-enable MD046 -->

### Create a virtual environment and install packages

In the cloned `AFMSlicer` directory use `uv` to create a virtual environment and as advised activate it.

**NB** TopoStats is constrained to using Python >=3.10 and < 3.12 hence why Python 3.11 is explicitly installed.

<!-- markdownlint-disable MD046 -->
```shell
cd ~/work/git/hub
uv venv --python 3.11
source .venv/bin/activate
```
<!-- markdownlint-enable MD046 -->

Install AFMSlicer and napari-afmslicer using the following commands.

<!-- markdownlint-disable MD046 -->
```shell
uv pip install AFMSlicer/.
uv pip install napari-afmslicer/.
```
<!-- markdownlint-enable MD046 -->

You can now proceed to the [usage](usage/index.md) section to learn how to use AFMSlicer.

[afmslicergh]: https://github.com/ns-rse/AFMSlicer
[miniforge]: https://github.com/conda-forge/miniforge
[uvinstall]: https://docs.astral.sh/uv/getting-started/installation
[topostats]: https://AFM-SPM.github.io/TopoStats
[topostatsgh]: https://github.com/AFM-SPM/TopoStats
[topolyissue]: https://github.com/ilbsm/topoly_tutorial/issues/4
[venvwrapper]: https://virtualenvwrapper.readthedocs.io/en/latest

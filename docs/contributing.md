# Contributing

Contributions of all forms are welcome, whether that is [bug
reports][afmslicer_bug], [feature requests][afmslicer_feature] or
[questions][afmslicer_discussion] because our documentation isn't clear.

Please refer to the [installation](installation.md) instructions for cloning and
install the development version of AFMSlicer and its dependencies.

## Cloning the repository

If you wish to make changes to the code base and are not a collaborator on the
repository you will have to [fork][gh_fork] the repository, [clone][git_clone]
it, make your changes and the submit a pull request.

```shell
# Collaborator
git clone git@github.com:ns-rse/AFMSlicer.git
# Forked copy
git clone git@github.com:<YOUR_GITHUB_USERNAME>/TopoStatsAFMSlicer.git
```

## Install Additional Dependencies

If you are to contribute you should install the additional dependencies for
undertaking work and enable the [pre-commit][pre-commit] hooks.

If you haven't already create a virtual environment and install the packages.

```shell
uv venv --python=3.11
source .venv/bin/activate
uv sync
uv pip install -e ".[dev,docs,tests]"
```

## Creating a Branch

Typically you will create a branch to make changes on (). It is not compulsory
but we try to use a consistent nomenclature for branches that shows who has
worked on the branch, the issue it pertains to and a short description of the
work. To which end you will see branches with the form
`<GITHUB_USERNAME>/<GITHUB_ISSUE>-<DESCRIPTION>`. Some examples are shown below…

| Branch                   | User                                   | Issue                                              | Description                  |
| :----------------------- | -------------------------------------- | -------------------------------------------------- | ---------------------------- |
| `ns-rse/5-readme-bagdes` | [`ns-rse`](https://github.com/ns-rse/) | [#5](https://github.com/ns-rse/AFMSlicer/issues/5) | Adding badges to `README.md` |

Here we ensure the `main` branch is up-to-date and create a new branch
`ns-rse/13-new-feature`

```shell
git switch main
git pull
git switch -c ns-rse/13-new-feature
```

You can now start working on your feature or bug fix making regular commits.

## Software Development

To make the codebase easier to maintain we ask that you follow the guidelines
below on coding style, linting, typing, documentation and testing. These entail
a number of additional dependencies that can be installed with the following
command.

```shell
pip install -e .[dev,tests,docs]
```

This will pull in all the dependencies we use for development (`dev`), tests
(`tests`) and writing documentation (`docs`).

### Coding Style/Linting

Using a consistent coding style has many benefits (see
[Linting : What is all the fluff about?](https://rse.shef.ac.uk/blog/2022-04-19-linting/)).
For this project we aim to adhere to [PEP8 - the style Guide for Python Code]
and do so using the formatting linters [black][black] and [ruff][ruff]. Ruff
implements the checks made by [Flake8][flake8]), [isort][isort], [mypy][mypy]
and [numpydoc-validation][numpydoc-validation]. We also like to ensure the code
passes [pylint][pylint] which helps identify code duplication and reduces some
of the [code smells][code_smell] that we are all prone to making. A `.pylintrc`
is included in the repository. These checks are run on all Pull Requests via
[pre-commit.ci][pre_commit_ci] and have to pass before contributions can be
merged to `main`.

Many popular IDEs such as VSCode, PyCharm, Spyder and Emacs all have support for
integrating these linters into your workflow such that when you save a file the
linting/formatting is automatically applied.

### Pre-commit

[pre-commit][pre-commit] is a powerful and useful tool that runs hooks on your
code prior to making commits. For a more detailed exposition see
[pre-commit : Protecting your future self](https://rse.shef.ac.uk/blog/pre-commit/).
The repository includes `pre-commit` as a development dependency as well as a
`.pre-commit-config.yaml`. To use these locally you should have already
installed all the `dev` dependencies in your virtual environment. You then need
to install `pre-commit` configuration and hooks (**NB** this will download
specific virtual environments that `pre-commit` uses when running hooks so the
first time this is run may take a little while).

```shell
pre-commit install --install-hooks
```

If these fail then you will not be able to make a commit until they are fixed.
Several of the linters will automatically format files so you can simply
`git add -u .` those and try committing straight away. `flake8` does not correct
files automatically so the errors will need manually correcting.

If you do not enable and resolve issues reported by `pre-commit` locally before
making a pull request you will find the [`pre-commit.ci`][pre_commit_ci] GitHub
Action will fail, preventing your Pull Request from being merged. You can
shorten the feedback loop and speed up the resolution of errors by enabling
`pre-commit` locally and resolving issues before making your commits.

### Typing

Whilst Python is a dynamically typed language (that is the type of an object is
determined dynamically) the use of Type Hints is expected as it makes reading
and understanding the code considerably easier for contributors and the code
base more robust. These are checked on commits and pull-requests via a
[pre-commit](#pre-commit) hook that runs [mypy][mypy]. For more on Type Hints
see [PEP483][pep483] and [PEP484][pep484].

### Documentation

All classes, methods and functions should have [numpy docstrings][numpydoc]
defining their functionality, parameters and return values. [Pylint][pylint]
will note and report the absence of docstrings by way of the
`missing-function-docstring` condition and the docstrings are checked the the
[pre-commit](#pre-commit) hook [numpydoc-validation][numpydoc-validation].
Further, when new methods that introduce changes to the configuration are
incorporated into the package they should be documented under
[Parameter Configuration](configuration). [pre-commit](#pre-commit) has the
[markdownlint-cli2][markdownlint-cli2] hook enabled to lint all Markdown files
and will where possible automatically fix things, but some issues need resolving
manually.

### Testing

New features should have unit-tests written and included under the `tests/`
directory to ensure the functions work as expected. The [pytest][pytest]
framework is used for running tests along with a number of plugins
([syrupy][syrupy] for regression testing; [pytest-mpl][pytest-mpl] for testing
generated Matplotlib images). In conjunction with [pre-commit](#pre-commit) we
leverage [pytest-testmon][pytest-testmon]) to run tests on each commit, but as
the test suite is large and can take a while to run `pytest-testmon` restricts
tests to only files that have changed (code or tests) or changes in environment
variables and dependencies. You will need to create a database locally on first
run and so should run the following before undertaking any development.

```shell
pytest --testmon
```

This will create a database (`.testmondata`) which tracks the current state of
the repository, this file is ignored by Git (via `.gitignore`) but keeps track
of the state of the repository and what has changed so that the `pre-commit`
hook `Pytest (testmon)` only attempts to run the tests when changes have been
made to files that impact the tests.

## Debugging

To aid with debugging we include the [snoop][snoop] package as a `dev`
dependency. The package is disabled by default, but when you have a class,
method or function you wish to debug you should add
`snoop.install(enabled=True)` to the file you wish to debug and use the `@snoop`
decorator around the function/method you wish to debug.

## Configuration

As described in [Parameter Configuration](configuration) options are primarily
passed to AFMSlicer via a [YAML][yaml] configuration file. When introducing new
features that require configuration options you will have to ensure that the
default configuration file (`afmslicer/default.yaml`) is updated to include your
options and that corresponding arguments are added to the entry point (please
refer to [Adding Modules](contributing/adding_modules) page which covers this).
Further the `afmslicer.validation.validate.config()` function, which checks a
valid configuration file with all necessary fields has been passed when invoking
`afmslicer` sub-commands, will also need updating to include new options in the
Schema against which validation of configuration files is made.

### IDE Configuration

Linters such as `black`, `flake8` and `pylint` can be configured to work with
your IDE so that say Black and/or formatting is applied on saving a file or the
code is analysed with `pylint` on saving and errors reported. Setting up and
configuring IDEs to work in this manner is beyond the scope of this document but
some links to articles on how to do so are provided.

- [Linting Python in Visual Studio Code](https://code.visualstudio.com/docs/python/linting)
- [Code Analysis — Spyder](http://docs.spyder-ide.org/current/panes/pylint.html)
  for `pylint` for Black see
  [How to use code formatter Black with Spyder](https://stackoverflow.com/a/66458706).
- [Code Quality Assistance Tips and Tricks, or How to Make Your Code Look Pretty? | PyCharm](https://www.jetbrains.com/help/pycharm/tutorial-code-quality-assistance-tips-and-tricks.html#525ee883)
- [Reformat and rearrange code | PyCharm](https://www.jetbrains.com/help/pycharm/reformat-and-rearrange-code.html)
- [Advanced Python Development Workflow in Emacs | Serghei's Blog](https://blog.serghei.pl/posts/emacs-python-ide/)
- [Getting started with lsp-mode for Python](https://www.mattduck.com/lsp-python-getting-started.html)

[afmslicer_bug]:
  https://github.com/ns-rse/AFMSlicer/issues/new?template=bug_report.yaml
[afmslicer_discussion]: https://github.com/ns-rse/AFMSlicer/discussions
[afmslicer_feature]:
  https://github.com/ns-rse/AFMSlicer/issues/new?template=feature_request.yaml
[black]: https://github.com/psf/black
[code_smell]: https://en.wikipedia.org/wiki/Code_smell
[flake8]: https://flake8.pycqa.org/en/latest/
[git_clone]:
  https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository
[gh_fork]: https://docs.github.com/en/get-started/quickstart/fork-a-repo
[isort]: https://pycqa.github.io/isort/
[markdownlint-cli2]: https://github.com/DavidAnson/markdownlint-cli2
[mypy]: https://www.mypy-lang.org/
[numpydoc]: https://numpydoc.readthedocs.io/en/latest/format.html
[numpydoc-validation]: https://numpydoc.readthedocs.io/en/latest/validation.html
[pep483]: https://peps.python.org/pep-0483/
[pep484]: https://peps.python.org/pep-0484/
[pre-commit]: https://pre-commit.com
[pre_commit_ci]: https://pre-commit.ci
[pylint]: https://www.pylint.org/
[pytest]: https://docs.pytest.org/en/latest/
[pytest-mpl]: https://github.com/matplotlib/pytest-mpl
[pytest-testmon]: https://github.com/tarpas/pytest-testmon/
[ruff]: https://github.com/astral-sh/ruff
[snoop]: https://github.com/alexmojaki/snoop
[syrupy]: https://syrupy-project.github.io/syrupy/
[yaml]: https://yaml.org

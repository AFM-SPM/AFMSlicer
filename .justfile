@default:
    @just --list

# Pre-commit ruff-check all files
pc_ruff:
    pre-commit run ruff-check --all-files ; pre-commit run ruff-format --all-files

# Pre-commit numpydoc all files
pc_npd:
    pre-commit run numpydoc-validation --all-files

# Pre-commit blacken all files
pc_black:
    pre-commit run blacken-docs --all-files

# Pre-commit prettier all files
pc_prettier:
    pre-commit run prettier --all-files

# Pre-commit markdownlint-cli2 all files
pc_markdown:
    pre-commit run markdownlint-cli2 --all-files

# Pre-commit pylint all files
pc_pylint:
    pre-commit run pylint --all-files

# uv pip install the package with all dev dependencies
pip_install:
    uv pip install --editable . --group dev

# Pytest everything
pytest:
    pytest --cov=src --cov-branch --numprocesses=logical

# Pytest with Matplotlib html report
pytest_mpl_html_report:
    mkdir -p tmp/mpl
    pytest --mpl-generate-summary html --mpl-results-path tmp/mpl tests/test_plotting.py tests/test_classes.py

# Pytest update Matplotlib snapshots
pytest_mpl_update:
    mkdir -p tests/__mpl_snapshots__/
    pytest --mpl-generate-path=tests/__mpl_snapshots__ tests/test_plotting.py
    pytest --mpl-generate-path=tests/__mpl_snapshots__ tests/test_classes.py

# Pytest update Syrupy snapshots
pytest_snapshot_update:
    pytest --snapshot-update

# Pytest update Maplotlib and Syrupy snapshots
pytest_update:
    just pytest_mpl_update
    just pytest_snapshot_update

# uv sync and lock
uv_upgrade:
    uv sync
    uv lock --upgrade

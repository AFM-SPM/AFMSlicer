"""Test the package."""

from __future__ import annotations

import importlib.metadata

import afmslicer as m


def test_version():
    """Check the package version."""
    assert importlib.metadata.version("afmslicer") == m.__version__

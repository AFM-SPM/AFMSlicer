"""
Copyright (c) 2025 Neil Shephard. All rights reserved.

AFMSlicer: Package for slicing AFM images.
"""

from __future__ import annotations

import os
from importlib.metadata import version

import snoop

from ._version import version as __version__

__all__ = ["__version__"]

# Disable TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

__release__ = ".".join(__version__.split(".")[:-2])

AFMSLICER_DETAILS = version("afmslicer").split("+g")
AFMSLICER_VERSION = AFMSLICER_DETAILS[0]
AFMSLICER_COMMIT = AFMSLICER_DETAILS[1].split(".d")[0]

# Disable snoop
snoop.install(enabled=False)

"""
Copyright (c) 2025 Neil Shephard. All rights reserved.

AFMSlicer: Package for slicing AFM images.
"""

from __future__ import annotations

import os
from importlib.metadata import version

import snoop
from packaging.version import Version

# Disable TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

__version__ = version("afmslicer")
__release__ = ".".join(__version__.split(".")[:-2])

AFMSLICER_VERSION = Version(__version__)
if AFMSLICER_VERSION.is_prerelease and AFMSLICER_VERSION.is_devrelease:
    AFMSLICER_BASE_VERSION = str(AFMSLICER_VERSION.base_version)
    AFMSLICER_COMMIT = str(AFMSLICER_VERSION).split("+g")[1]
else:
    AFMSLICER_BASE_VERSION = str(AFMSLICER_VERSION)
    AFMSLICER_COMMIT = ""
CONFIG_DOCUMENTATION_REFERENCE = """# For more information on configuration and how to use it:
# https://afm-spm.github.io/Afmslicer/main/configuration.html\n"""


# Disable snoop
snoop.install(enabled=False)

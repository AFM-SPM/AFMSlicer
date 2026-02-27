"""Test the each processing module runs."""

from __future__ import annotations

from pathlib import Path
from platform import python_version

import numpy as np
import pytest
from packaging.version import parse as parse_version

from afmslicer import processing

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
SPM_DIR = RESOURCES / "spm"


@pytest.mark.parametrize(
    (
        "afmslicer_fixture",
        "config_fixture",
        "expected_slices",
        "expected_pores_per_layer",
        "expected_min_height",
        "expected_max_height",
    ),
    [
        pytest.param(
            "afmslicer_sample1",
            None,
            5,
            [1, 43, 31, 63, 1],
            -312.40853721614576,
            551.5217325223152,
            id="sample1, no config",
        ),
        pytest.param(
            "afmslicer_sample1",
            "default_config",
            5,
            [1, 43, 31, 63, 1],
            -312.40853721614576,
            551.5217325223152,
            id="sample1, with config",
        ),
        pytest.param(
            "afmslicer_sample2",
            None,
            5,
            [1, 76, 84, 56, 1],
            -296.85145995382425,
            -152.62116556541318,
            id="sample2, no config",
        ),
        pytest.param(
            "afmslicer_sample2",
            "default_config",
            5,
            [1, 76, 84, 56, 1],
            -296.85145995382425,
            -152.62116556541318,
            id="sample2, with config",
        ),
    ],
)
def test_slicer(
    afmslicer_fixture: str,
    config_fixture: str,
    expected_slices: int,
    expected_pores_per_layer: list[int],
    expected_min_height: float,
    expected_max_height: float,
    request,
    snapshot,
) -> None:
    """Test for ``processing.slicer()``."""
    afmslicer = request.getfixturevalue(afmslicer_fixture)
    # The fixture hasn't been flattened we instead use the raw/original image
    afmslicer.image = afmslicer.image_original
    if config_fixture is None:
        processing.slicer_scan(topostats_object=afmslicer)
    else:
        config = request.getfixturevalue(config_fixture)
        processing.slicer_scan(topostats_object=afmslicer, config=config)
    assert afmslicer.slices == expected_slices
    assert afmslicer.pores_per_layer == expected_pores_per_layer
    assert afmslicer.min_height == expected_min_height
    assert afmslicer.max_height == expected_max_height
    if parse_version(python_version()) >= parse_version("3.11"):
        assert afmslicer.area_by_layer == snapshot


@pytest.mark.parametrize(
    (
        "afmslicer_fixture",
        "config_fixture",
        "expected_pixel_to_nm_scaling",
        "expected_filtered_image_sum",
    ),
    [
        pytest.param(
            "afmslicer_sample1",
            None,
            39.0625,
            2493.8169754143455,
            id="sample1, no config",
        ),
        pytest.param(
            "afmslicer_sample1",
            "default_config",
            39.0625,
            2493.8169754143455,
            id="sample1, with config",
        ),
        pytest.param(
            "afmslicer_sample2",
            None,
            0.625,
            70.05077383773278,
            id="sample2, no config",
        ),
        pytest.param(
            "afmslicer_sample2",
            "default_config",
            0.625,
            70.05077383773278,
            id="sample2, with config",
        ),
    ],
)
def test_filter_scan(
    afmslicer_fixture: str,
    config_fixture: str,
    expected_pixel_to_nm_scaling: float,
    expected_filtered_image_sum: float,
    request,
    snapshot,
) -> None:
    """Test for ``processing.filter_scan()``."""
    afmslicer = request.getfixturevalue(afmslicer_fixture)
    if config_fixture is None:
        processing.filter_scan(topostats_object=afmslicer)
    else:
        config = request.getfixturevalue(config_fixture)
        processing.filter_scan(topostats_object=afmslicer, config=config)
    assert afmslicer.pixel_to_nm_scaling == expected_pixel_to_nm_scaling
    assert isinstance(afmslicer.image, np.ndarray)
    assert afmslicer.image.sum() == pytest.approx(expected_filtered_image_sum, abs=1e-6)
    if parse_version(python_version()) >= parse_version("3.11"):
        assert afmslicer == snapshot


@pytest.mark.parametrize(
    (
        "afmslicer_fixture",
        "config_fixture",
        "expected_slices",
        "expected_pixel_to_nm_scaling",
        "expected_filtered_image_sum",
        "expected_pores_per_layer",
        "expected_min_height",
        "expected_max_height",
    ),
    [
        pytest.param(
            "afmslicer_sample1",
            None,
            5,
            39.0625,
            2493.8169754143455,
            [1, 8, 21, 55, 1],
            -178.92149516951616,
            641.8099083963831,
            id="sample1, no config",
        ),
        pytest.param(
            "afmslicer_sample1",
            "default_config",
            5,
            39.0625,
            2493.8169754143455,
            [1, 8, 21, 55, 1],
            -178.92149516951616,
            641.8099083963831,
            id="sample1, with config",
        ),
        pytest.param(
            "afmslicer_sample2",
            None,
            5,
            0.625,
            70.05077383773278,
            [1, 1, 1, 1, 1],
            -13.29568018896186,
            10.522422529403713,
            id="sample2, no config",
        ),
        pytest.param(
            "afmslicer_sample2",
            "default_config",
            5,
            0.625,
            70.05077383773278,
            [1, 1, 1, 1, 1],
            -13.29568018896186,
            10.522422529403713,
            id="sample2, with config",
        ),
    ],
)
def test_process(  # pylint: disable=too-many-positional-arguments
    afmslicer_fixture: str,
    config_fixture: str,
    expected_slices: int,
    expected_pixel_to_nm_scaling: float,
    expected_filtered_image_sum: float,
    expected_pores_per_layer: list[int],
    expected_min_height: float,
    expected_max_height: float,
    request,
    snapshot,
) -> None:
    """Test for ``processing.process()``."""
    afmslicer = request.getfixturevalue(afmslicer_fixture)
    if config_fixture is None:
        processing.process_scan(topostats_object=afmslicer)
    else:
        config = request.getfixturevalue(config_fixture)
        processing.process_scan(topostats_object=afmslicer, config=config)
    assert afmslicer.slices == expected_slices
    assert afmslicer.pixel_to_nm_scaling == expected_pixel_to_nm_scaling
    assert isinstance(afmslicer.image, np.ndarray)
    assert afmslicer.image.sum() == pytest.approx(expected_filtered_image_sum, abs=1e-6)
    assert afmslicer.pores_per_layer == expected_pores_per_layer
    assert afmslicer.min_height == expected_min_height
    assert afmslicer.max_height == expected_max_height
    if parse_version(python_version()) >= parse_version("3.11"):
        assert afmslicer == snapshot

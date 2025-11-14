"""Test the slice module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from afmslicer import slicer

# pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments,protected-access

BASE_DIR = Path.cwd()
RESOURCES = BASE_DIR / "tests" / "resources"
RESOURCES_SLICER = RESOURCES / "slicer"

ABSOLUTE_TOLERANCE = 1e-5
RELATIVE_TOLERANCE = 1e-5


@pytest.mark.parametrize(
    ("height_fixture", "slices", "shape"),
    [
        pytest.param(
            "pyramid_array",
            5,
            (11, 11, 5),
            id="pyramid",
        ),
        pytest.param(
            "square_array",
            5,
            (7, 7, 5),
            id="square",
        ),
        pytest.param(
            "sample1_spm",
            5,
            (512, 512, 5),
            id="sample1",
        ),
        pytest.param(
            "sample2_spm",
            5,
            (640, 640, 5),
            id="sample2",
        ),
    ],
)
def test_slicer(
    height_fixture: str,
    slices: int,
    shape: tuple[int],
    request,
    snapshot,
) -> None:
    """Test for slicer() function."""
    if height_fixture in ["pyramid_array", "square_array"]:
        heights = request.getfixturevalue(height_fixture)
    else:
        heights, _ = request.getfixturevalue(height_fixture)
    sliced = slicer.slicer(heights=heights, slices=slices)
    assert sliced.shape == shape
    np.savez_compressed(RESOURCES_SLICER / f"{height_fixture}_sliced.npz", sliced)
    # ns-rse: syrupy doesn't yet support numpy arrays so we convert to string
    #         https://github.com/syrupy-project/syrupy/issues/887
    assert np.array2string(sliced) == snapshot


@pytest.mark.parametrize(
    (
        "sliced_fixture",
        "slices",
        "min_height",
        "max_height",
    ),
    [
        pytest.param(
            "pyramid_array_sliced",
            None,
            None,
            None,
            id="pyramid array, no slices/min/max",
        ),
        pytest.param(
            "square_array_sliced",
            None,
            None,
            None,
            id="square array, no slices/min/max",
        ),
        pytest.param(
            "sample1_spm_sliced",
            None,
            None,
            None,
            id="sample1, no slices/min/max",
        ),
        pytest.param(
            "sample2_spm_sliced",
            None,
            None,
            None,
            id="sample2, no slices/min/max",
        ),
    ],
)
def test_mask_slices(
    sliced_fixture: npt.NDArray[np.int8],
    slices: int,
    min_height: float,
    max_height: float,
    request,
    snapshot,
) -> None:
    """Test for mask_slices()."""
    sliced_array: npt.NDArray = request.getfixturevalue(sliced_fixture)
    masked_slices = slicer.mask_slices(
        stacked_array=sliced_array,
        slices=slices,
        min_height=min_height,
        max_height=max_height,
    )
    np.savez_compressed(RESOURCES_SLICER / f"{sliced_fixture}_mask.npz", masked_slices)
    # ns-rse: syrupy doesn't yet support numpy arrays so we convert to string
    #         https://github.com/syrupy-project/syrupy/issues/887
    assert np.array2string(masked_slices) == snapshot


@pytest.mark.parametrize(
    ("array_fixture", "expected"),
    [
        pytest.param(
            "basic_three_segments",
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 1, 0, 2, 2, 2, 0],
                    [0, 1, 0, 0, 1, 0, 2, 0, 2, 0],
                    [0, 1, 1, 1, 1, 0, 2, 0, 2, 0],
                    [0, 0, 0, 0, 0, 0, 2, 0, 2, 0],
                    [0, 3, 3, 3, 3, 0, 2, 0, 2, 0],
                    [0, 3, 0, 0, 3, 0, 2, 2, 2, 0],
                    [0, 3, 3, 3, 3, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int32,
            ),
            id="unconnected",
        ),
    ],
)
def test_label(array_fixture: str, expected: npt.NDArray, request) -> None:
    """Test for slicer._label()."""
    array: npt.NDArray = request.getfixturevalue(array_fixture)
    np.testing.assert_array_equal(slicer._label(array), expected)


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        pytest.param(
            "basic_three_segments",
            np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 2, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 2, 2, 2, 1, 1, 3, 1, 1],
                    [1, 1, 2, 2, 2, 1, 1, 3, 3, 1],
                    [1, 1, 1, 1, 1, 1, 1, 3, 3, 1],
                    [1, 1, 1, 1, 1, 1, 1, 3, 3, 1],
                    [1, 1, 1, 1, 1, 1, 1, 3, 3, 1],
                    [1, 1, 4, 4, 4, 1, 1, 3, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
                dtype=np.int32,
            ),
            id="unconnected",
        ),
    ],
)
def test_watershed(fixture: str, expected: npt.NDArray, request) -> None:
    """Test for slicer._watershed()."""
    array = request.getfixturevalue(fixture)
    np.testing.assert_array_equal(slicer._watershed(array), expected)


@pytest.mark.parametrize(
    ("sliced_mask_fixture", "method"),
    [
        pytest.param(
            "pyramid_array_sliced_mask",
            "label",
            id="pyramid height array (5 layers)",
        ),
        pytest.param(
            "square_array_sliced_mask",
            "label",
            id="square height array (5 layers)",
        ),
        pytest.param(
            "pyramid_array_mask_stacked_2",
            "label",
            id="pyramid height array (2 layers)",
        ),
        pytest.param(
            "three_layer_three_segments",
            "label",
            id="simple three layers with three segments using label",
        ),
        pytest.param(
            "three_layer_three_segments",
            "watershed",
            id="simple three layers with three segments using watershed",
        ),
        pytest.param(
            "sample1_spm_sliced_mask", "label", id="sample1 segment with label"
        ),
        pytest.param(
            "sample2_spm_sliced_mask", "label", id="sample2 segment with label"
        ),
    ],
)
def test_segment_slices(
    sliced_mask_fixture: npt.NDArray[np.bool],
    method: str,
    request,
    snapshot,
) -> None:
    """Test slicer.segment_slices()."""
    sliced_mask = request.getfixturevalue(sliced_mask_fixture)
    sliced_mask_segment = slicer.segment_slices(sliced_mask, method)
    np.savez_compressed(
        RESOURCES_SLICER / f"{sliced_mask_fixture}_segment.npz", sliced_mask_segment
    )
    # ns-rse: syrupy doesn't yet support numpy arrays so we convert to string
    #         https://github.com/syrupy-project/syrupy/issues/887
    assert np.array2string(sliced_mask_segment) == snapshot


# ns-rse 2025-11-13 : Currently feret calculations and tests are disabled as the memory requirements go through the roof
# which suggests that I've got incorrectly labelled images (my suspicion is the background is no 0 in sample1 and
# sample2 labelled images but I need to theck)
@pytest.mark.parametrize(
    (
        "sliced_labels_fixture",
        "scaling_fixture",
        "objects",
        "expected_area",
        "expected_centroid",
        # "expected_feret_diameter_max",
    ),
    [
        pytest.param(
            "pyramid_array_sliced_mask_segment",
            1,
            1,
            np.asarray([165], dtype=np.float64),
            np.asarray([(5.0, 5.0, 0.7878787878787878)], dtype=np.float64),
            # [12.041594578792296],
            id="pyramid height array (5 layers)",
            # marks=pytest.mark.skip(reason="development"),
        ),
        pytest.param(
            "square_array_sliced_mask_segment",
            1,
            1,
            np.asarray([125], dtype=np.float64),
            np.asarray([(3.0, 3.0, 2.0)], dtype=np.float64),
            # [12.041594578792296],
            id="square height array (5 layers)",
            # marks=pytest.mark.skip(reason="development"),
        ),
        pytest.param(
            "sample1_spm_sliced_segment",
            "sample1_scaling",
            63,
            np.array(
                [
                    15645682811.73706,
                    73313713.07373047,
                    465214252.4719238,
                    533878803.2531738,
                    142395496.3684082,
                    232100486.7553711,
                    26643276.21459961,
                    202953815.46020508,
                    28371810.913085938,
                    56922435.76049805,
                    505328178.4057617,
                    208199024.20043945,
                    38385391.23535156,
                    286102294.921875,
                    759005546.5698242,
                    302374362.94555664,
                    28252601.623535156,
                    41127204.89501953,
                    176846981.04858398,
                    11265277.862548828,
                    1057267189.0258789,
                    59843063.35449219,
                    25272369.384765625,
                    14781951.904296875,
                    40888786.31591797,
                    122725963.5925293,
                    43928623.19946289,
                    20325183.868408203,
                    18537044.525146484,
                    32484531.40258789,
                    5841255.187988281,
                    16689300.537109375,
                    90539455.41381836,
                    1013278.9611816406,
                    6794929.504394531,
                    46491622.92480469,
                    78439712.52441406,
                    129878520.96557617,
                    14662742.614746094,
                    81241130.82885742,
                    476837.158203125,
                    8404254.913330078,
                    17881393.432617188,
                    7152557.373046875,
                    54001808.166503906,
                    30815601.348876953,
                    596046.4477539062,
                    894069.6716308594,
                    15556812.286376953,
                    19848346.710205078,
                    2622604.3701171875,
                    4887580.871582031,
                    14364719.39086914,
                    9000301.361083984,
                    3874301.9104003906,
                    13887882.232666016,
                    6437301.6357421875,
                    4529953.0029296875,
                    2384185.791015625,
                    6258487.701416016,
                    6198883.056640625,
                    9238719.940185547,
                    2026557.9223632812,
                ],
                dtype=np.float64,
            ),
            np.asarray(
                [
                    (9967.418972221905, 9971.379673493568, 0.0806575273056981),
                    (775.2794715447154, 8056.370680894309, 76.53709349593495),
                    (2182.785474055093, 15421.920043241513, 70.9931534272902),
                    (2689.3595791001453, 17150.05111225857, 40.32286200736854),
                    (3563.3698723315197, 2544.1476559229804, 80.62670050230221),
                    (4774.342337570622, 12478.974062660503, 79.31874357986646),
                    (2195.0153803131993, 5799.4267337807605, 97.7873322147651),
                    (4964.20704845815, 7470.379038179149, 80.31617107195301),
                    (3768.3823529411766, 14321.576286764706, 82.3923319327731),
                    (2618.7418193717276, 4336.837369109948, 44.82984293193717),
                    (5958.109408174098, 8691.286454647323, 55.84767191554612),
                    (3929.2647795591183, 2226.8644431720586, 47.349162610936155),
                    (7099.730687111802, 9954.750582298137, 82.7348602484472),
                    (5363.924153645833, 7270.442708333333, 44.392903645833336),
                    (12275.036320087953, 14682.555069106329, 78.82747467410083),
                    (7896.392359057757, 3551.9385718509757, 44.29085353834023),
                    (8304.489715189873, 11639.553665611815, 80.18525843881856),
                    (8526.268115942028, 13520.210597826086, 61.82065217391305),
                    (8370.671132457028, 8091.440744017526, 42.709386585776876),
                    (4806.9609788359785, 5704.3650793650795, 115.12070105820106),
                    (12328.143057982861, 14882.048339017927, 39.87731142180629),
                    (14449.817915836653, 6483.052166334662, 81.27645667330677),
                    (8903.854658018869, 6223.743366745283, 103.82886202830188),
                    (14554.403981854839, 15497.259324596775, 88.36315524193549),
                    (16913.891672740523, 2498.348669825073, 83.47758746355684),
                    (15307.700188198154, 8313.653472559496, 64.2188258863526),
                    (15446.118130936227, 6096.665111940299, 88.40739484396201),
                    (13556.062133431085, 12060.919171554253, 92.4440982404692),
                    (11093.373191318327, 8122.487942122187, 100.48231511254019),
                    (11118.04759174312, 7699.397935779816, 100.98910550458716),
                    (12765.066964285714, 12305.086096938776, 96.46045918367346),
                    (10188.61607142857, 1672.154017857143, 78.125),
                    (14851.619075049375, 5314.582990454246, 45.594346609611584),
                    (9613.970588235294, 10082.720588235294, 89.61397058823529),
                    (8348.75274122807, 3206.5515350877195, 110.3344298245614),
                    (15579.477163461539, 14770.783253205129, 57.39182692307692),
                    (18244.354340805472, 1110.84726443769, 45.117781155015194),
                    (18613.505335016063, 10563.295949977053, 42.79127466727857),
                    (9480.278201219513, 3495.776168699187, 111.78861788617886),
                    (18227.25605282465, 7184.462123991196, 45.711436170212764),
                    (14013.671875, 5532.2265625, 78.125),
                    (16233.377659574468, 16021.165780141844, 66.21232269503547),
                    (11126.171875, 15241.145833333334, 108.07291666666667),
                    (10274.739583333334, 14743.489583333334, 117.1875),
                    (11843.310223509934, 15580.591197571744, 117.1875),
                    (12154.33087524178, 14088.869076402321, 117.1875),
                    (11777.34375, 14785.15625, 117.1875),
                    (11848.958333333334, 16934.895833333332, 117.1875),
                    (12495.360392720306, 16552.77179118774, 117.1875),
                    (13147.522522522522, 13575.333145645645, 117.1875),
                    (12964.311079545454, 18232.421875, 117.1875),
                    (13208.841463414634, 15853.658536585366, 117.1875),
                    (13385.794865145228, 17458.344139004148, 117.1875),
                    (13321.6059602649, 14462.696605960266, 117.1875),
                    (13325.721153846154, 12790.264423076924, 117.1875),
                    (14003.319474248927, 13275.885193133046, 117.1875),
                    (14104.817708333334, 12362.196180555555, 117.1875),
                    (14237.767269736842, 10656.86677631579, 117.1875),
                    (14277.34375, 11499.0234375, 117.1875),
                    (14918.89880952381, 12048.735119047618, 117.1875),
                    (15108.173076923076, 11051.3070791346154, 117.1875),
                    (19205.645161290322, 277.7217741935484, 117.1875),
                    (19789.751838235294, 10659.466911764706, 117.1875),
                ],
                dtype=np.float64,
            ),
            # np.asarray(, dtype=np.float64),
            id="sample1",
            # marks=pytest.mark.skip(reason="development"),
        ),
        pytest.param(
            "sample2_spm_sliced_segment",
            "sample2_scaling",
            84,
            np.asarray(
                [
                    184957.763671875,
                    42.724609375,
                    7.080078125,
                    1.220703125,
                    46643.310546875,
                    9181.396484375,
                    1.46484375,
                    3.41796875,
                    2.197265625,
                    15.13671875,
                    1.220703125,
                    0.732421875,
                    36.1328125,
                    2.685546875,
                    2.44140625,
                    1.220703125,
                    1.708984375,
                    8.7890625,
                    1.708984375,
                    0.9765625,
                    2.685546875,
                    2.44140625,
                    5.859375,
                    5.859375,
                    0.9765625,
                    2.44140625,
                    60.05859375,
                    2.197265625,
                    1.708984375,
                    0.732421875,
                    114.501953125,
                    7.568359375,
                    28.076171875,
                    8.30078125,
                    2.9296875,
                    0.732421875,
                    1.708984375,
                    2.197265625,
                    4.39453125,
                    3.41796875,
                    6.103515625,
                    2.197265625,
                    0.9765625,
                    2.44140625,
                    2.44140625,
                    87.890625,
                    2.9296875,
                    60.546875,
                    1.46484375,
                    2.685546875,
                    16.6015625,
                    3.41796875,
                    0.9765625,
                    5.37109375,
                    2.685546875,
                    1.220703125,
                    0.732421875,
                    0.48828125,
                    1.708984375,
                    3.41796875,
                    0.732421875,
                    1.46484375,
                    0.9765625,
                    1.708984375,
                    0.48828125,
                    2.197265625,
                    0.732421875,
                    0.48828125,
                    1.220703125,
                    2.197265625,
                    1.220703125,
                    0.732421875,
                    0.9765625,
                    0.732421875,
                    1.220703125,
                    1.708984375,
                    0.244140625,
                    0.244140625,
                    0.244140625,
                    0.244140625,
                    0.48828125,
                    0.9765625,
                    0.732421875,
                    0.732421875,
                ],
                dtype=np.float64,
            ),
            np.asarray(
                [
                    (212.628315295801, 194.69046294352992, 0.2870916145604399),
                    (5.75, 95.72142857142858, 0.6357142857142857),
                    (203.10344827586206, 37.62931034482759, 1.4870689655172413),
                    (121.125, 20.375, 1.125),
                    (304.47655992379, 184.07405941345505, 1.25),
                    (368.446349748717, 108.84459675060494, 1.8746676150716621),
                    (107.39583333333333, 63.125, 1.0416666666666667),
                    (105.9375, 49.330357142857146, 1.0267857142857142),
                    (61.736111111111114, 88.68055555555556, 0.8333333333333334),
                    (26.391129032258064, 67.83266129032258, 0.6653225806451613),
                    (167.0, 159.0, 1.25),
                    (166.66666666666666, 49.166666666666664, 1.25),
                    (184.33277027027026, 77.88429054054055, 1.3175675675675675),
                    (58.40909090909091, 201.13636363636363, 0.7954545454545454),
                    (237.1875, 131.5, 1.5),
                    (145.375, 87.0, 1.125),
                    (177.14285714285714, 99.73214285714286, 1.25),
                    (151.14583333333334, 63.854166666666664, 1.1458333333333333),
                    (203.03571428571428, 86.96428571428571, 1.3392857142857142),
                    (223.4375, 111.40625, 1.40625),
                    (96.64772727272727, 94.20454545454545, 0.9090909090909091),
                    (202.1875, 127.5, 1.3125),
                    (296.4583333333333, 297.6302083333333, 1.640625),
                    (99.6875, 122.39583333333333, 0.8854166666666666),
                    (153.4375, 159.6875, 1.09375),
                    (177.5, 142.1875, 1.1875),
                    (195.6758130081301, 201.0086382113821, 1.247459349593496),
                    (248.68055555555554, 205.27777777777777, 1.4583333333333333),
                    (217.5, 227.5, 1.3392857142857142),
                    (199.16666666666666, 181.45833333333334, 1.25),
                    (368.68736673773986, 304.91337953091687, 1.8643390191897655),
                    (195.78629032258064, 162.03629032258064, 1.25),
                    (201.52717391304347, 238.125, 1.2554347826086956),
                    (149.79779411764707, 235.6433823529412, 1.0294117647058822),
                    (149.79166666666666, 238.02083333333334, 1.0416666666666667),
                    (208.33333333333334, 231.875, 1.25),
                    (229.55357142857142, 225.17857142857142, 1.3392857142857142),
                    (265.69444444444446, 233.88888888888889, 1.4583333333333333),
                    (273.2638888888889, 252.67361111111111, 1.4930555555555556),
                    (192.41071428571428, 234.6875, 1.1607142857142858),
                    (116.75, 188.05, 0.875),
                    (198.75, 234.86111111111111, 1.1805555555555556),
                    (257.34375, 222.96875, 1.40625),
                    (281.0, 254.1875, 1.5),
                    (198.8125, 222.5, 1.1875),
                    (387.0486111111111, 344.1111111111111, 1.8559027777777777),
                    (128.75, 291.6145833333333, 0.8854166666666666),
                    (387.73941532258067, 369.851310483871, 1.8472782258064515),
                    (253.22916666666666, 288.6458333333333, 1.3541666666666667),
                    (205.0, 245.51136363636363, 1.1931818181818181),
                    (79.47610294117646, 311.93014705882354, 0.6525735294117647),
                    (231.42857142857142, 260.17857142857144, 1.2946428571428572),
                    (269.21875, 281.40625, 1.40625),
                    (227.41477272727272, 220.6534090909091, 1.2784090909090908),
                    (160.73863636363637, 313.46590909090907, 0.9659090909090909),
                    (263.125, 247.25, 1.375),
                    (125.20833333333333, 232.08333333333334, 0.8333333333333334),
                    (150.3125, 235.625, 0.9375),
                    (181.51785714285714, 241.78571428571428, 1.0714285714285714),
                    (120.66964285714286, 293.5267857142857, 0.8035714285714286),
                    (130.83333333333334, 251.66666666666666, 0.8333333333333334),
                    (199.6875, 311.7708333333333, 1.1458333333333333),
                    (190.9375, 277.34375, 1.09375),
                    (108.30357142857143, 297.05357142857144, 0.7142857142857143),
                    (159.6875, 277.8125, 0.9375),
                    (123.68055555555556, 353.47222222222223, 0.7638888888888888),
                    (185.41666666666666, 306.6666666666667, 1.0416666666666667),
                    (165.3125, 308.75, 0.9375),
                    (127.875, 383.0, 0.75),
                    (174.58333333333334, 328.47222222222223, 0.9722222222222222),
                    (129.75, 291.5, 0.75),
                    (147.91666666666666, 367.0833333333333, 0.8333333333333334),
                    (202.03125, 336.5625, 1.09375),
                    (193.33333333333334, 274.7916666666667, 1.0416666666666667),
                    (143.125, 374.25, 0.75),
                    (172.76785714285714, 377.5892857142857, 0.8928571428571429),
                    (236.875, 255.625, 1.25),
                    (236.875, 349.375, 1.25),
                    (236.875, 364.375, 1.25),
                    (244.375, 370.625, 1.25),
                    (248.125, 372.1875, 1.25),
                    (248.125, 375.9375, 1.25),
                    (249.58333333333334, 374.5833333333333, 1.25),
                    (252.5, 373.75, 1.25),
                ],
                dtype=np.float64,
            ),
            # np.asarray(, dtype=np.int64),
            id="sample2",
            # marks=pytest.mark.skip(reason="development"),
        ),
    ],
)
def test_calculate_regionprops(
    sliced_labels_fixture: str,
    scaling_fixture: int | str,
    objects: int,
    expected_area: list[float],
    expected_centroid: list[tuple[float, float, float]],
    # expected_feret_diameter_max: list[float],
    request,
) -> None:
    """Test for slicer.calculate_regionprops()."""
    labeled_arrays = request.getfixturevalue(sliced_labels_fixture)
    spacing = (
        request.getfixturevalue(scaling_fixture)
        if isinstance(scaling_fixture, str)
        else scaling_fixture
    )
    region_properties = slicer.calculate_region(labeled_arrays, spacing=spacing)
    # Extract area(i.e. volume) and centroid for checking
    area = [props.area for props in region_properties]
    centroid = [region.centroid for region in region_properties]
    # feret_diameter_max = [region.feret_diameter_max for region in region_properties]
    assert len(region_properties) == objects
    np.testing.assert_allclose(
        np.asarray(area, dtype=np.float64),
        expected_area,
        atol=ABSOLUTE_TOLERANCE,
        rtol=RELATIVE_TOLERANCE,
    )
    np.testing.assert_allclose(
        np.asarray(centroid, dtype=np.float64),
        expected_centroid,
        atol=ABSOLUTE_TOLERANCE,
        rtol=RELATIVE_TOLERANCE,
    )
    # np.testing.assert_allclose(
    #       np.asarray(feret_diameter_max),
    #       expected_feret_diameter_max,
    #       atol=ABSOLUTE_TOLERANCE,
    #       rtol=RELATIVE_TOLERANCE,
    # )

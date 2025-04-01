from collections import namedtuple

import numpy as np
import pytest
import xarray as xr

from parcels.tools.converters import TimeConverter
from parcels.v4.gridadapter import GridAdapter

N = 100
T = 10

ds_2d_left = xr.Dataset(
    {
        "data_g": (["time", "ZG", "YG", "XG"], np.random.rand(T, 3 * N, 2 * N, N)),
        "data_c": (["time", "ZC", "YC", "XC"], np.random.rand(T, 3 * N, 2 * N, N)),
    },
    coords={
        "XG": (
            ["XG"],
            2 * np.pi / N * np.arange(0, N),
            {"axis": "X", "c_grid_axis_shift": -0.5},
        ),
        "XC": (["XC"], 2 * np.pi / N * (np.arange(0, N) + 0.5), {"axis": "X"}),
        "YG": (
            ["YG"],
            2 * np.pi / (2 * N) * np.arange(0, 2 * N),
            {"axis": "Y", "c_grid_axis_shift": -0.5},
        ),
        "YC": (
            ["YC"],
            2 * np.pi / (2 * N) * (np.arange(0, 2 * N) + 0.5),
            {"axis": "Y"},
        ),
        "ZG": (
            ["ZG"],
            np.arange(3 * N),
            {"axis": "Z", "c_grid_axis_shift": -0.5},
        ),
        "ZC": (
            ["ZC"],
            np.arange(3 * N) + 0.5,
            {"axis": "Z"},
        ),
        "time": (["time"], np.arange(T), {"axis": "T"}),
    },
)


TestCase = namedtuple("TestCase", ["Grid", "attr", "expected"])

test_cases = [
    TestCase(ds_2d_left, "lon", ds_2d_left.XG.values),
    TestCase(ds_2d_left, "lat", ds_2d_left.YG.values),
    TestCase(ds_2d_left, "depth", ds_2d_left.ZG.values),
    TestCase(ds_2d_left, "time", ds_2d_left.time.values),
    TestCase(ds_2d_left, "xdim", N),
    TestCase(ds_2d_left, "ydim", 2 * N),
    TestCase(ds_2d_left, "zdim", 3 * N),
    TestCase(ds_2d_left, "tdim", T),
    TestCase(ds_2d_left, "time_origin", TimeConverter(ds_2d_left.time.values[0])),
]


def assert_equal(actual, expected):
    if expected is None:
        assert actual is None
    elif isinstance(expected, TimeConverter):
        assert actual == expected
    else:
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("ds, attr, expected", test_cases)
def test_grid_adapter_properties(ds, attr, expected):
    adapter = GridAdapter(ds, periodic=False)
    actual = getattr(adapter, attr)
    assert_equal(actual, expected)

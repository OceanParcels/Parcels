from collections import namedtuple

import numpy as np
import pytest
import xarray as xr

from parcels.v4.gridadapter import GridAdapter

N = 100
T = 10

ds_2d_left = xr.Dataset(
    {
        "data_g": (["time", "YG", "XG"], np.random.rand(10, 2 * N, N)),
        "data_c": (["time", "YC", "XC"], np.random.rand(10, 2 * N, N)),
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
        "time": (["time"], np.arange(T), {"axis": "T"}),
    },
)


@pytest.fixture
def grid():
    return


TestCase = namedtuple("TestCase", ["Grid", "attr", "expected"])

test_cases = [
    # TestCase(ds_2d_left, "lon", ds_2d_left.XC.values),
    # TestCase(ds_2d_left, "lat", ds_2d_left.YC.values),
    # TestCase(ds_2d_left, "depth", None),
    # TestCase(ds_2d_left, "time", None),
    TestCase(ds_2d_left, "xdim", N),
    TestCase(ds_2d_left, "ydim", 2 * N),
    TestCase(ds_2d_left, "zdim", 1),
    TestCase(ds_2d_left, "tdim", T),
]


def assert_equal(actual, expected):
    if expected is None:
        assert actual is None
    else:
        assert np.allclose(actual, expected)


@pytest.mark.parametrize("ds, attr, expected", test_cases)
def test_grid_adapter_properties(ds, attr, expected):
    adapter = GridAdapter(ds, periodic=False)
    actual = getattr(adapter, attr)
    assert_equal(actual, expected)

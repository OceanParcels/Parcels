from collections import namedtuple

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from parcels.grid import Grid as OldGrid
from parcels.tools.converters import TimeConverter
from parcels.v4.gridadapter import GridAdapter
from tests.v4.grid_datasets import N, T, datasets

TestCase = namedtuple("TestCase", ["Grid", "attr", "expected"])

test_cases = [
    TestCase(datasets["ds_2d_left"], "lon", datasets["ds_2d_left"].XG.values),
    TestCase(datasets["ds_2d_left"], "lat", datasets["ds_2d_left"].YG.values),
    TestCase(datasets["ds_2d_left"], "depth", datasets["ds_2d_left"].ZG.values),
    TestCase(datasets["ds_2d_left"], "time", datasets["ds_2d_left"].time.values),
    TestCase(datasets["ds_2d_left"], "xdim", N),
    TestCase(datasets["ds_2d_left"], "ydim", 2 * N),
    TestCase(datasets["ds_2d_left"], "zdim", 3 * N),
    TestCase(datasets["ds_2d_left"], "tdim", T),
    TestCase(datasets["ds_2d_left"], "time_origin", TimeConverter(datasets["ds_2d_left"].time.values[0])),
]


def assert_equal(actual, expected):
    if expected is None:
        assert actual is None
    elif isinstance(expected, TimeConverter):
        assert actual == expected
    else:
        np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("ds, attr, expected", test_cases)
def test_grid_adapter_properties_ground_truth(ds, attr, expected):
    adapter = GridAdapter(ds, periodic=False)
    actual = getattr(adapter, attr)
    assert_equal(actual, expected)


@pytest.mark.parametrize("ds", datasets.values())
def test_grid_adapter_against_old(ds):
    adapter = GridAdapter(ds, periodic=False)

    grid = OldGrid.create_grid(
        lon=ds.lon.values,
        lat=ds.lat.values,
        depth=ds.depth.values,
        time=ds.time.values,
        time_origin=TimeConverter(ds.time.values[0]),
        mesh="spherical",
    )
    assert grid.lon.shape == adapter.lon.shape
    assert grid.lat.shape == adapter.lat.shape
    assert grid.depth.shape == adapter.depth.shape
    assert grid.time.shape == adapter.time.shape

    assert_array_equal(grid.lon, adapter.lon)
    assert_array_equal(grid.lat, adapter.lat)
    assert_array_equal(grid.depth, adapter.depth)
    assert_array_equal(grid.time, adapter.time)

    assert grid.time_origin == adapter.time_origin

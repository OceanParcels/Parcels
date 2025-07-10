from collections import namedtuple

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from parcels import xgcm
from parcels._datasets.structured.generic import X, Y, Z, datasets
from parcels.xgrid import XGrid, _search_1d_array

GridTestCase = namedtuple("GridTestCase", ["Grid", "attr", "expected"])

test_cases = [
    GridTestCase(datasets["ds_2d_left"], "lon", datasets["ds_2d_left"].XG.values),
    GridTestCase(datasets["ds_2d_left"], "lat", datasets["ds_2d_left"].YG.values),
    GridTestCase(datasets["ds_2d_left"], "depth", datasets["ds_2d_left"].ZG.values),
    GridTestCase(datasets["ds_2d_left"], "time", datasets["ds_2d_left"].time.values.astype(np.float64) / 1e9),
]


def assert_equal(actual, expected):
    if expected is None:
        assert actual is None
    elif isinstance(expected, np.ndarray):
        assert actual.shape == expected.shape
        assert_allclose(actual, expected)
    else:
        assert_allclose(actual, expected)


@pytest.mark.parametrize("ds, attr, expected", test_cases)
def test_xgrid_properties_ground_truth(ds, attr, expected):
    grid = XGrid(xgcm.Grid(ds, periodic=False))
    actual = getattr(grid, attr)
    assert_equal(actual, expected)


@pytest.mark.parametrize("ds", [pytest.param(ds, id=key) for key, ds in datasets.items()])
def test_xgrid_init_on_generic_datasets(ds):
    XGrid(xgcm.Grid(ds, periodic=False))


@pytest.mark.parametrize("ds", [datasets["ds_2d_left"]])
def test_xgrid_axes(ds):
    grid = XGrid(xgcm.Grid(ds, periodic=False))
    assert grid.axes == ["Z", "Y", "X"]


@pytest.mark.parametrize("ds", [datasets["ds_2d_left"]])
def test_xgrid_get_axis_dim(ds):
    grid = XGrid(xgcm.Grid(ds, periodic=False))
    assert grid.get_axis_dim("Z") == Z
    assert grid.get_axis_dim("Y") == Y
    assert grid.get_axis_dim("X") == X


def test_invalid_xgrid_field_array():
    """Stress test initialiser by creating incompatible datasets that test the edge cases"""
    ...


def test_invalid_lon_lat():
    """Stress test the grid initialiser by creating incompatible datasets that test the edge cases"""
    ds = datasets["ds_2d_left"].copy()
    ds["lon"], ds["lat"] = xr.broadcast(ds["YC"], ds["XC"])

    with pytest.raises(
        ValueError,
        match=".*is defined on the center of the grid, but must be defined on the F points\.",
    ):
        XGrid(xgcm.Grid(ds, periodic=False))

    ds = datasets["ds_2d_left"].copy()
    ds["lon"], _ = xr.broadcast(ds["YG"], ds["XG"])
    with pytest.raises(
        ValueError,
        match=".*have different dimensionalities\.",
    ):
        XGrid(xgcm.Grid(ds, periodic=False))

    ds = datasets["ds_2d_left"].copy()
    ds["lon"], ds["lat"] = xr.broadcast(ds["YG"], ds["XG"])
    ds["lon"], ds["lat"] = ds["lon"].transpose(), ds["lat"].transpose()

    with pytest.raises(
        ValueError,
        match=".*must be defined on the X and Y axes and transposed to have dimensions in order of Y, X\.",
    ):
        XGrid(xgcm.Grid(ds, periodic=False))


@pytest.mark.parametrize(
    "ds",
    [
        pytest.param(datasets["ds_2d_left"], id="1D lon/lat"),
        pytest.param(datasets["2d_left_rotated"], id="2D lon/lat"),
    ],
)  # for key, ds in datasets.items()])
def test_xgrid_search_cpoints(ds):
    grid = XGrid(xgcm.Grid(ds, periodic=False))
    lat_array, lon_array = get_2d_fpoint_mesh(grid)
    lat_array, lon_array = corner_to_cell_center_points(lat_array, lon_array)

    for xi in range(grid.xdim - 1):
        for yi in range(grid.ydim - 1):
            axis_indices = {"Z": 0, "Y": yi, "X": xi}

            lat, lon = lat_array[yi, xi], lon_array[yi, xi]
            axis_indices_bcoords = grid.search(0, lat, lon, ei=None)
            axis_indices_test = {k: v[0] for k, v in axis_indices_bcoords.items()}
            assert axis_indices == axis_indices_test

            # assert np.isclose(bcoords[0], 0.5) #? Should this not be the case with the cell center points?
            # assert np.isclose(bcoords[1], 0.5)


def get_2d_fpoint_mesh(grid: XGrid):
    lat, lon = grid.lat, grid.lon
    if lon.ndim == 1:
        lat, lon = np.meshgrid(lat, lon, indexing="ij")
    return lat, lon


def corner_to_cell_center_points(lat, lon):
    """Convert F points to C points."""
    lon_c = (lon[:-1, :-1] + lon[:-1, 1:]) / 2
    lat_c = (lat[:-1, :-1] + lat[1:, :-1]) / 2
    return lat_c, lon_c


@pytest.mark.parametrize(
    "array, x, expected_xi, expected_xsi",
    [
        (np.array([1, 2, 3, 4, 5]), 1.1, 0, 0.1),
        (np.array([1, 2, 3, 4, 5]), 2.1, 1, 0.1),
        (np.array([1, 2, 3, 4, 5]), 3.1, 2, 0.1),
        (np.array([1, 2, 3, 4, 5]), 4.5, 3, 0.5),
    ],
)
def test_search_1d_array(array, x, expected_xi, expected_xsi):
    xi, xsi = _search_1d_array(array, x)
    assert xi == expected_xi
    assert np.isclose(xsi, expected_xsi)

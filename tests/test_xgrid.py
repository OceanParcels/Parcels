import itertools
from collections import namedtuple

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from parcels._core.constants import LEFT_OUT_OF_BOUNDS, RIGHT_OUT_OF_BOUNDS
from parcels._core.utils.array import _search_1d_array
from parcels._core.xgrid import (
    XGrid,
    _transpose_xfield_data_to_tzyx,
)
from parcels._datasets.structured.generic import X, Y, Z, datasets
from tests import utils

GridTestCase = namedtuple("GridTestCase", ["ds", "attr", "expected"])

test_cases = [
    GridTestCase(datasets["ds_2d_left"], "lon", datasets["ds_2d_left"].XG.values),
    GridTestCase(datasets["ds_2d_left"], "lat", datasets["ds_2d_left"].YG.values),
    GridTestCase(datasets["ds_2d_left"], "depth", datasets["ds_2d_left"].ZG.values),
    GridTestCase(datasets["ds_2d_left"], "time", datasets["ds_2d_left"].time.values.astype(np.float64) / 1e9),
    GridTestCase(datasets["ds_2d_left"], "xdim", X - 1),
    GridTestCase(datasets["ds_2d_left"], "ydim", Y - 1),
    GridTestCase(datasets["ds_2d_left"], "zdim", Z - 1),
]


def assert_equal(actual, expected):
    if expected is None:
        assert actual is None
    elif isinstance(expected, np.ndarray):
        assert actual.shape == expected.shape
        assert_allclose(actual, expected)
    else:
        assert_allclose(actual, expected)


@pytest.mark.parametrize("ds", [datasets["ds_2d_left"]])
def test_grid_init_param_types(ds):
    with pytest.raises(ValueError, match="Invalid value 'invalid'. Valid options are.*"):
        XGrid.from_dataset(ds, mesh="invalid")


@pytest.mark.parametrize("ds, attr, expected", test_cases)
def test_xgrid_properties_ground_truth(ds, attr, expected):
    grid = XGrid.from_dataset(ds)
    actual = getattr(grid, attr)
    assert_equal(actual, expected)


@pytest.mark.parametrize("ds", [pytest.param(ds, id=key) for key, ds in datasets.items()])
def test_xgrid_from_dataset_on_generic_datasets(ds):
    XGrid.from_dataset(ds)


@pytest.mark.parametrize("ds", [datasets["ds_2d_left"]])
def test_xgrid_axes(ds):
    grid = XGrid.from_dataset(ds)
    assert grid.axes == ["Z", "Y", "X"]


@pytest.mark.parametrize("ds", [datasets["ds_2d_left"]])
def test_transpose_xfield_data_to_tzyx(ds):
    da = ds["data_g"]
    grid = XGrid.from_dataset(ds)

    all_combinations = (itertools.combinations(da.dims, n) for n in range(len(da.dims)))
    all_combinations = itertools.chain(*all_combinations)
    for subset_dims in all_combinations:
        isel = {dim: 0 for dim in subset_dims}
        da_subset = da.isel(isel, drop=True)
        da_test = _transpose_xfield_data_to_tzyx(da_subset, grid.xgcm_grid)
        utils.assert_valid_field_data(da_test, grid)


@pytest.mark.parametrize("ds", [datasets["ds_2d_left"]])
def test_xgrid_get_axis_dim(ds):
    grid = XGrid.from_dataset(ds)
    assert grid.get_axis_dim("Z") == Z - 1
    assert grid.get_axis_dim("Y") == Y - 1
    assert grid.get_axis_dim("X") == X - 1


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
        XGrid.from_dataset(ds)

    ds = datasets["ds_2d_left"].copy()
    ds["lon"], _ = xr.broadcast(ds["YG"], ds["XG"])
    with pytest.raises(
        ValueError,
        match=".*have different dimensionalities\.",
    ):
        XGrid.from_dataset(ds)

    ds = datasets["ds_2d_left"].copy()
    ds["lon"], ds["lat"] = xr.broadcast(ds["YG"], ds["XG"])
    ds["lon"], ds["lat"] = ds["lon"].transpose(), ds["lat"].transpose()

    with pytest.raises(
        ValueError,
        match=".*must be defined on the X and Y axes and transposed to have dimensions in order of Y, X\.",
    ):
        XGrid.from_dataset(ds)


@pytest.mark.parametrize(
    "ds",
    [
        pytest.param(datasets["ds_2d_left"], id="1D lon/lat"),
        pytest.param(datasets["2d_left_rotated"], id="2D lon/lat"),
    ],
)  # for key, ds in datasets.items()])
def test_xgrid_search_cpoints(ds):
    grid = XGrid.from_dataset(ds)
    lat_array, lon_array = get_2d_fpoint_mesh(grid)
    lat_array, lon_array = corner_to_cell_center_points(lat_array, lon_array)

    for xi in range(grid.xdim - 1):
        for yi in range(grid.ydim - 1):
            axis_indices = {"Z": 0, "Y": yi, "X": xi}

            lat, lon = lat_array[yi, xi], lon_array[yi, xi]
            axis_indices_bcoords = grid.search(0, np.atleast_1d(lat), np.atleast_1d(lon), ei=None)
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
        (np.array([1, 2, 3, 4, 5]), (1.1, 2.1), (0, 1), (0.1, 0.1)),
        (np.array([1, 2, 3, 4, 5]), 2.1, 1, 0.1),
        (np.array([1, 2, 3, 4, 5]), 3.1, 2, 0.1),
        (np.array([1, 2, 3, 4, 5]), 4.5, 3, 0.5),
    ],
)
def test_search_1d_array(array, x, expected_xi, expected_xsi):
    xi, xsi = _search_1d_array(array, x)
    np.testing.assert_array_equal(xi, expected_xi)
    np.testing.assert_allclose(xsi, expected_xsi)


@pytest.mark.parametrize(
    "array, x, expected_xi",
    [
        (np.array([1, 2, 3, 4, 5]), -0.1, LEFT_OUT_OF_BOUNDS),
        (np.array([1, 2, 3, 4, 5]), 6.5, RIGHT_OUT_OF_BOUNDS),
    ],
)
def test_search_1d_array_out_of_bounds(array, x, expected_xi):
    xi, xsi = _search_1d_array(array, x)
    assert xi == expected_xi


@pytest.mark.parametrize(
    "array, x, expected_xi",
    [
        (np.array([1, 2, 3, 4, 5]), (-0.1, 2.5), (LEFT_OUT_OF_BOUNDS, 1)),
        (np.array([1, 2, 3, 4, 5]), (6.5, 1), (RIGHT_OUT_OF_BOUNDS, 0)),
    ],
)
def test_search_1d_array_some_out_of_bounds(array, x, expected_xi):
    xi, _ = _search_1d_array(array, x)
    np.testing.assert_array_equal(xi, expected_xi)


@pytest.mark.parametrize(
    "ds, da_name, expected",
    [
        pytest.param(
            datasets["ds_2d_left"],
            "U (C grid)",
            {
                "XG": (np.int64(0), np.float64(0.0)),
                "YC": (np.int64(-1), np.float64(0.5)),
                "ZG": (np.int64(0), np.float64(0.0)),
            },
            id="MITgcm indexing style U (C grid)",
        ),
        pytest.param(
            datasets["ds_2d_left"],
            "V (C grid)",
            {
                "XC": (np.int64(-1), np.float64(0.5)),
                "YG": (np.int64(0), np.float64(0.0)),
                "ZG": (np.int64(0), np.float64(0.0)),
            },
            id="MITgcm indexing style V (C grid)",
        ),
        pytest.param(
            datasets["ds_2d_right"],
            "U (C grid)",
            {
                "XG": (np.int64(0), np.float64(0.0)),
                "YC": (np.int64(0), np.float64(0.5)),
                "ZG": (np.int64(0), np.float64(0.0)),
            },
            id="NEMO indexing style U (C grid)",
        ),
        pytest.param(
            datasets["ds_2d_right"],
            "V (C grid)",
            {
                "XC": (np.int64(0), np.float64(0.5)),
                "YG": (np.int64(0), np.float64(0.0)),
                "ZG": (np.int64(0), np.float64(0.0)),
            },
            id="NEMO indexing style V (C grid)",
        ),
    ],
)
def test_xgrid_localize_zero_position(ds, da_name, expected):
    """Test localize function using left and right datasets."""
    grid = XGrid.from_dataset(ds)
    da = ds[da_name]
    position = grid.search(0, 0, 0)

    local_position = grid.localize(position, da.dims)
    assert local_position == expected, f"Expected {expected}, got {local_position}"

from collections import namedtuple
from typing import Literal

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from parcels import xgcm
from parcels._datasets.structured.generic import T, X, Y, Z, datasets
from parcels.grid import Grid as OldGrid
from parcels.tools.converters import TimeConverter
from parcels.xgrid import XGrid, _generate_cells, _search_1d_array

GridTestCase = namedtuple("GridTestCase", ["Grid", "attr", "expected"])

test_cases = [
    GridTestCase(datasets["ds_2d_left"], "lon", datasets["ds_2d_left"].XG.values),
    GridTestCase(datasets["ds_2d_left"], "lat", datasets["ds_2d_left"].YG.values),
    GridTestCase(datasets["ds_2d_left"], "depth", datasets["ds_2d_left"].ZG.values),
    GridTestCase(datasets["ds_2d_left"], "time", datasets["ds_2d_left"].time.values.astype(np.float64) / 1e9),
    GridTestCase(datasets["ds_2d_left"], "xdim", X),
    GridTestCase(datasets["ds_2d_left"], "ydim", Y),
    GridTestCase(datasets["ds_2d_left"], "zdim", Z),
    GridTestCase(datasets["ds_2d_left"], "tdim", T),
    GridTestCase(datasets["ds_2d_left"], "time_origin", TimeConverter(datasets["ds_2d_left"].time.values[0])),
]


def assert_equal(actual, expected):
    if expected is None:
        assert actual is None
    elif isinstance(expected, TimeConverter):
        assert actual == expected
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


@pytest.mark.parametrize(
    "attr",
    [
        "lon",
        "lat",
        "depth",
        "time",
        "xdim",
        "ydim",
        "zdim",
        "tdim",
        "time_origin",
        "_gtype",
    ],
)
@pytest.mark.parametrize("ds", [pytest.param(ds, id=key) for key, ds in datasets.items()])
def test_xgrid_against_old(ds, attr):
    grid = XGrid(xgcm.Grid(ds, periodic=False))

    old_grid = OldGrid.create_grid(
        lon=ds.lon.values,
        lat=ds.lat.values,
        depth=ds.depth.values,
        time=ds.time.values.astype("float64") / 1e9,
        time_origin=TimeConverter(ds.time.values[0]),
        mesh="spherical",
    )
    actual = getattr(grid, attr)
    expected = getattr(old_grid, attr)
    assert_equal(actual, expected)


@pytest.mark.parametrize("ds", [pytest.param(ds, id=key) for key, ds in datasets.items()])
def test_grid_init_on_generic_datasets(ds):
    XGrid(xgcm.Grid(ds, periodic=False))


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


def test_xgrid_ravel_unravel_index():
    ds = datasets["ds_2d_left"]
    grid = XGrid(xgcm.Grid(ds, periodic=False))

    xdim = grid.xdim
    ydim = grid.ydim
    zdim = grid.zdim

    encountered_eis = []
    for xi in range(xdim):
        for yi in range(ydim):
            for zi in range(zdim):
                ei = grid.ravel_index(zi, yi, xi)
                zi_test, yi_test, xi_test = grid.unravel_index(ei)
                assert xi == xi_test, f"Expected xi {xi} but got {xi_test} for ei {ei}"
                assert yi == yi_test, f"Expected yi {yi} but got {yi_test} for ei {ei}"
                assert zi == zi_test, f"Expected zi {zi} but got {zi_test} for ei {ei}"
                encountered_eis.append(ei)

    encountered_eis = sorted(encountered_eis)
    assert len(set(encountered_eis)) == len(encountered_eis), "Raveled indices are not unique."
    assert np.allclose(np.diff(np.array(encountered_eis)), 1), "Raveled indices are not consecutive integers."
    assert encountered_eis[0] == 0, "Raveled indices do not start at 0."


def test_generate_cells():
    ydim = 3  # Number of cells in the y-direction
    xdim = 6  # Number of cells in the x-direction
    lon = np.arange(ydim + 1)
    lat = np.arange(xdim + 1)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")

    # Call the function and collect the output
    cells = list(_generate_cells(lat=LAT, lon=LON))
    assert len(cells) == ydim * xdim, "Number of cells does not match expected."

    for cell in cells:
        _assert_point_is("east", 1, cell[0], cell[1])
        _assert_point_is("north", 1, cell[1], cell[2])
        _assert_point_is("west", 1, cell[2], cell[3])


def test__assert_point_is():
    _assert_point_is("east", 1, np.array([0, 0]), np.array([0, 1]))
    _assert_point_is("west", 1, np.array([0, 1]), np.array([0, 0]))
    _assert_point_is("north", 1, np.array([0, 0]), np.array([1, 0]))
    _assert_point_is("south", 1, np.array([1, 0]), np.array([0, 0]))


def _assert_point_is(
    direction: Literal["east", "west", "north", "south"], by: int, reference_cell: np.ndarray, test_cell: np.ndarray
):
    """cell1 and cell2 are arrays of (lat, lon)"""
    match direction:
        case "east":
            delta = np.array([0, by])
        case "west":
            delta = np.array([0, -by])
        case "north":
            delta = np.array([by, 0])
        case "south":
            delta = np.array([-by, 0])
        case _:
            raise ValueError(f"Invalid method: {direction}")

    np.testing.assert_allclose(reference_cell + delta, test_cell)


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
            lat, lon = lat_array[yi, xi], lon_array[yi, xi]
            (zi_test, yi_test, xi_test), bcoords = grid.search(0, lat, lon, ei=None, search2D=True)
            assert xi == xi_test
            assert yi == yi_test
            assert zi_test == 0

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

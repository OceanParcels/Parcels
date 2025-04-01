import numpy as np
import pytest
import xarray as xr

from parcels.gridv4 import Axis, Grid

from tests.v4.datasets import all_2d  # noqa: F401
from tests.v4.datasets import all_datasets  # noqa: F401
from tests.v4.datasets import datasets  # noqa: F401
from tests.v4.datasets import datasets_grid_metric  # noqa: F401
from tests.v4.datasets import nonperiodic_1d  # noqa: F401
from tests.v4.datasets import nonperiodic_2d  # noqa: F401
from tests.v4.datasets import periodic_1d  # noqa: F401
from tests.v4.datasets import periodic_2d  # noqa: F401


# helper function to produce axes from datasets
def _get_axes(ds):
    all_axes = {ds[c].attrs["axis"] for c in ds.dims if "axis" in ds[c].attrs}
    axis_objs = {ax: Axis(ds, ax) for ax in all_axes}
    return axis_objs


def test_create_axis(all_datasets):
    ds, periodic, expected = all_datasets
    axis_objs = _get_axes(ds)
    for ax_expected, coords_expected in expected["axes"].items():
        assert ax_expected in axis_objs
        this_axis = axis_objs[ax_expected]
        for axis_name, coord_name in coords_expected.items():
            assert axis_name in this_axis.coords
            assert this_axis.coords[axis_name] == coord_name


def _assert_axes_equal(ax1, ax2):
    assert ax1.name == ax2.name
    for pos, coord in ax1.coords.items():
        assert pos in ax2.coords
        assert coord == ax2.coords[pos]
    assert ax1._periodic == ax2._periodic
    assert ax1._default_shifts == ax2._default_shifts
    assert ax1._facedim == ax2._facedim
    # TODO: make this work...
    # assert ax1._connections == ax2._connections


def test_create_axis_no_comodo(all_datasets):
    ds, periodic, expected = all_datasets
    axis_objs = _get_axes(ds)

    # now strip out the metadata
    ds_noattr = ds.copy()
    for var in ds.variables:
        ds_noattr[var].attrs.clear()

    for axis_name, axis_coords in expected["axes"].items():
        # now create the axis from scratch with no attributes
        ax2 = Axis(ds_noattr, axis_name, coords=axis_coords)
        # and compare to the one created with attributes
        ax1 = axis_objs[axis_name]

        assert ax1.name == ax2.name
        for pos, coord_name in ax1.coords.items():
            assert pos in ax2.coords
            assert coord_name == ax2.coords[pos]
        assert ax1._periodic == ax2._periodic
        assert ax1._default_shifts == ax2._default_shifts
        assert ax1._facedim == ax2._facedim


def test_create_axis_no_coords(all_datasets):
    ds, periodic, expected = all_datasets
    axis_objs = _get_axes(ds)

    ds_drop = ds.drop_vars(list(ds.coords))

    for axis_name, axis_coords in expected["axes"].items():
        # now create the axis from scratch with no attributes OR coords
        ax2 = Axis(ds_drop, axis_name, coords=axis_coords)
        # and compare to the one created with attributes
        ax1 = axis_objs[axis_name]

        assert ax1.name == ax2.name
        for pos, coord in ax1.coords.items():
            assert pos in ax2.coords
        assert ax1._periodic == ax2._periodic
        assert ax1._default_shifts == ax2._default_shifts
        assert ax1._facedim == ax2._facedim


def test_axis_repr(all_datasets):
    ds, periodic, expected = all_datasets
    axis_objs = _get_axes(ds)
    for ax_name, axis in axis_objs.items():
        r = repr(axis).split("\n")
        assert r[0].startswith("<xgcm.Axis")
    # TODO: make this more complete


def test_get_position_name(all_datasets):
    ds, periodic, expected = all_datasets
    axis_objs = _get_axes(ds)
    for ax_name, axis in axis_objs.items():
        # create a dataarray with each axis coordinate
        for position, coord in axis.coords.items():
            da = 1 * ds[coord]
            assert axis._get_position_name(da) == (position, coord)


# helper functions for padding arrays
# this feels silly...I'm basically just re-coding the function in order to
# test it
def _pad_left(data, boundary, fill_value=0.0):
    pad_val = data[0] if boundary == "extend" else fill_value
    return np.hstack([pad_val, data])


def _pad_right(data, boundary, fill_value=0.0):
    pad_val = data[-1] if boundary == "extend" else fill_value
    return np.hstack([data, pad_val])


def test_axis_errors():
    ds = datasets["1d_left"]

    ds_noattr = ds.copy()
    del ds_noattr.XC.attrs["axis"]
    with pytest.raises(ValueError, match="Couldn't find a center coordinate for axis X"):
        _ = Axis(ds_noattr, "X", periodic=True)

    del ds_noattr.XG.attrs["axis"]
    with pytest.raises(ValueError, match="Couldn't find any coordinates for axis X"):
        _ = Axis(ds_noattr, "X", periodic=True)

    ds_chopped = ds.copy().isel(XG=slice(None, 3))
    del ds_chopped["data_g"]
    with pytest.raises(ValueError, match="coordinate XG has incompatible length"):
        _ = Axis(ds_chopped, "X", periodic=True)

    ds_chopped.XG.attrs["c_grid_axis_shift"] = -0.5
    with pytest.raises(ValueError, match="coordinate XG has incompatible length"):
        _ = Axis(ds_chopped, "X", periodic=True)

    del ds_chopped.XG.attrs["c_grid_axis_shift"]
    with pytest.raises(
        ValueError,
        match="Found two coordinates without `c_grid_axis_shift` attribute for axis X",
    ):
        _ = Axis(ds_chopped, "X", periodic=True)

    # This case is broken, need to fix!
    # with pytest.raises(
    #    ValueError, match="`boundary=fill` is not allowed " "with periodic axis X."
    # ):
    #    ax.interp(ds.data_c, "left", boundary="fill")


@pytest.mark.parametrize(
    "boundary",
    [
        None,
        "fill",
        "extend",
        pytest.param("extrapolate", marks=pytest.mark.xfail(strict=True)),
        {"X": "fill", "Y": "extend"},
    ],
)
@pytest.mark.parametrize("fill_value", [None, 0, 1.0])
def test_grid_create(all_datasets, boundary, fill_value):
    ds, periodic, expected = all_datasets
    grid = Grid(ds, periodic=periodic)
    assert grid is not None
    for ax in grid.axes.values():
        assert ax.boundary is None
    grid = Grid(ds, periodic=periodic, boundary=boundary, fill_value=fill_value)
    for name, ax in grid.axes.items():
        if isinstance(boundary, dict):
            expected = boundary.get(name)
        else:
            expected = boundary
        assert ax.boundary == expected

        if fill_value is None:
            expected = 0.0
        elif isinstance(fill_value, dict):
            expected = fill_value.get(name)
        else:
            expected = fill_value
        assert ax.fill_value == expected


def test_create_grid_no_comodo(all_datasets):
    ds, periodic, expected = all_datasets
    grid_expected = Grid(ds, periodic=periodic)

    ds_noattr = ds.copy()
    for var in ds.variables:
        ds_noattr[var].attrs.clear()

    coords = expected["axes"]
    grid = Grid(ds_noattr, periodic=periodic, coords=coords)

    for axis_name_expected in grid_expected.axes:
        axis_expected = grid_expected.axes[axis_name_expected]
        axis_actual = grid.axes[axis_name_expected]
        _assert_axes_equal(axis_expected, axis_actual)


def test_grid_no_coords(periodic_1d):
    """Ensure that you can use Grid with Xarray datasets that don't have dimension coordinates."""
    ds, periodic, expected = periodic_1d
    ds_nocoords = ds.drop_vars(list(ds.dims.keys()))

    coords = expected["axes"]
    Grid(ds_nocoords, periodic=periodic, coords=coords)


def test_grid_repr(all_datasets):
    ds, periodic, _ = all_datasets
    grid = Grid(ds, periodic=periodic)
    r = repr(grid).split("\n")
    assert r[0] == "<xgcm.Grid>"


def test_grid_dict_input_boundary_fill(nonperiodic_1d):
    """Test axis kwarg input functionality using dict input"""
    ds, _, _ = nonperiodic_1d
    grid_direct = Grid(ds, periodic=False, boundary="fill", fill_value=5)
    grid_dict = Grid(ds, periodic=False, boundary={"X": "fill"}, fill_value={"X": 5})
    assert grid_direct.axes["X"].fill_value == grid_dict.axes["X"].fill_value
    assert grid_direct.axes["X"].boundary == grid_dict.axes["X"].boundary


def test_invalid_boundary_error():
    ds = datasets["1d_left"]
    with pytest.raises(ValueError):
        Axis(ds, "X", boundary="bad")
    with pytest.raises(ValueError):
        Grid(ds, boundary="bad")
    with pytest.raises(ValueError):
        Grid(ds, boundary={"X": "bad"})
    with pytest.raises(ValueError):
        Grid(ds, boundary={"X": 0})
    with pytest.raises(ValueError):
        Grid(ds, boundary=0)


def test_invalid_fill_value_error():
    ds = datasets["1d_left"]
    with pytest.raises(ValueError):
        Axis(ds, "X", fill_value="x")
    with pytest.raises(ValueError):
        Grid(ds, fill_value="bad")
    with pytest.raises(ValueError):
        Grid(ds, fill_value={"X": "bad"})


def test_input_not_dims():
    data = np.random.rand(4, 5)
    coord = np.random.rand(4, 5)
    ds = xr.DataArray(data, dims=["x", "y"], coords={"c": (["x", "y"], coord)}).to_dataset(name="data")
    msg = r"is not a dimension in the input dataset"
    with pytest.raises(ValueError, match=msg):
        Grid(ds, coords={"X": {"center": "c"}})


def test_input_dim_notfound():
    data = np.random.rand(4, 5)
    coord = np.random.rand(4, 5)
    ds = xr.DataArray(data, dims=["x", "y"], coords={"c": (["x", "y"], coord)}).to_dataset(name="data")
    msg = r"Could not find dimension `other` \(for the `center` position on axis `X`\) in input dataset."
    with pytest.raises(ValueError, match=msg):
        Grid(ds, coords={"X": {"center": "other"}})

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


def test_axis_wrap_and_replace_2d(periodic_2d):
    ds, periodic, expected = periodic_2d
    axis_objs = _get_axes(ds)

    da_xc_yc = 0 * ds.XC * ds.YC + 1
    da_xc_yg = 0 * ds.XC * ds.YG + 1
    da_xg_yc = 0 * ds.XG * ds.YC + 1

    da_xc_yg_test = axis_objs["Y"]._wrap_and_replace_coords(da_xc_yc, da_xc_yc.data, "left")
    assert da_xc_yg.equals(da_xc_yg_test)

    da_xg_yc_test = axis_objs["X"]._wrap_and_replace_coords(da_xc_yc, da_xc_yc.data, "left")
    assert da_xg_yc.equals(da_xg_yc_test)


def test_axis_wrap_and_replace_nonperiodic(nonperiodic_1d):
    ds, periodic, expected = nonperiodic_1d
    axis = Axis(ds, "X")

    da_c = 0 * ds.XC + 1
    da_g = 0 * ds.XG + 1

    to = (set(expected["axes"]["X"].keys()) - {"center"}).pop()

    da_g_test = axis._wrap_and_replace_coords(da_c, da_g.data, to)
    assert da_g.equals(da_g_test)

    da_c_test = axis._wrap_and_replace_coords(da_g, da_c.data, "center")
    assert da_c.equals(da_c_test)


# helper functions for padding arrays
# this feels silly...I'm basically just re-coding the function in order to
# test it
def _pad_left(data, boundary, fill_value=0.0):
    pad_val = data[0] if boundary == "extend" else fill_value
    return np.hstack([pad_val, data])


def _pad_right(data, boundary, fill_value=0.0):
    pad_val = data[-1] if boundary == "extend" else fill_value
    return np.hstack([data, pad_val])


@pytest.mark.parametrize(
    "boundary",
    [None, "extend", "fill", pytest.param("extrapolate", marks=pytest.mark.xfail)],
)
@pytest.mark.parametrize("from_center", [True, False])
def test_axis_neighbor_pairs_nonperiodic_1d(nonperiodic_1d, boundary, from_center):
    ds, periodic, expected = nonperiodic_1d
    axis = Axis(ds, "X", periodic=periodic)

    # detect whether this is an outer or inner case
    # outer --> dim_line_diff = 1
    # inner --> dim_line_diff = -1
    dim_len_diff = len(ds.XG) - len(ds.XC)

    if from_center:
        to = (set(expected["axes"]["X"].keys()) - {"center"}).pop()
        da = ds.data_c
    else:
        to = "center"
        da = ds.data_g

    shift = expected.get("shift") or False

    # need boundary condition for everything but outer to center
    if (boundary is None) and (
        dim_len_diff == 0 or (dim_len_diff == 1 and from_center) or (dim_len_diff == -1 and not from_center)
    ):
        with pytest.raises(ValueError):
            data_left, data_right = axis._get_neighbor_data_pairs(da, to, boundary=boundary)
    else:
        data_left, data_right = axis._get_neighbor_data_pairs(da, to, boundary=boundary)
        if ((dim_len_diff == 1) and not from_center) or ((dim_len_diff == -1) and from_center):
            expected_left = da.data[:-1]
            expected_right = da.data[1:]
        elif ((dim_len_diff == 1) and from_center) or ((dim_len_diff == -1) and not from_center):
            expected_left = _pad_left(da.data, boundary)
            expected_right = _pad_right(da.data, boundary)
        elif (shift and not from_center) or (not shift and from_center):
            expected_right = da.data
            expected_left = _pad_left(da.data, boundary)[:-1]
        else:
            expected_left = da.data
            expected_right = _pad_right(da.data, boundary)[1:]

        np.testing.assert_allclose(data_left, expected_left)
        np.testing.assert_allclose(data_right, expected_right)


@pytest.mark.parametrize("boundary", ["extend", "fill", pytest.param("extrapolate", marks=pytest.mark.xfail)])
def test_axis_cumsum(nonperiodic_1d, boundary):
    ds, periodic, expected = nonperiodic_1d
    axis = Axis(ds, "X", periodic=periodic)

    axis_expected = expected["axes"]["X"]

    cumsum_g = axis.cumsum(ds.data_g, to="center", boundary=boundary)
    assert cumsum_g.dims == ds.data_c.dims
    # check default "to"
    assert cumsum_g.equals(axis.cumsum(ds.data_g, boundary=boundary))

    to = set(axis_expected).difference({"center"}).pop()
    cumsum_c = axis.cumsum(ds.data_c, to=to, boundary=boundary)
    assert cumsum_c.dims == ds.data_g.dims
    # check default "to"
    assert cumsum_c.equals(axis.cumsum(ds.data_c, boundary=boundary))

    cumsum_c_raw = np.cumsum(ds.data_c.data)
    cumsum_g_raw = np.cumsum(ds.data_g.data)

    if to == "right":
        np.testing.assert_allclose(cumsum_c.data, cumsum_c_raw)
        fill_value = 0.0 if boundary == "fill" else cumsum_g_raw[0]
        np.testing.assert_allclose(cumsum_g.data, np.hstack([fill_value, cumsum_g_raw[:-1]]))
    elif to == "left":
        np.testing.assert_allclose(cumsum_g.data, cumsum_g_raw)
        fill_value = 0.0 if boundary == "fill" else cumsum_c_raw[0]
        np.testing.assert_allclose(cumsum_c.data, np.hstack([fill_value, cumsum_c_raw[:-1]]))
    elif to == "inner":
        np.testing.assert_allclose(cumsum_c.data, cumsum_c_raw[:-1])
        fill_value = 0.0 if boundary == "fill" else cumsum_g_raw[0]
        np.testing.assert_allclose(cumsum_g.data, np.hstack([fill_value, cumsum_g_raw]))
    elif to == "outer":
        np.testing.assert_allclose(cumsum_g.data, cumsum_g_raw[:-1])
        fill_value = 0.0 if boundary == "fill" else cumsum_c_raw[0]
        np.testing.assert_allclose(cumsum_c.data, np.hstack([fill_value, cumsum_c_raw]))

    # not much point doing this...we don't have the right test datasets
    # to really test the errors
    # other_positions = {'left', 'right', 'inner', 'outer'}.difference({to})
    # for pos in other_positions:
    #     with pytest.raises(KeyError):
    #         axis.cumsum(ds.data_c, to=pos, boundary=boundary)


@pytest.mark.parametrize(
    "varname, axis_name, to, roll, roll_axis, swap_order",
    [
        ("data_c", "X", "left", 1, 1, False),
        ("data_c", "Y", "left", 1, 0, False),
        ("data_g", "X", "center", -1, 1, True),
        ("data_g", "Y", "center", -1, 0, True),
    ],
)
def test_axis_neighbor_pairs_2d(periodic_2d, varname, axis_name, to, roll, roll_axis, swap_order):
    ds, _, _ = periodic_2d

    axis = Axis(ds, axis_name)

    data = ds[varname]
    data_left, data_right = axis._get_neighbor_data_pairs(data, to)
    if swap_order:
        data_left, data_right = data_right, data_left
    np.testing.assert_allclose(data_left, np.roll(data.data, roll, axis=roll_axis))
    np.testing.assert_allclose(data_right, data.data)


@pytest.mark.parametrize("boundary", ["extend", "fill", pytest.param("extrapolate", marks=pytest.mark.xfail)])
@pytest.mark.parametrize("from_center", [True, False])
def test_axis_diff_and_interp_nonperiodic_1d(nonperiodic_1d, boundary, from_center):
    ds, periodic, expected = nonperiodic_1d
    axis = Axis(ds, "X", periodic=periodic)

    dim_len_diff = len(ds.XG) - len(ds.XC)

    if from_center:
        to = (set(expected["axes"]["X"].keys()) - {"center"}).pop()
        coord_to = "XG"
        da = ds.data_c
    else:
        to = "center"
        coord_to = "XC"
        da = ds.data_g

    shift = expected.get("shift") or False

    data = da.data
    if (dim_len_diff == 1 and not from_center) or (dim_len_diff == -1 and from_center):
        data_left = data[:-1]
        data_right = data[1:]
    elif (dim_len_diff == 1 and from_center) or (dim_len_diff == -1 and not from_center):
        data_left = _pad_left(data, boundary)
        data_right = _pad_right(data, boundary)
    elif (shift and not from_center) or (not shift and from_center):
        data_left = _pad_left(data[:-1], boundary)
        data_right = data
    else:
        data_left = data
        data_right = _pad_right(data[1:], boundary)

    # interpolate
    data_interp_expected = xr.DataArray(
        0.5 * (data_left + data_right), dims=[coord_to], coords={coord_to: ds[coord_to]}
    )
    data_interp = axis.interp(da, to, boundary=boundary)
    assert data_interp_expected.equals(data_interp)
    # check without "to" specified
    assert data_interp.equals(axis.interp(da, boundary=boundary))

    # difference
    data_diff_expected = xr.DataArray(data_right - data_left, dims=[coord_to], coords={coord_to: ds[coord_to]})
    data_diff = axis.diff(da, to, boundary=boundary)
    assert data_diff_expected.equals(data_diff)
    # check without "to" specified
    assert data_diff.equals(axis.diff(da, boundary=boundary))

    # max
    data_max_expected = xr.DataArray(
        np.maximum(data_right, data_left),
        dims=[coord_to],
        coords={coord_to: ds[coord_to]},
    )
    data_max = axis.max(da, to, boundary=boundary)
    assert data_max_expected.equals(data_max)
    # check without "to" specified
    assert data_max.equals(axis.max(da, boundary=boundary))

    # min
    data_min_expected = xr.DataArray(
        np.minimum(data_right, data_left),
        dims=[coord_to],
        coords={coord_to: ds[coord_to]},
    )
    data_min = axis.min(da, to, boundary=boundary)
    assert data_min_expected.equals(data_min)
    # check without "to" specified
    assert data_min.equals(axis.min(da, boundary=boundary))


# this mega test covers all options for 2D data


@pytest.mark.parametrize("boundary", ["extend", "fill", pytest.param("extrapolate", marks=pytest.mark.xfail)])
@pytest.mark.parametrize("axis_name", ["X", "Y"])
@pytest.mark.parametrize("varname, this, to", [("data_c", "center", "left"), ("data_g", "left", "center")])
def test_axis_diff_and_interp_nonperiodic_2d(all_2d, boundary, axis_name, varname, this, to):
    ds, periodic, _ = all_2d

    try:
        ax_periodic = axis_name in periodic
    except TypeError:
        ax_periodic = periodic

    boundary_arg = boundary if not ax_periodic else None
    axis = Axis(ds, axis_name, periodic=ax_periodic, boundary=boundary_arg)
    da = ds[varname]

    # everything is left shift
    data = ds[varname].data

    axis_num = da.get_axis_num(axis.coords[this])

    # lookups for numpy.pad
    numpy_pad_arg = {"extend": "edge", "fill": "constant"}
    # args for numpy.pad
    pad_left = (1, 0)
    pad_right = (0, 1)
    pad_none = (0, 0)

    if this == "center":
        if ax_periodic:
            data_left = np.roll(data, 1, axis=axis_num)
        else:
            pad_width = [pad_left if i == axis_num else pad_none for i in range(data.ndim)]
            the_slice = tuple([slice(0, -1) if i == axis_num else slice(None) for i in range(data.ndim)])
            data_left = np.pad(data, pad_width, numpy_pad_arg[boundary])[the_slice]
        data_right = data
    elif this == "left":
        if ax_periodic:
            data_left = data
            data_right = np.roll(data, -1, axis=axis_num)
        else:
            pad_width = [pad_right if i == axis_num else pad_none for i in range(data.ndim)]
            the_slice = tuple([slice(1, None) if i == axis_num else slice(None) for i in range(data.ndim)])
            data_right = np.pad(data, pad_width, numpy_pad_arg[boundary])[the_slice]
            data_left = data

    data_interp = 0.5 * (data_left + data_right)
    data_diff = data_right - data_left

    # determine new dims
    dims = list(da.dims)
    dims[axis_num] = axis.coords[to]
    coords = {dim: ds[dim] for dim in dims}

    da_interp_expected = xr.DataArray(data_interp, dims=dims, coords=coords)
    da_diff_expected = xr.DataArray(data_diff, dims=dims, coords=coords)

    da_interp = axis.interp(da, to)
    da_diff = axis.diff(da, to)

    assert da_interp_expected.equals(da_interp)
    assert da_diff_expected.equals(da_diff)

    if boundary_arg is not None:
        if boundary == "extend":
            bad_boundary = "fill"
        elif boundary == "fill":
            bad_boundary = "extend"

        da_interp_wrong = axis.interp(da, to, boundary=bad_boundary)
        assert not da_interp_expected.equals(da_interp_wrong)
        da_diff_wrong = axis.diff(da, to, boundary=bad_boundary)
        assert not da_diff_expected.equals(da_diff_wrong)


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

    ax = Axis(ds, "X", periodic=True)

    with pytest.raises(ValueError, match="Can't get neighbor pairs for the same position."):
        ax.interp(ds.data_c, "center")

    with pytest.raises(ValueError, match="This axis doesn't contain a `right` position"):
        ax.interp(ds.data_c, "right")

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
    """Ensure that you can use xgcm with Xarray datasets that don't have dimension coordinates."""
    ds, periodic, expected = periodic_1d
    ds_nocoords = ds.drop_vars(list(ds.dims.keys()))

    coords = expected["axes"]
    grid = Grid(ds_nocoords, periodic=periodic, coords=coords)

    diff = grid.diff(ds["data_c"], "X")
    assert len(diff.coords) == 0
    interp = grid.interp(ds["data_c"], "X")
    assert len(interp.coords) == 0


def test_grid_repr(all_datasets):
    ds, periodic, _ = all_datasets
    grid = Grid(ds, periodic=periodic)
    r = repr(grid).split("\n")
    assert r[0] == "<xgcm.Grid>"


def test_grid_ops(all_datasets):
    """
    Check that we get the same answer using Axis or Grid objects
    """
    ds, periodic, _ = all_datasets
    grid = Grid(ds, periodic=periodic)

    for axis_name in grid.axes.keys():
        try:
            ax_periodic = axis_name in periodic
        except TypeError:
            ax_periodic = periodic
        axis = Axis(ds, axis_name, periodic=ax_periodic)

        bcs = [None] if ax_periodic else ["fill", "extend"]
        for varname in ["data_c", "data_g"]:
            for boundary in bcs:
                da_interp = grid.interp(ds[varname], axis_name, boundary=boundary)
                da_interp_ax = axis.interp(ds[varname], boundary=boundary)
                assert da_interp.equals(da_interp_ax)

                da_diff = grid.diff(ds[varname], axis_name, boundary=boundary)
                da_diff_ax = axis.diff(ds[varname], boundary=boundary)
                assert da_diff.equals(da_diff_ax)

                if boundary is not None:
                    da_cumsum = grid.cumsum(ds[varname], axis_name, boundary=boundary)
                    da_cumsum_ax = axis.cumsum(ds[varname], boundary=boundary)
                    assert da_cumsum.equals(da_cumsum_ax)


@pytest.mark.parametrize("func", ["interp", "max", "min", "diff", "cumsum"])
@pytest.mark.parametrize("periodic", ["True", "False", ["X"], ["Y"], ["X", "Y"]])
@pytest.mark.parametrize(
    "boundary",
    [
        "fill",
        pytest.param("extrapolate", marks=pytest.mark.xfail(strict=True)),
        "extend",
        {"X": "fill", "Y": "extend"},
        {"X": "extend", "Y": "fill"},
    ],
)
def test_multi_axis_input(all_datasets, func, periodic, boundary):
    ds, periodic_unused, expected_unused = all_datasets
    grid = Grid(ds, periodic=periodic)
    axes = list(grid.axes.keys())
    for varname in ["data_c", "data_g"]:
        serial = ds[varname]
        for axis in axes:
            boundary_axis = boundary
            if isinstance(boundary, dict):
                boundary_axis = boundary[axis]
            serial = getattr(grid, func)(serial, axis, boundary=boundary_axis)
        full = getattr(grid, func)(ds[varname], axes, boundary=boundary)
        xr.testing.assert_allclose(serial, full)


@pytest.mark.parametrize(
    "func",
    ["interp", "max", "min", "diff", "cumsum"],
)
@pytest.mark.parametrize(
    "boundary",
    [
        "fill",
        pytest.param("extrapolate", marks=pytest.mark.xfail),
        "extend",
        {"X": "fill", "Y": "extend"},
        {"X": "extend", "Y": "fill"},
    ],
)
def test_dask_vs_eager(all_datasets, func, boundary):
    ds, coords, metrics = datasets_grid_metric("C")
    grid = Grid(ds, coords=coords)
    grid_method = getattr(grid, func)
    eager_result = grid_method(ds.tracer, "X", boundary=boundary)

    ds = ds.chunk({"xt": 1, "yt": 1, "time": 1, "zt": 1})
    grid = Grid(ds, coords=coords)
    grid_method = getattr(grid, func)
    dask_result = grid_method(ds.tracer, "X", boundary=boundary).compute()

    xr.testing.assert_allclose(dask_result, eager_result)


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


@pytest.mark.parametrize(
    "funcname",
    [
        "diff",
        "interp",
        "min",
        "max",
        "integrate",
        "average",
        "cumsum",
        "cumint",
        "derivative",
        # TODO: we can get rid of many of these after the release. With the grid_ufunc logic many of these go through the same codepath
        # e.g. diff/interp/min/max all are the same, so we can probably reduce this to diff, cumsum, integrate, derivative, cumint
    ],
)
@pytest.mark.parametrize("gridtype", ["B", "C"])
def test_keep_coords(funcname, gridtype):
    ds, coords, metrics = datasets_grid_metric(gridtype)
    ds = ds.assign_coords(yt_bis=ds["yt"], xt_bis=ds["xt"])
    grid = Grid(ds, coords=coords, metrics=metrics)
    func = getattr(grid, funcname)
    for axis_name in grid.axes.keys():
        result = func(ds.tracer, axis_name)
        base_coords = list(result.dims)
        augmented_coords = [c for c in ds.coords if set(ds[c].dims).issubset(result.dims) and c not in result.dims]

        if funcname in ["integrate", "average"]:
            assert set(result.coords) == set(base_coords + augmented_coords)
        else:
            assert set(result.coords) == set(base_coords)

        # TODO: why is the behavior different for integrate and average?
        if funcname not in ["integrate", "average"]:
            result = func(ds.tracer, axis_name, keep_coords=False)
            assert set(result.coords) == set(base_coords)

            result = func(ds.tracer, axis_name, keep_coords=True)
            assert set(result.coords) == set(base_coords + augmented_coords)


def test_keep_coords_deprecation():
    ds, coords, metrics = datasets_grid_metric("B")
    ds = ds.assign_coords(yt_bis=ds["yt"], xt_bis=ds["xt"])
    grid = Grid(ds, coords=coords, metrics=metrics)
    for axis_name in grid.axes.keys():
        with pytest.warns(DeprecationWarning):
            grid.diff(ds.tracer, axis_name, keep_coords=False)


def test_boundary_kwarg_same_as_grid_constructor_kwarg():
    ds = datasets["2d_left"]
    grid1 = Grid(ds, periodic=False)
    grid2 = Grid(ds, periodic=False, boundary={"X": "fill", "Y": "fill"})

    actual1 = grid1.interp(ds.data_g, ("X", "Y"), boundary={"X": "fill", "Y": "fill"})
    actual2 = grid2.interp(ds.data_g, ("X", "Y"))

    xr.testing.assert_identical(actual1, actual2)


@pytest.mark.parametrize(
    "metric_axes,metric_name",
    [
        (["Y", "X"], "area_n"),
        ("X", "dx_t"),
        ("Y", "dy_ne"),
        (["Y", "X"], "dy_n"),
        (["X"], "tracer"),
    ],
)
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize(
    "boundary, boundary_expected",
    [
        ({"X": "fill", "Y": "fill"}, {"X": "fill", "Y": "fill"}),
        ({"X": "extend", "Y": "extend"}, {"X": "extend", "Y": "extend"}),
        pytest.param(
            {"X": "extrapolate", "Y": "extrapolate"},
            {"X": "extrapolate", "Y": "extrapolate"},
            marks=pytest.mark.xfail(reason="padding via extrapolation not yet supported in grid_ufunc refactor"),
        ),
        ("fill", {"X": "fill", "Y": "fill"}),
        ("extend", {"X": "extend", "Y": "extend"}),
        pytest.param(
            "extrapolate",
            {"X": "extrapolate", "Y": "extrapolate"},
            marks=pytest.mark.xfail(reason="padding via extrapolation not yet supported in grid_ufunc refactor"),
        ),
        ({"X": "extend", "Y": "fill"}, {"X": "extend", "Y": "fill"}),
        pytest.param(
            {"X": "extrapolate", "Y": "fill"},
            {"X": "extrapolate", "Y": "fill"},
            marks=pytest.mark.xfail(reason="padding via extrapolation not yet supported in grid_ufunc refactor"),
        ),
        pytest.param(
            "fill",
            {"X": "fill", "Y": "extend"},
            marks=pytest.mark.xfail,
            id="boundary not equal to boundary_expected",
        ),
    ],
)
@pytest.mark.parametrize("fill_value", [None, 0.1])
def test_interp_like(metric_axes, metric_name, periodic, boundary, boundary_expected, fill_value):
    ds, coords, _ = datasets_grid_metric("C")
    grid = Grid(ds, coords=coords, periodic=periodic)
    grid.set_metrics(metric_axes, metric_name)
    metric_available = grid._metrics.get(frozenset(metric_axes), None)
    metric_available = metric_available[0]
    interp_metric = grid.interp_like(metric_available, ds.u, boundary=boundary, fill_value=fill_value)
    expected_metric = grid.interp(ds[metric_name], metric_axes, boundary=boundary_expected, fill_value=fill_value)

    xr.testing.assert_allclose(interp_metric, expected_metric)


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


@pytest.mark.parametrize(
    "funcname",
    [
        "interp",
        "diff",
        "min",
        "max",
        "cumsum",
        "derivative",
        "cumint",
    ],
)
@pytest.mark.parametrize(
    "boundary",
    ["fill", "extend"],
)
@pytest.mark.parametrize(
    "fill_value",
    [0, 10, None],
)
def test_boundary_global_input(funcname, boundary, fill_value):
    """Test that globally defined boundary values result in
    the same output as when the parameters are defined the grid methods
    """
    ds, coords, metrics = datasets_grid_metric("C")
    axis = "X"
    # Test results by globally specifying fill value/boundary on grid object
    grid_global = Grid(
        ds,
        coords=coords,
        metrics=metrics,
        periodic=False,
        boundary=boundary,
        fill_value=fill_value,
    )
    func_global = getattr(grid_global, funcname)
    global_result = func_global(ds.tracer, axis)

    # Test results by manually specifying fill value/boundary on grid method
    grid_manual = Grid(ds, coords=coords, metrics=metrics, periodic=False, boundary=boundary)
    func_manual = getattr(grid_manual, funcname)
    manual_result = func_manual(ds.tracer, axis, boundary=boundary, fill_value=fill_value)
    xr.testing.assert_allclose(global_result, manual_result)


class TestInputErrorGridMethods:
    def test_multiple_keys_vector_input(self):
        ds, _, _ = datasets_grid_metric("C")
        grid = Grid(ds)
        msg = "Vector components provided as dictionaries should contain exactly one key/value pair. .*?"
        with pytest.raises(
            ValueError,
            match=msg,
        ):
            grid.diff({"X": xr.DataArray(), "Y": xr.DataArray()}, "X")

    def test_wrong_input_type_scalar(self):
        ds, _, _ = datasets_grid_metric("C")
        grid = Grid(ds)
        msg = "All data arguments must be either a DataArray or Dictionary .*?"
        with pytest.raises(
            TypeError,
            match=msg,
        ):
            grid.diff("not_a_dataarray", "X")

    def test_wrong_input_type_vector(self):
        ds, _, _ = datasets_grid_metric("C")
        grid = Grid(ds)
        msg = "Dictionary inputs must have a DataArray as value. Got .*?"
        with pytest.raises(
            TypeError,
            match=msg,
        ):
            grid.diff({"X": "not_a_dataarray"}, "X")

    def test_wrong_axis_vector_input_axis(self):
        ds, _, _ = datasets_grid_metric("C")
        grid = Grid(ds)
        msg = "Vector component with unknown axis provided. Grid has axes .*?"
        with pytest.raises(
            ValueError,
            match=msg,
        ):
            grid.diff({"wrong": xr.DataArray()}, "X")


class TestInputErrorApplyAsGridUfunc:
    def test_multiple_keys_vector_input(self):
        ds, _, _ = datasets_grid_metric("C")
        grid = Grid(ds)
        msg = "Vector components provided as dictionaries should contain exactly one key/value pair. .*?"
        with pytest.raises(
            ValueError,
            match=msg,
        ):
            grid.apply_as_grid_ufunc(lambda x: x, {"X": xr.DataArray(), "Y": xr.DataArray()}, "X")

    def test_wrong_input_type_scalar(self):
        ds, _, _ = datasets_grid_metric("C")
        grid = Grid(ds)
        msg = "All data arguments must be either a DataArray or Dictionary .*?"
        with pytest.raises(
            TypeError,
            match=msg,
        ):
            grid.apply_as_grid_ufunc(lambda x: x, "not_a_dataarray", "X")

    def test_wrong_input_type_vector(self):
        ds, _, _ = datasets_grid_metric("C")
        grid = Grid(ds)
        msg = "Dictionary inputs must have a DataArray as value. Got .*?"
        with pytest.raises(
            TypeError,
            match=msg,
        ):
            grid.apply_as_grid_ufunc(lambda x: x, {"X": "not_a_dataarray"}, "X")

    def test_wrong_axis_vector_input_axis(self):
        ds, _, _ = datasets_grid_metric("C")
        grid = Grid(ds)
        msg = "Vector component with unknown axis provided. Grid has axes .*?"
        with pytest.raises(
            ValueError,
            match=msg,
        ):
            grid.apply_as_grid_ufunc(lambda x: x, {"wrong": xr.DataArray()}, "X")

    def test_vector_input_data_other_mismatch(self):
        ds, _, _ = datasets_grid_metric("C")
        grid = Grid(ds)
        msg = "When providing multiple input arguments, `other_component`" " needs to provide one dictionary per input"
        with pytest.raises(
            ValueError,
            match=msg,
        ):
            # Passing 3 args and 2 other components should fail.
            grid.apply_as_grid_ufunc(
                lambda x: x,
                {"X": xr.DataArray()},
                {"Y": xr.DataArray()},
                {"Z": xr.DataArray()},
                axis="X",
                other_component=[{"X": xr.DataArray()}, {"Y": xr.DataArray()}],
            )

    def test_wrong_input_type_vector_multi_input(self):
        ds, _, _ = datasets_grid_metric("C")
        grid = Grid(ds)
        msg = "Dictionary inputs must have a DataArray as value. Got .*?"
        with pytest.raises(
            TypeError,
            match=msg,
        ):
            # Passing 3 args and 2 other components should fail.
            grid.apply_as_grid_ufunc(
                lambda x: x,
                {"X": xr.DataArray()},
                {"Y": "not_a_data_array"},
                axis="X",
                other_component=[{"X": xr.DataArray()}, {"Y": xr.DataArray()}],
            )

    def test_wrong_axis_vector_input_axis_multi_input(self):
        ds, _, _ = datasets_grid_metric("C")
        grid = Grid(ds)
        msg = "Vector component with unknown axis provided. Grid has axes .*?"
        with pytest.raises(
            ValueError,
            match=msg,
        ):
            # Passing 3 args and 2 other components should fail.
            grid.apply_as_grid_ufunc(
                lambda x: x,
                {"X": xr.DataArray()},
                {"Y": xr.DataArray()},
                axis="X",
                other_component=[{"wrong": xr.DataArray()}, {"Y": xr.DataArray()}],
            )

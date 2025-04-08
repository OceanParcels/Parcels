"""Generate"""

import cftime
import numpy as np
import xarray as xr

from tests.utils import TEST_DATA


def get_mom_metadata_rich() -> xr.Dataset:
    """MOM5 datasets based off of the access-om2-01 dataset that used to be in the codebase in parcels v3.

    Metadata has been kept close to the original.
    """
    xlim = (-247.0, -246.0)
    ylim = (-65.0, -63.0)
    gridsize = 0.1

    yu_ocean = np.arange(ylim[0], ylim[1], gridsize)
    xu_ocean = np.arange(xlim[0], xlim[1], gridsize)
    ydim, xdim = len(yu_ocean), len(xu_ocean)

    tdim, zdim = (3, 4)
    st_ocean = np.linspace(0.0, 4.0, zdim)

    times_closed = np.array(
        [cftime.DatetimeNoLeap(1950, 1, i, 12, 0, 0, 0, has_year_zero=True) for i in range(1, tdim + 2)]
    )
    times = times_closed[:-1]

    time_bounds = np.array([times_closed[:-1], times_closed[1:]]).T
    time_bounds -= cftime.DatetimeNoLeap(1900, 1, 1, 0, 0, 0, 0, has_year_zero=True)
    time_bounds = time_bounds.astype(np.float64)

    U = np.random.randn(tdim, zdim, ydim, xdim).astype(np.float32)
    V = np.random.randn(tdim, zdim, ydim, xdim).astype(np.float32)

    time_bounds_attrs = {"long_name": "time axis boundaries", "units": "days"}
    u_attrs = {
        "long_name": "i-current",
        "units": "m/sec",
        "valid_range": np.array([-10.0, 10.0], dtype=np.float32),
        "cell_methods": "time: mean",
        "time_avg_info": "average_T1,average_T2,average_DT",
        "standard_name": "sea_water_x_velocity",
        "number_of_significant_digits": np.int32(3),
    }
    v_attrs = {
        "long_name": "j-current",
        "units": "m/sec",
        "valid_range": np.array([-10.0, 10.0], dtype=np.float32),
        "cell_methods": "time: mean",
        "time_avg_info": "average_T1,average_T2,average_DT",
        "standard_name": "sea_water_y_velocity",
        "number_of_significant_digits": np.int32(3),
    }
    wt_attrs = {
        "long_name": "dia-surface velocity T-points",
        "units": "m/sec",
        "valid_range": np.array([-100000.0, 100000.0], dtype=np.float32),
        "cell_methods": "time: mean",
        "time_avg_info": "average_T1,average_T2,average_DT",
        "number_of_significant_digits": np.int32(2),
    }
    attrs = {
        "grid_type": "mosaic",
        "grid_tile": "1",
    }

    return xr.Dataset(
        {
            "time_bounds": (("time", "nv"), time_bounds, time_bounds_attrs),
            "u": (("time", "st_ocean", "yu_ocean", "xu_ocean"), U, u_attrs),
            "v": (("time", "st_ocean", "yu_ocean", "xu_ocean"), V, v_attrs),
            "wt": (("time", "sw_ocean", "yt_ocean", "xt_ocean"), V, wt_attrs),
        },
        attrs=attrs,
        coords={
            "st_ocean": (
                ("st_ocean",),
                st_ocean,
                {
                    "long_name": "tcell zstar depth",
                    "units": "meters",
                    "cartesian_axis": "Z",
                    "positive": "down",
                    "edges": "st_edges_ocean",
                },
            ),
            "time": (
                ("time",),
                times,
                {
                    "long_name": "time",
                    "cartesian_axis": "T",
                    "calendar_type": "NOLEAP",
                    "bounds": "time_bounds",
                    "units": "days since 1900-01-01",
                    "calendar": "NOLEAP",
                },
            ),
            "xu_ocean": (
                "xu_ocean",
                xu_ocean,
                {"long_name": "ucell longitude", "units": "degrees_E", "cartesian_axis": "X"},
            ),
            "yu_ocean": (
                ("yu_ocean",),
                yu_ocean,
                {"long_name": "ucell latitude", "units": "degrees_N", "cartesian_axis": "Y"},
            ),
            "sw_ocean": (
                ("sw_ocean",),
                #! In the equally dataset these weren't equally spaced based on the gridsize. Something to be concerned about?
                st_ocean + 5 * gridsize,
                {
                    "long_name": "ucell zstar depth",
                    "units": "meters",
                    "cartesian_axis": "Z",
                    "positive": "down",
                    "edges": "sw_edges_ocean",
                },
            ),
            "xt_ocean": (
                ("xt_ocean",),
                xu_ocean + gridsize / 2,
                {"long_name": "tcell longitude", "units": "degrees_E", "cartesian_axis": "X"},
            ),
            "yt_ocean": (
                ("yt_ocean",),
                yu_ocean + gridsize / 2,
                {"long_name": "tcell latitude", "units": "degrees_N", "cartesian_axis": "Y"},
            ),
        },
    )


def assert_common_attrs_equal(do1: xr.Dataset | xr.DataArray, do2: xr.Dataset | xr.DataArray):
    common_attrs = set(do1.attrs) & set(do2.attrs)

    for attr in common_attrs:
        if isinstance(do1.attrs[attr], np.ndarray):
            np.testing.assert_array_equal(do1.attrs[attr], do2.attrs[attr])
        else:
            assert do1.attrs[attr] == do2.attrs[attr], f"error on key {attr!r}"


def test_dataset_complete_metadata():
    ds = get_mom_metadata_rich()

    ds_expected = xr.open_mfdataset(
        [TEST_DATA / "access-om2-01_u.nc", TEST_DATA / "access-om2-01_v.nc", TEST_DATA / "access-om2-01_wt.nc"],
        decode_times=False,
    )

    # check dataset variables both in ds and ds_expected
    assert set(ds.variables) == set(
        ds_expected.variables
    ), f"Variables differ: {set(ds.variables) ^ set(ds_expected.variables)}"

    assert_common_attrs_equal(ds, ds_expected)
    for var in ds.variables:
        assert_common_attrs_equal(ds[var], ds_expected[var])

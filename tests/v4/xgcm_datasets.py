"""Datasets vendored from xgcm test suite for the testing of grids."""

import numpy as np
import pytest
import xarray as xr

# example from comodo website
# https://web.archive.org/web/20160417032300/http://pycomodo.forge.imag.fr/norm.html
# netcdf example {
#         dimensions:
#                 ni = 9 ;
#                 ni_u = 10 ;
#         variables:
#                 float ni(ni) ;
#                         ni:axis = "X" ;
#                         ni:standard_name = "x_grid_index" ;
#                         ni:long_name = "x-dimension of the grid" ;
#                         ni:c_grid_dynamic_range = "2:8" ;
#                 float ni_u(ni_u) ;
#                         ni_u:axis = "X" ;
#                         ni_u:standard_name = "x_grid_index_at_u_location" ;
#                         ni_u:long_name = "x-dimension of the grid" ;
#                         ni_u:c_grid_dynamic_range = "3:8" ;
#                         ni_u:c_grid_axis_shift = -0.5 ;
#         data:
#                 ni = 1, 2, 3, 4, 5, 6, 7, 8, 9 ;
#                 ni_u = 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5 ;
# }

N = 100
datasets = {
    # the comodo example, with renamed dimensions
    "1d_outer": xr.Dataset(
        {"data_c": (["XC"], np.random.rand(9)), "data_g": (["XG"], np.random.rand(10))},
        coords={
            "XC": (
                ["XC"],
                np.arange(1, 10),
                {
                    "axis": "X",
                    "standard_name": "x_grid_index",
                    "long_name": "x-dimension of the grid",
                    "c_grid_dynamic_range": "2:8",
                },
            ),
            "XG": (
                ["XG"],
                np.arange(0.5, 10),
                {
                    "axis": "X",
                    "standard_name": "x_grid_index_at_u_location",
                    "long_name": "x-dimension of the grid",
                    "c_grid_dynamic_range": "3:8",
                    "c_grid_axis_shift": -0.5,
                },
            ),
        },
    ),
    "1d_inner": xr.Dataset(
        {"data_c": (["XC"], np.random.rand(9)), "data_g": (["XG"], np.random.rand(8))},
        coords={
            "XC": (
                ["XC"],
                np.arange(1, 10),
                {
                    "axis": "X",
                    "standard_name": "x_grid_index",
                    "long_name": "x-dimension of the grid",
                    "c_grid_dynamic_range": "2:8",
                },
            ),
            "XG": (
                ["XG"],
                np.arange(1.5, 9),
                {
                    "axis": "X",
                    "standard_name": "x_grid_index_at_u_location",
                    "long_name": "x-dimension of the grid",
                    "c_grid_dynamic_range": "3:8",
                    "c_grid_axis_shift": -0.5,
                },
            ),
        },
    ),
    # my own invention
    "1d_left": xr.Dataset(
        {"data_g": (["XG"], np.random.rand(N)), "data_c": (["XC"], np.random.rand(N))},
        coords={
            "XG": (
                ["XG"],
                2 * np.pi / N * np.arange(0, N),
                {"axis": "X", "c_grid_axis_shift": -0.5},
            ),
            "XC": (["XC"], 2 * np.pi / N * (np.arange(0, N) + 0.5), {"axis": "X"}),
        },
    ),
    "1d_right": xr.Dataset(
        {"data_g": (["XG"], np.random.rand(N)), "data_c": (["XC"], np.random.rand(N))},
        coords={
            "XG": (
                ["XG"],
                2 * np.pi / N * np.arange(1, N + 1),
                {"axis": "X", "c_grid_axis_shift": 0.5},
            ),
            "XC": (["XC"], 2 * np.pi / N * (np.arange(0, N) - 0.5), {"axis": "X"}),
        },
    ),
    "2d_left": xr.Dataset(
        {
            "data_g": (["YG", "XG"], np.random.rand(2 * N, N)),
            "data_c": (["YC", "XC"], np.random.rand(2 * N, N)),
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
        },
    ),
}

# include periodicity
datasets_with_periodicity = {
    "nonperiodic_1d_outer": (datasets["1d_outer"], False),
    "nonperiodic_1d_inner": (datasets["1d_inner"], False),
    "periodic_1d_left": (datasets["1d_left"], True),
    "nonperiodic_1d_left": (datasets["1d_left"], False),
    "periodic_1d_right": (datasets["1d_right"], True),
    "nonperiodic_1d_right": (datasets["1d_right"], False),
    "periodic_2d_left": (datasets["2d_left"], True),
    "nonperiodic_2d_left": (datasets["2d_left"], False),
    "xperiodic_2d_left": (datasets["2d_left"], ["X"]),
    "yperiodic_2d_left": (datasets["2d_left"], ["Y"]),
}

expected_values = {
    "nonperiodic_1d_outer": {"axes": {"X": {"center": "XC", "outer": "XG"}}},
    "nonperiodic_1d_inner": {"axes": {"X": {"center": "XC", "inner": "XG"}}},
    "periodic_1d_left": {"axes": {"X": {"center": "XC", "left": "XG"}}},
    "nonperiodic_1d_left": {"axes": {"X": {"center": "XC", "left": "XG"}}},
    "periodic_1d_right": {
        "axes": {"X": {"center": "XC", "right": "XG"}},
        "shift": True,
    },
    "nonperiodic_1d_right": {
        "axes": {"X": {"center": "XC", "right": "XG"}},
        "shift": True,
    },
    "periodic_2d_left": {
        "axes": {
            "X": {"center": "XC", "left": "XG"},
            "Y": {"center": "YC", "left": "YG"},
        }
    },
    "nonperiodic_2d_left": {
        "axes": {
            "X": {"center": "XC", "left": "XG"},
            "Y": {"center": "YC", "left": "YG"},
        }
    },
    "xperiodic_2d_left": {
        "axes": {
            "X": {"center": "XC", "left": "XG"},
            "Y": {"center": "YC", "left": "YG"},
        }
    },
    "yperiodic_2d_left": {
        "axes": {
            "X": {"center": "XC", "left": "XG"},
            "Y": {"center": "YC", "left": "YG"},
        }
    },
}


@pytest.fixture(scope="module", params=datasets_with_periodicity.keys())
def all_datasets(request):
    ds, periodic = datasets_with_periodicity[request.param]
    return ds, periodic, expected_values[request.param]


@pytest.fixture(
    scope="module",
    params=[
        "nonperiodic_1d_outer",
        "nonperiodic_1d_inner",
        "nonperiodic_1d_left",
        "nonperiodic_1d_right",
    ],
)
def nonperiodic_1d(request):
    ds, periodic = datasets_with_periodicity[request.param]
    return ds, periodic, expected_values[request.param]


@pytest.fixture(scope="module", params=["periodic_1d_left", "periodic_1d_right"])
def periodic_1d(request):
    ds, periodic = datasets_with_periodicity[request.param]
    return ds, periodic, expected_values[request.param]


@pytest.fixture(
    scope="module",
    params=[
        "periodic_2d_left",
        "nonperiodic_2d_left",
        "xperiodic_2d_left",
        "yperiodic_2d_left",
    ],
)
def all_2d(request):
    ds, periodic = datasets_with_periodicity[request.param]
    return ds, periodic, expected_values[request.param]


@pytest.fixture(scope="module", params=["periodic_2d_left"])
def periodic_2d(request):
    ds, periodic = datasets_with_periodicity[request.param]
    return ds, periodic, expected_values[request.param]


@pytest.fixture(
    scope="module",
    params=["nonperiodic_2d_left", "xperiodic_2d_left", "yperiodic_2d_left"],
)
def nonperiodic_2d(request):
    ds, periodic = datasets_with_periodicity[request.param]
    return ds, periodic, expected_values[request.param]


def datasets_grid_metric(grid_type):
    """Uniform grid test dataset.
    Should eventually be extended to nonuniform grid
    """
    xt = np.arange(4)
    xu = xt + 0.5
    yt = np.arange(5)
    yu = yt + 0.5
    zt = np.arange(6)
    zw = zt + 0.5
    t = np.arange(10)

    def data_generator():
        return np.random.rand(len(xt), len(yt), len(t), len(zt))

    # Need to add a tracer here to get the tracer dimsuffix
    tr = xr.DataArray(data_generator(), coords=[("xt", xt), ("yt", yt), ("time", t), ("zt", zt)])

    u_b = xr.DataArray(data_generator(), coords=[("xu", xu), ("yu", yu), ("time", t), ("zt", zt)])

    v_b = xr.DataArray(data_generator(), coords=[("xu", xu), ("yu", yu), ("time", t), ("zt", zt)])

    u_c = xr.DataArray(data_generator(), coords=[("xu", xu), ("yt", yt), ("time", t), ("zt", zt)])

    v_c = xr.DataArray(data_generator(), coords=[("xt", xt), ("yu", yu), ("time", t), ("zt", zt)])

    wt = xr.DataArray(data_generator(), coords=[("xt", xt), ("yt", yt), ("time", t), ("zw", zw)])

    # maybe also add some other combo of x,t y,t arrays....
    timeseries = xr.DataArray(np.random.rand(len(t)), coords=[("time", t)])

    # northeast distance
    dx = 0.3
    dy = 2
    dz = 20

    dx_ne = xr.DataArray(np.ones([len(xt), len(yt)]) * dx - 0.1, coords=[("xu", xu), ("yu", yu)])
    dx_n = xr.DataArray(np.ones([len(xt), len(yt)]) * dx - 0.2, coords=[("xt", xt), ("yu", yu)])
    dx_e = xr.DataArray(np.ones([len(xt), len(yt)]) * dx - 0.3, coords=[("xu", xu), ("yt", yt)])
    dx_t = xr.DataArray(np.ones([len(xt), len(yt)]) * dx - 0.4, coords=[("xt", xt), ("yt", yt)])

    dy_ne = xr.DataArray(np.ones([len(xt), len(yt)]) * dy + 0.1, coords=[("xu", xu), ("yu", yu)])
    dy_n = xr.DataArray(np.ones([len(xt), len(yt)]) * dy + 0.2, coords=[("xt", xt), ("yu", yu)])
    dy_e = xr.DataArray(np.ones([len(xt), len(yt)]) * dy + 0.3, coords=[("xu", xu), ("yt", yt)])
    dy_t = xr.DataArray(np.ones([len(xt), len(yt)]) * dy + 0.4, coords=[("xt", xt), ("yt", yt)])

    # dz elements at horizontal tracer points
    dz_t = xr.DataArray(data_generator() * dz, coords=[("xt", xt), ("yt", yt), ("time", t), ("zt", zt)])
    dz_w = xr.DataArray(data_generator() * dz, coords=[("xt", xt), ("yt", yt), ("time", t), ("zw", zw)])
    # dz elements at velocity points
    dz_w_ne = xr.DataArray(data_generator() * dz, coords=[("xu", xu), ("yu", yu), ("time", t), ("zw", zw)])
    dz_w_n = xr.DataArray(data_generator() * dz, coords=[("xt", xt), ("yu", yu), ("time", t), ("zw", zw)])
    dz_w_e = xr.DataArray(data_generator() * dz, coords=[("xu", xu), ("yt", yt), ("time", t), ("zw", zw)])

    # Make sure the areas are not just the product of x and y distances
    area_ne = (dx_ne * dy_ne) + 0.1
    area_n = (dx_n * dy_n) + 0.2
    area_e = (dx_e * dy_e) + 0.3
    area_t = (dx_t * dy_t) + 0.4

    # calculate volumes, but again add small differences.
    volume_t = (dx_t * dy_t * dz_t) + 0.25

    def _add_metrics(obj):
        obj = obj.copy()
        for name, data in [
            ("dx_ne", dx_ne),
            ("dx_n", dx_n),
            ("dx_e", dx_e),
            ("dx_t", dx_t),
            ("dy_ne", dy_ne),
            ("dy_n", dy_n),
            ("dy_e", dy_e),
            ("dy_t", dy_t),
            ("dz_t", dz_t),
            ("dz_w", dz_w),
            ("dz_w_ne", dz_w_ne),
            ("dz_w_n", dz_w_n),
            ("dz_w_e", dz_w_e),
            ("area_ne", area_ne),
            ("area_n", area_n),
            ("area_e", area_e),
            ("area_t", area_t),
            ("volume_t", volume_t),
        ]:
            obj.coords[name] = data
            obj.coords[name].attrs["tracked_name"] = name
        # add xgcm attrs
        for ii in ["xu", "xt"]:
            obj[ii].attrs["axis"] = "X"
        for ii in ["yu", "yt"]:
            obj[ii].attrs["axis"] = "Y"
        for ii in ["zt", "zw"]:
            obj[ii].attrs["axis"] = "Z"
        for ii in ["time"]:
            obj[ii].attrs["axis"] = "T"
        for ii in ["xu", "yu", "zw"]:
            obj[ii].attrs["c_grid_axis_shift"] = 0.5
        return obj

    coords = {
        "X": {"center": "xt", "right": "xu"},
        "Y": {"center": "yt", "right": "yu"},
        "Z": {"center": "zt", "right": "zw"},
    }

    metrics = {
        ("X",): ["dx_t", "dx_n", "dx_e", "dx_ne"],
        ("Y",): ["dy_t", "dy_n", "dy_e", "dy_ne"],
        ("Z",): ["dz_t", "dz_w", "dz_w_ne", "dz_w_n", "dz_w_e"],
        ("X", "Y"): ["area_t", "area_n", "area_e", "area_ne"],
        ("X", "Y", "Z"): ["volume_t"],
    }

    # combine to different grid configurations (B and C grid)
    if grid_type == "B":
        ds = _add_metrics(xr.Dataset({"u": u_b, "v": v_b, "wt": wt, "tracer": tr, "timeseries": timeseries}))
    elif grid_type == "C":
        ds = _add_metrics(xr.Dataset({"u": u_c, "v": v_c, "wt": wt, "tracer": tr, "timeseries": timeseries}))
    else:
        raise ValueError(f"Invalid input [{grid_type}] for `grid_type`. Only supports `B` and `C` at the moment ")

    return ds, coords, metrics

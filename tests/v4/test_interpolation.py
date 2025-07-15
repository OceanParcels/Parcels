import itertools

import numpy as np
import xarray as xr

from parcels.application_kernels.interpolation import XTriCurviLinear
from parcels.field import Field
from parcels.xgcm import Grid
from parcels.xgrid import XGrid


def get_unit_square_ds():
    T, Z, Y, X = 2, 2, 2, 2
    TIME = xr.date_range("2000", "2001", T)

    _, data_z, data_y, data_x = np.meshgrid(
        np.zeros(T),
        np.linspace(0, 1, Z),
        np.linspace(0, 1, Y),
        np.linspace(0, 1, X),
        indexing="ij",
    )

    return xr.Dataset(
        {
            "0 to 1 in X": (["time", "ZG", "YG", "XG"], data_x),
            "0 to 1 in Y": (["time", "ZG", "YG", "XG"], data_y),
            "0 to 1 in Z": (["time", "ZG", "YG", "XG"], data_z),
            "0 to 1 in X (T-points)": (["time", "ZC", "YC", "XC"], data_x + 0.5),
            "0 to 1 in Y (T-points)": (["time", "ZC", "YC", "XC"], data_y + 0.5),
            "0 to 1 in Z (T-points)": (["time", "ZC", "YC", "XC"], data_z + 0.5),
            "0 to 1 in X (U velocity C-grid points)": (["time", "ZC", "YC", "XG"], data_x),
            "0 to 1 in Y (V velocity C-grid points)": (["time", "ZC", "YG", "XC"], data_y),
        },
        coords={
            "XG": (
                ["XG"],
                np.arange(0, X),
                {"axis": "X", "c_grid_axis_shift": -0.5},
            ),
            "XC": (["XC"], np.arange(0, X) + 0.5, {"axis": "X"}),
            "YG": (
                ["YG"],
                np.arange(0, Y),
                {"axis": "Y", "c_grid_axis_shift": -0.5},
            ),
            "YC": (
                ["YC"],
                np.arange(0, Y) + 0.5,
                {"axis": "Y"},
            ),
            "ZG": (
                ["ZG"],
                np.arange(Z),
                {"axis": "Z", "c_grid_axis_shift": -0.5},
            ),
            "ZC": (
                ["ZC"],
                np.arange(Z) + 0.5,
                {"axis": "Z"},
            ),
            "lon": (["XG"], np.arange(0, X)),
            "lat": (["YG"], np.arange(0, Y)),
            "depth": (["ZG"], np.arange(Z)),
            "time": (["time"], TIME, {"axis": "T"}),
        },
    )


def test_XTriRectiLinear_interpolation():
    ds = get_unit_square_ds()
    grid = XGrid(Grid(ds))
    field = Field("test", ds["0 to 1 in X"], grid=grid, interp_method=XTriCurviLinear)
    left = field.time_interval.left

    epsilon = 1e-6
    N = 4

    # Interpolate wrt. items on f-points
    for x, y, z in itertools.product(np.linspace(0 + epsilon, 1 - epsilon, N), repeat=3):
        assert np.isclose(x, field.eval(left, z, y, x)), f"Failed for x={x}, y={y}, z={z}"

    field = Field("test", ds["0 to 1 in Y"], grid=grid, interp_method=XTriCurviLinear)
    for x, y, z in itertools.product(np.linspace(0 + epsilon, 1 - epsilon, N), repeat=3):
        assert np.isclose(y, field.eval(left, z, y, x)), f"Failed for x={x}, y={y}, z={z}"

    field = Field("test", ds["0 to 1 in Z"], grid=grid, interp_method=XTriCurviLinear)
    for x, y, z in itertools.product(np.linspace(0 + epsilon, 1 - epsilon, N), repeat=3):
        assert np.isclose(z, field.eval(left, z, y, x)), f"Failed for x={x}, y={y}, z={z}"

    # Interpolate wrt. items on T-points
    field = Field("test", ds["0 to 1 in X (T-points)"], grid=grid, interp_method=XTriCurviLinear)
    for x, y, z in itertools.product(np.linspace(0.5 + epsilon, 1 - epsilon, N), repeat=3):
        assert np.isclose(x, field.eval(left, z, y, x)), f"Failed for x={x}, y={y}, z={z}"

    field = Field("test", ds["0 to 1 in Y (T-points)"], grid=grid, interp_method=XTriCurviLinear)
    for x, y, z in itertools.product(np.linspace(0.5 + epsilon, 1 - epsilon, N), repeat=3):
        assert np.isclose(y, field.eval(left, z, y, x)), f"Failed for x={x}, y={y}, z={z}"

    field = Field("test", ds["0 to 1 in Z (T-points)"], grid=grid, interp_method=XTriCurviLinear)
    for x, y, z in itertools.product(np.linspace(0.5 + epsilon, 1 - epsilon, N), repeat=3):
        assert np.isclose(z, field.eval(left, z, y, x)), f"Failed for x={x}, y={y}, z={z}"

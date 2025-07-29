import numpy as np
import xarray as xr


def simple_UV_dataset(dims=(360, 2, 30, 4), maxdepth=1, mesh_type="spherical"):
    max_lon = 180.0 if mesh_type == "spherical" else 1e6

    return xr.Dataset(
        {"U": (["time", "depth", "YG", "XG"], np.zeros(dims)), "V": (["time", "depth", "YG", "XG"], np.zeros(dims))},
        coords={
            "time": (["time"], xr.date_range("2000", "2001", dims[0]), {"axis": "T"}),
            "depth": (["depth"], np.linspace(0, maxdepth, dims[1]), {"axis": "Z"}),
            "YC": (["YC"], np.arange(dims[2]) + 0.5, {"axis": "Y"}),
            "YG": (["YG"], np.arange(dims[2]), {"axis": "Y", "c_grid_axis_shift": -0.5}),
            "XC": (["XC"], np.arange(dims[3]) + 0.5, {"axis": "X"}),
            "XG": (["XG"], np.arange(dims[3]), {"axis": "X", "c_grid_axis_shift": -0.5}),
            "lat": (["YG"], np.linspace(-90, 90, dims[2]), {"axis": "Y", "c_grid_axis_shift": 0.5}),
            "lon": (["XG"], np.linspace(-max_lon, max_lon, dims[3]), {"axis": "X", "c_grid_axis_shift": -0.5}),
        },
    )

import numpy as np
import pandas as pd
import xarray as xr


def _dummy_same_points_dataset():
    tdim, zdim = 15, 10
    xy_grid_width = 1
    time = pd.date_range(start="1993-01-01 12:00:00", periods=tdim, freq="D")
    lon = np.arange(40.0, 100.0, xy_grid_width)
    lat = np.arange(-30.0, 30.0, xy_grid_width)
    ydim, xdim = len(lat), len(lon)
    depth = np.linspace(0, 900, zdim)

    U = np.random.randn(tdim, zdim, ydim, xdim)
    V = np.random.randn(tdim, zdim, ydim, xdim)

    return xr.Dataset(
        {
            "U": (["time", "depth", "lat", "lon"], U),
            "V": (["time", "depth", "lat", "lon"], V),
        },
        coords={"time": time, "depth": depth, "lat": lat, "lon": lon},
    )


def dummy_agrid_dataset():
    """Dummy b-grid dataset using COMODO grid conventions."""
    return _dummy_same_points_dataset().rename({"depth": "ZG", "lat": "YG", "lon": "XG"})


def dummy_bgrid_dataset():
    """Dummy b-grid dataset using COMODO grid conventions."""
    return _dummy_same_points_dataset().rename({"depth": "ZC", "lat": "YC", "lon": "XC"})

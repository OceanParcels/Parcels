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


def radial_rotation_dataset(xdim=200, ydim=200):  # Define 2D flat, square fieldset for testing purposes.
    lon = np.linspace(0, 60, xdim, dtype=np.float32)
    lat = np.linspace(0, 60, ydim, dtype=np.float32)

    x0 = 30.0  # Define the origin to be the centre of the Field.
    y0 = 30.0

    U = np.zeros((1, 1, ydim, xdim), dtype=np.float32)
    V = np.zeros((1, 1, ydim, xdim), dtype=np.float32)

    omega = 2 * np.pi / 86400.0  # Define the rotational period as 1 day.

    for i in range(lon.size):
        for j in range(lat.size):
            r = np.sqrt((lon[i] - x0) ** 2 + (lat[j] - y0) ** 2)
            assert r >= 0.0
            assert r <= np.sqrt(x0**2 + y0**2)

            theta = np.arctan2((lat[j] - y0), (lon[i] - x0))
            assert abs(theta) <= np.pi

            U[:, :, j, i] = r * np.sin(theta) * omega
            V[:, :, j, i] = -r * np.cos(theta) * omega

    return xr.Dataset(
        {"U": (["time", "depth", "YG", "XG"], U), "V": (["time", "depth", "YG", "XG"], V)},
        coords={
            "time": (["time"], np.array([np.timedelta64(0, "s")]), {"axis": "T"}),
            "depth": (["depth"], np.array([0]), {"axis": "Z"}),
            "YC": (["YC"], np.arange(ydim) + 0.5, {"axis": "Y"}),
            "YG": (["YG"], np.arange(ydim), {"axis": "Y", "c_grid_axis_shift": -0.5}),
            "XC": (["XC"], np.arange(xdim) + 0.5, {"axis": "X"}),
            "XG": (["XG"], np.arange(xdim), {"axis": "X", "c_grid_axis_shift": -0.5}),
            "lat": (["YG"], lat, {"axis": "Y", "c_grid_axis_shift": 0.5}),
            "lon": (["XG"], lon, {"axis": "X", "c_grid_axis_shift": -0.5}),
        },
    )

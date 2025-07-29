import numpy as np
import xarray as xr

from . import T, X, Y, Z

__all__ = ["T", "X", "Y", "Z", "datasets"]

TIME = xr.date_range("2000", "2001", T)


def _rotated_curvilinear_grid():
    XG = np.arange(X)
    YG = np.arange(Y)
    LON, LAT = np.meshgrid(XG, YG)

    angle = -np.pi / 24
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    # rotate the LON and LAT grids
    LON, LAT = np.einsum("ji, mni -> jmn", rotation, np.dstack([LON, LAT]))

    return xr.Dataset(
        {
            "data_g": (["time", "ZG", "YG", "XG"], np.random.rand(T, Z, Y, X)),
            "data_c": (["time", "ZC", "YC", "XC"], np.random.rand(T, Z, Y, X)),
            "U (A grid)": (["time", "ZG", "YG", "XG"], np.random.rand(T, Z, Y, X)),
            "V (A grid)": (["time", "ZG", "YG", "XG"], np.random.rand(T, Z, Y, X)),
            "U (C grid)": (["time", "ZG", "YC", "XG"], np.random.rand(T, Z, Y, X)),
            "V (C grid)": (["time", "ZG", "YG", "XC"], np.random.rand(T, Z, Y, X)),
        },
        coords={
            "XG": (["XG"], XG, {"axis": "X", "c_grid_axis_shift": -0.5}),
            "YG": (["YG"], YG, {"axis": "Y", "c_grid_axis_shift": -0.5}),
            "XC": (["XC"], XG + 0.5, {"axis": "X"}),
            "YC": (["YC"], YG + 0.5, {"axis": "Y"}),
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
            "depth": (["ZG"], np.arange(Z), {"axis": "Z"}),
            "time": (["time"], TIME, {"axis": "T"}),
            "lon": (
                ["YG", "XG"],
                LON,
                {"axis": "X", "c_grid_axis_shift": -0.5},  # ? Needed?
            ),
            "lat": (
                ["YG", "XG"],
                LAT,
                {"axis": "Y", "c_grid_axis_shift": -0.5},  # ? Needed?
            ),
        },
    )


def _cartesion_to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def _polar_to_cartesian(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def _unrolled_cone_curvilinear_grid():
    # Not a great unrolled cone, but this is good enough for testing
    # you can use matplotlib pcolormesh to plot
    XG = np.arange(X)
    YG = np.arange(Y) * 0.25

    pivot = -10, 0
    LON, LAT = np.meshgrid(XG, YG)

    new_lon_lat = []

    min_lon = np.min(XG)
    for lon, lat in zip(LON.flatten(), LAT.flatten(), strict=True):
        r, _ = _cartesion_to_polar(lon - pivot[0], lat - pivot[1])
        _, theta = _cartesion_to_polar(min_lon - pivot[0], lat - pivot[1])
        theta *= 1.2
        r *= 1.2
        lon, lat = _polar_to_cartesian(r, theta)
        new_lon_lat.append((lon + pivot[0], lat + pivot[1]))

    new_lon, new_lat = zip(*new_lon_lat, strict=True)
    LON, LAT = np.array(new_lon).reshape(LON.shape), np.array(new_lat).reshape(LAT.shape)

    return xr.Dataset(
        {
            "data_g": (["time", "ZG", "YG", "XG"], np.random.rand(T, Z, Y, X)),
            "data_c": (["time", "ZC", "YC", "XC"], np.random.rand(T, Z, Y, X)),
            "U (A grid)": (["time", "ZG", "YG", "XG"], np.random.rand(T, Z, Y, X)),
            "V (A grid)": (["time", "ZG", "YG", "XG"], np.random.rand(T, Z, Y, X)),
            "U (C grid)": (["time", "ZG", "YC", "XG"], np.random.rand(T, Z, Y, X)),
            "V (C grid)": (["time", "ZG", "YG", "XC"], np.random.rand(T, Z, Y, X)),
        },
        coords={
            "XG": (["XG"], XG, {"axis": "X", "c_grid_axis_shift": -0.5}),
            "YG": (["YG"], YG, {"axis": "Y", "c_grid_axis_shift": -0.5}),
            "XC": (["XC"], XG + 0.5, {"axis": "X"}),
            "YC": (["YC"], YG + 0.5, {"axis": "Y"}),
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
            "depth": (["ZG"], np.arange(Z), {"axis": "Z"}),
            "time": (["time"], TIME, {"axis": "T"}),
            "lon": (
                ["YG", "XG"],
                LON,
                {"axis": "X", "c_grid_axis_shift": -0.5},  # ? Needed?
            ),
            "lat": (
                ["YG", "XG"],
                LAT,
                {"axis": "Y", "c_grid_axis_shift": -0.5},  # ? Needed?
            ),
        },
    )


datasets = {
    "2d_left_rotated": _rotated_curvilinear_grid(),
    "ds_2d_left": xr.Dataset(  # MITgcm indexing style
        {
            "data_g": (["time", "ZG", "YG", "XG"], np.random.rand(T, Z, Y, X)),
            "data_c": (["time", "ZC", "YC", "XC"], np.random.rand(T, Z, Y, X)),
            "U (A grid)": (["time", "ZG", "YG", "XG"], np.random.rand(T, Z, Y, X)),
            "V (A grid)": (["time", "ZG", "YG", "XG"], np.random.rand(T, Z, Y, X)),
            "U (C grid)": (["time", "ZG", "YC", "XG"], np.random.rand(T, Z, Y, X)),
            "V (C grid)": (["time", "ZG", "YG", "XC"], np.random.rand(T, Z, Y, X)),
        },
        coords={
            "XG": (
                ["XG"],
                2 * np.pi / X * np.arange(0, X),
                {"axis": "X", "c_grid_axis_shift": -0.5},
            ),
            "XC": (["XC"], 2 * np.pi / X * (np.arange(0, X) + 0.5), {"axis": "X"}),
            "YG": (
                ["YG"],
                2 * np.pi / (Y) * np.arange(0, Y),
                {"axis": "Y", "c_grid_axis_shift": -0.5},
            ),
            "YC": (
                ["YC"],
                2 * np.pi / (Y) * (np.arange(0, Y) + 0.5),
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
            "lon": (["XG"], 2 * np.pi / X * np.arange(0, X)),
            "lat": (["YG"], 2 * np.pi / (Y) * np.arange(0, Y)),
            "depth": (["ZG"], np.arange(Z)),
            "time": (["time"], TIME, {"axis": "T"}),
        },
    ),
    "ds_2d_right": xr.Dataset(  # NEMO indexing style
        {
            "data_g": (["time", "ZG", "YG", "XG"], np.random.rand(T, Z, Y, X)),
            "data_c": (["time", "ZC", "YC", "XC"], np.random.rand(T, Z, Y, X)),
            "U (A grid)": (["time", "ZG", "YG", "XG"], np.random.rand(T, Z, Y, X)),
            "V (A grid)": (["time", "ZG", "YG", "XG"], np.random.rand(T, Z, Y, X)),
            "U (C grid)": (["time", "ZG", "YC", "XG"], np.random.rand(T, Z, Y, X)),
            "V (C grid)": (["time", "ZG", "YG", "XC"], np.random.rand(T, Z, Y, X)),
        },
        coords={
            "XG": (
                ["XG"],
                2 * np.pi / X * np.arange(0, X),
                {"axis": "X", "c_grid_axis_shift": 0.5},
            ),
            "XC": (["XC"], 2 * np.pi / X * (np.arange(0, X) - 0.5), {"axis": "X"}),
            "YG": (
                ["YG"],
                2 * np.pi / (Y) * np.arange(0, Y),
                {"axis": "Y", "c_grid_axis_shift": 0.5},
            ),
            "YC": (
                ["YC"],
                2 * np.pi / (Y) * (np.arange(0, Y) - 0.5),
                {"axis": "Y"},
            ),
            "ZG": (
                ["ZG"],
                np.arange(Z),
                {"axis": "Z", "c_grid_axis_shift": 0.5},
            ),
            "ZC": (
                ["ZC"],
                np.arange(Z) - 0.5,
                {"axis": "Z"},
            ),
            "lon": (["XG"], 2 * np.pi / X * np.arange(0, X)),
            "lat": (["YG"], 2 * np.pi / (Y) * np.arange(0, Y)),
            "depth": (["ZG"], np.arange(Z)),
            "time": (["time"], TIME, {"axis": "T"}),
        },
    ),
    "2d_left_unrolled_cone": _unrolled_cone_curvilinear_grid(),
}

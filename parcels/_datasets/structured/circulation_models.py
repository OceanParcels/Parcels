"""Datasets mimicking the layout of real-world hydrodynamic models"""

import numpy as np
import xarray as xr

from . import T, X, Y, Z

__all__ = ["T", "X", "Y", "Z", "datasets"]

TIME = xr.date_range("2000", "2001", T)


datasets = {
    "ds_copernicusmarine": xr.Dataset(
        # Copernicus Marine Service dataset as retrieved by the `copernicusmarine` toolkit
        {
            "uo": (
                ["time", "depth", "latitude", "longitude"],
                np.random.rand(T, Z, Y, X),
                {
                    "valid_max": 5.0,
                    "unit_long": "Meters per second",
                    "units": "m s-1",
                    "long_name": "Eastward velocity",
                    "standard_name": "eastward_sea_water_velocity",
                    "valid_min": -5.0,
                },
            ),
            "vo": (
                ["time", "depth", "latitude", "longitude"],
                np.random.rand(T, Z, Y, X),
                {
                    "valid_max": 5.0,
                    "unit_long": "Meters per second",
                    "units": "m s-1",
                    "long_name": "Northward velocity",
                    "standard_name": "northward_sea_water_velocity",
                    "valid_min": -5.0,
                },
            ),
        },
        coords={
            "depth": (
                ["depth"],
                np.linspace(0.49, 5727.92, Z),
                {
                    "unit_long": "Meters",
                    "units": "m",
                    "axis": "Z",
                    "long_name": "depth",
                    "standard_name": "depth",
                    "positive": "down",
                },
            ),
            "latitude": (
                ["latitude"],
                np.linspace(-90, 90, Y),
                {
                    "unit_long": "Degrees North",
                    "units": "degrees_north",
                    "axis": "Y",
                    "long_name": "Latitude",
                    "standard_name": "latitude",
                },
            ),
            "longitude": (
                ["longitude"],
                np.linspace(-180, 180, X),
                {
                    "unit_long": "Degrees East",
                    "units": "degrees_east",
                    "axis": "X",
                    "long_name": "Longitude",
                    "standard_name": "longitude",
                },
            ),
            "time": (
                ["time"],
                TIME,
                {
                    "unit_long": "Hours Since 1950-01-01",
                    "axis": "T",
                    "long_name": "Time",
                    "standard_name": "time",
                },
            ),
        },
    ),
    "ds_NEMO_MOI_U": xr.Dataset(
        # NEMO model dataset (U component) as serviced by Mercator Ocean International
        {
            "vozocrtx": (
                ["deptht", "y", "x"],
                np.random.rand(Z, Y, X, dtype="float64"),
                {
                    "units": "m s-1",
                    "valid_min": -10.0,
                    "valid_max": 10.0,
                    "long_name": "Zonal velocity",
                    "unit_long": "Meters per second",
                    "standard_name": "sea_water_x_velocity",
                    "short_name": "vozocrtx",
                    "online_operation": "N/A",
                    "interval_operation": 86400,
                    "interval_write": 86400,
                    "associate": "time_counter deptht nav_lat nav_lon",
                },
            ),
            "sotkeavmu1": (
                ["y", "x"],
                np.random.rand(Y, X, dtype="float64"),
                {
                    "units": "m2 s-1",
                    "valid_min": 0.0,
                    "valid_max": 100.0,
                    "long_name": "Vertical Eddy Viscosity U 1m",
                    "standard_name": "ocean_vertical_eddy_viscosity_u_1m",
                    "short_name": "sotkeavmu1",
                    "online_operation": "N/A",
                    "interval_operation": 86400,
                    "interval_write": 86400,
                    "associate": "time_counter nav_lat nav_lon",
                },
            ),
        },
        coords={
            "nav_lon": (
                ["y, x"],
                np.tile(np.linspace(-179, 179, X, endpoint=False), (Y, 1)),  # note that this is not curvilinear
                {
                    "units": "degrees_east",
                    "valid_min": -179.99984754002182,
                    "valid_max": 179.999842386314,
                    "long_name": "Longitude",
                    "nav_model": "Default grid",
                    "standard_name": "longitude",
                },
            ),
            "nav_lat": (
                ["y, x"],
                np.tile(np.linspace(-75, 85, Y).reshape(-1, 1), (1, X)),  # note that this is not curvilinear
                {
                    "units": "degrees_north",
                    "valid_min": -77.0104751586914,
                    "valid_max": 89.9591064453125,
                    "long_name": "Latitude",
                    "nav_model": "Default grid",
                    "standard_name": "latitude",
                },
            ),
            "x": (
                ["x"],
                np.arange(X, dtype="int32"),
                {
                    "standard_name": "projection_x_coordinate",
                    "axis": "X",
                    "units": "1",
                },
            ),
            "y": (
                ["y"],
                np.arange(Y, dtype="int32"),
                {
                    "standard_name": "projection_y_coordinate",
                    "axis": "Y",
                    "units": "1",
                },
            ),
            "time_counter": (
                [],
                np.empty(0, dtype="datetime64[ns]"),
                {
                    "standard_name": "time",
                    "long_name": "Time axis",
                    "axis": "T",
                    "time_origin": "1950-JAN-01 00:00:00",
                },
            ),
            "deptht": (
                ["deptht"],
                np.linspace(1, 5500, Z, dtype="float64"),
                {
                    "units": "m",
                    "positive": "down",
                    "valid_min": 0.4940253794193268,
                    "valid_max": 5727.91650390625,
                    "long_name": "Vertical T levels",
                    "standard_name": "depth",
                    "axis": "Z",
                },
            ),
        },
    ),
    "ds_NEMO_MOI_V": xr.Dataset(
        # NEMO model dataset (V component) as serviced by Mercator Ocean International
        {
            "vomecrty": (
                ["deptht", "y", "x"],
                np.random.rand(Z, Y, X, dtype="float64"),
                {
                    "units": "m s-1",
                    "valid_min": -10.0,
                    "valid_max": 10.0,
                    "long_name": "Meridional velocity",
                    "unit_long": "Meters per second",
                    "standard_name": "sea_water_y_velocity",
                    "short_name": "vomecrty",
                    "online_operation": "N/A",
                    "interval_operation": 86400,
                    "interval_write": 86400,
                    "associate": "time_counter deptht nav_lat nav_lon",
                },
            ),
        },
        coords={
            "nav_lon": (
                ["y, x"],
                np.tile(np.linspace(-179, 179, X, endpoint=False), (Y, 1)),  # note that this is not curvilinear
                {
                    "units": "degrees_east",
                    "valid_min": -179.99984754002182,
                    "valid_max": 179.999842386314,
                    "long_name": "Longitude",
                    "nav_model": "Default grid",
                    "standard_name": "longitude",
                },
            ),
            "nav_lat": (
                ["y, x"],
                np.tile(np.linspace(-75, 85, Y).reshape(-1, 1), (1, X)),  # note that this is not curvilinear
                {
                    "units": "degrees_north",
                    "valid_min": -77.0104751586914,
                    "valid_max": 89.9591064453125,
                    "long_name": "Latitude",
                    "nav_model": "Default grid",
                    "standard_name": "latitude",
                },
            ),
            "x": (
                ["x"],
                np.arange(X, dtype="int32"),
                {
                    "standard_name": "projection_x_coordinate",
                    "axis": "X",
                    "units": "1",
                },
            ),
            "y": (
                ["y"],
                np.arange(Y, dtype="int32"),
                {
                    "standard_name": "projection_y_coordinate",
                    "axis": "Y",
                    "units": "1",
                },
            ),
            "time_counter": (
                [],
                np.empty(0, dtype="datetime64[ns]"),
                {
                    "standard_name": "time",
                    "long_name": "Time axis",
                    "axis": "T",
                    "time_origin": "1950-JAN-01 00:00:00",
                },
            ),
            "deptht": (
                ["deptht"],
                np.linspace(1, 5500, Z, dtype="float64"),
                {
                    "units": "m",
                    "positive": "down",
                    "valid_min": 0.4940253794193268,
                    "valid_max": 5727.91650390625,
                    "long_name": "Vertical T levels",
                    "standard_name": "depth",
                    "axis": "Z",
                },
            ),
        },
    ),
}

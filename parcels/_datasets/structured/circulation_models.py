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
}

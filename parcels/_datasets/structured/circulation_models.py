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
            "uo": (["depth", "latitude", "longitude", "time"], np.random.rand(T, Z, Y, X)),
            "vo": (["depth", "latitude", "longitude", "time"], np.random.rand(T, Z, Y, X)),
        },
        coords={
            "depth": (["depth"], np.linspace(0.49, 5727.92, Z), {"axis": "Z"}),
            "latitude": (["latitude"], np.linspace(-90, 90, Y), {"axis": "Y"}),
            "longitude": (["longitude"], np.linspace(-180, 180, X), {"axis": "X"}),
            "time": (["time"], TIME, {"axis": "T"}),
        },
    ),
}

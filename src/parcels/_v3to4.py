"""
Temporary utilities to help with the transition from v3 to v4 of Parcels.

TODO v4: Remove this module. Move functions that are still relevant into other modules
"""

from collections.abc import Callable

import xarray as xr


def Unit_to_units(d: dict) -> dict:
    if "Unit" in d:
        d["units"] = d.pop("Unit")
    return d


def xarray_patch_metadata(ds: xr.Dataset, f: Callable[[dict], dict]) -> xr.Dataset:
    """Convert attrs"""
    for var in ds.variables:
        ds[var].attrs = f(ds[var].attrs)
    return ds


def patch_dataset_v4_compat(ds: xr.Dataset) -> xr.Dataset:
    """Patches an xarray dataset to be compatible with v4"""
    return ds.pipe(xarray_patch_metadata, Unit_to_units)

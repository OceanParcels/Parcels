import datetime
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from parcels._typing import InterpMethodOption, PathLike
from parcels.tools.warnings import FileWarning


class NetcdfFileBuffer:
    def __init__(
        self,
        filename,
        dimensions,
        indices,
        *,
        interp_method: InterpMethodOption = "linear",
        gridindexingtype="nemo",
    ):
        self.filename: PathLike | list[PathLike] = filename
        self.dimensions = dimensions  # Dict with dimension keys for file data
        self.indices = indices
        self.dataset = None
        self.interp_method = interp_method
        self.gridindexingtype = gridindexingtype

    def __enter__(self):
        self.dataset = open_xarray_dataset(self.filename)
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None

    @property
    def xdim(self):
        lon = self.dataset[self.dimensions["lon"]]
        xdim = lon.size if len(lon.shape) == 1 else lon.shape[-1]
        if self.gridindexingtype in ["croco"]:
            xdim -= 1
        return xdim

    @property
    def ydim(self):
        lat = self.dataset[self.dimensions["lat"]]
        ydim = lat.size if len(lat.shape) == 1 else lat.shape[-2]
        if self.gridindexingtype in ["croco"]:
            ydim -= 1
        return ydim

    @property
    def zdim(self):
        if "depth" not in self.dimensions:
            return 1
        depth = self.dataset[self.dimensions["depth"]]
        zdim = depth.size if len(depth.shape) == 1 else depth.shape[-3]
        if self.gridindexingtype in ["croco"]:
            zdim -= 1
        return zdim

    @property
    def latlon(self):
        lon = self.dataset[self.dimensions["lon"]]
        lat = self.dataset[self.dimensions["lat"]]
        if self.gridindexingtype not in ["croco"]:
            if len(lon.shape) == 3:  # some lon, lat have a time dimension 1
                lon = lon[0, :, :]
                lat = lat[0, :, :]
            elif len(lon.shape) == 4:  # some lon, lat have a time and depth dimension 1
                lon = lon[0, 0, :, :]
                lat = lat[0, 0, :, :]

        if len(lon.shape) > 1:
            if is_rectilinear(lon, lat):
                lon = lon[0, :]
                lat = lat[:, 0]
        return lat, lon

    @property
    def depth(self):
        self.indices["depth"] = range(self.zdim)
        if "depth" in self.dimensions:
            return self.dataset[self.dimensions["depth"]]
        return np.zeros(1)

    @property
    def data_full_zdim(self):
        return self.zdim

    def _check_extend_depth(self, data, dim):
        return (
            self.indices["depth"][-1] == self.data_full_zdim - 1
            and data.shape[dim] == self.data_full_zdim - 1
            and self.interp_method in ["bgrid_velocity", "bgrid_w_velocity", "bgrid_tracer"]
        )

    def _apply_indices(self, data, ti):
        if len(data.shape) == 1:
            if self.indices["depth"] is not None:
                data = data[self.indices["depth"]]
        elif len(data.shape) == 3:
            if self._check_extend_depth(data, 0):
                data = data[self.indices["depth"][:-1], :, :]
            elif len(self.indices["depth"]) > 1:
                data = data[self.indices["depth"], :, :]
            else:
                data = data[ti, :, :]
        else:
            if self._check_extend_depth(data, 1):
                data = data[ti, self.indices["depth"][:-1], :, :]
            else:
                data = data[ti, self.indices["depth"], :, :]
        return data

    @property
    def data(self):
        return self.data_access()

    def data_access(self):
        data = self.dataset[self.name]
        ti = range(data.shape[0])
        return np.array(self._apply_indices(data, ti))

    @property
    def time(self):
        return self.time_access()

    def time_access(self):
        if "time" not in self.dimensions:
            return np.array([None])

        time_da = self.dataset[self.dimensions["time"]]
        time = (
            np.array([time_da[self.dimensions["time"]].data])
            if len(time_da.shape) == 0
            else np.array(time_da[self.dimensions["time"]])
        )
        if isinstance(time[0], datetime.datetime):
            raise NotImplementedError(
                "Parcels currently only parses dates ranging from 1678 AD to 2262 AD, which are stored by xarray as np.datetime64. If you need a wider date range, please open an Issue on the parcels github page."
            )
        return time


def open_xarray_dataset(filename: Path | str) -> xr.Dataset:
    try:
        # Unfortunately we need to do if-else here, cause the lock-parameter is either False or a Lock-object
        # (which we would rather want to have being auto-managed).
        # If 'lock' is not specified, the Lock-object is auto-created and managed by xarray internally.
        ds = xr.open_mfdataset(filename, decode_cf=True)
        ds["decoded"] = True
    except:
        warnings.warn(  # TODO: Is this warning necessary? What cases does this except block get triggered - is it to do with the bare except???
            f"File {filename} could not be decoded properly by xarray (version {xr.__version__}). "
            "It will be opened with no decoding. Filling values might be wrongly parsed.",
            FileWarning,
            stacklevel=2,
        )

        ds = xr.open_mfdataset(filename, decode_cf=False)
        ds["decoded"] = False
    return ds


def is_rectilinear(lon_subset, lat_subset) -> bool:
    """Test if all columns and rows are the same for lon and lat (in which case grid is rectilinear).

    lon_subset and lat_subset are 2D numpy arrays
    """
    for xi in range(1, lon_subset.shape[0]):
        if not np.allclose(lon_subset[0, :], lon_subset[xi, :]):
            return False

    for yi in range(1, lat_subset.shape[1]):
        if not np.allclose(lat_subset[:, 0], lat_subset[:, yi]):
            return False

    return True

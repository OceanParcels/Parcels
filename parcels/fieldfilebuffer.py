import datetime
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from parcels._typing import InterpMethodOption
from parcels.tools.converters import convert_xarray_time_units
from parcels.tools.warnings import FileWarning


class NetcdfFileBuffer:
    def __init__(
        self,
        filename,
        dimensions,
        indices,
        timestamp=None,
        interp_method: InterpMethodOption = "linear",
        data_full_zdim=None,
        cast_data_dtype=np.float32,
        gridindexingtype="nemo",
        **kwargs,
    ):
        self.filename = filename
        self.dimensions = dimensions  # Dict with dimension keys for file data
        self.indices = indices
        self.dataset = None
        self.timestamp = timestamp
        self.cast_data_dtype = cast_data_dtype
        self.ti = None
        self.interp_method = interp_method
        self.gridindexingtype = gridindexingtype
        self.data_full_zdim = data_full_zdim
        if ("lon" in self.indices) or ("lat" in self.indices):
            self.nolonlatindices = False
        else:
            self.nolonlatindices = True
        self.netcdf_engine = kwargs.pop("netcdf_engine", "netcdf4")

    def __enter__(self):
        self.dataset = open_xarray_dataset(self.filename, self.netcdf_engine)
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None

    @property
    def latlon(self):
        lon = self.dataset[self.dimensions["lon"]]
        lat = self.dataset[self.dimensions["lat"]]
        if self.nolonlatindices and self.gridindexingtype not in ["croco"]:
            if len(lon.shape) < 3:
                lon_subset = np.array(lon)
                lat_subset = np.array(lat)
            elif len(lon.shape) == 3:  # some lon, lat have a time dimension 1
                lon_subset = np.array(lon[0, :, :])
                lat_subset = np.array(lat[0, :, :])
            elif len(lon.shape) == 4:  # some lon, lat have a time and depth dimension 1
                lon_subset = np.array(lon[0, 0, :, :])
                lat_subset = np.array(lat[0, 0, :, :])
        else:
            xdim = lon.size if len(lon.shape) == 1 else lon.shape[-1]
            ydim = lat.size if len(lat.shape) == 1 else lat.shape[-2]
            if self.gridindexingtype in ["croco"]:
                xdim -= 1
                ydim -= 1
            self.indices["lon"] = self.indices["lon"] if "lon" in self.indices else range(xdim)
            self.indices["lat"] = self.indices["lat"] if "lat" in self.indices else range(ydim)
            if len(lon.shape) == 1:
                lon_subset = np.array(lon[self.indices["lon"]])
                lat_subset = np.array(lat[self.indices["lat"]])
            elif len(lon.shape) == 2:
                lon_subset = np.array(lon[self.indices["lat"], self.indices["lon"]])
                lat_subset = np.array(lat[self.indices["lat"], self.indices["lon"]])
            elif len(lon.shape) == 3:  # some lon, lat have a time dimension 1
                lon_subset = np.array(lon[0, self.indices["lat"], self.indices["lon"]])
                lat_subset = np.array(lat[0, self.indices["lat"], self.indices["lon"]])
            elif len(lon.shape) == 4:  # some lon, lat have a time and depth dimension 1
                lon_subset = np.array(lon[0, 0, self.indices["lat"], self.indices["lon"]])
                lat_subset = np.array(lat[0, 0, self.indices["lat"], self.indices["lon"]])

        if len(lon.shape) > 1:  # Tests if lon, lat are rectilinear but were stored in arrays
            rectilinear = True
            # test if all columns and rows are the same for lon and lat (in which case grid is rectilinear)
            for xi in range(1, lon_subset.shape[0]):
                if not np.allclose(lon_subset[0, :], lon_subset[xi, :]):
                    rectilinear = False
                    break
            if rectilinear:
                for yi in range(1, lat_subset.shape[1]):
                    if not np.allclose(lat_subset[:, 0], lat_subset[:, yi]):
                        rectilinear = False
                        break
            if rectilinear:
                lon_subset = lon_subset[0, :]
                lat_subset = lat_subset[:, 0]
        return lat_subset, lon_subset

    @property
    def depth(self):
        if "depth" in self.dimensions:
            depth = self.dataset[self.dimensions["depth"]]
            depthsize = depth.size if len(depth.shape) == 1 else depth.shape[-3]
            if self.gridindexingtype in ["croco"]:
                depthsize -= 1
            self.data_full_zdim = depthsize
            self.indices["depth"] = self.indices["depth"] if "depth" in self.indices else range(depthsize)
            if len(depth.shape) == 1:
                return np.array(depth[self.indices["depth"]])
            elif len(depth.shape) == 3:
                if self.nolonlatindices:
                    return np.array(depth[self.indices["depth"], :, :])
                else:
                    return np.array(depth[self.indices["depth"], self.indices["lat"], self.indices["lon"]])
            elif len(depth.shape) == 4:
                if self.nolonlatindices:
                    return np.array(depth[:, self.indices["depth"], :, :])
                else:
                    return np.array(depth[:, self.indices["depth"], self.indices["lat"], self.indices["lon"]])
        else:
            self.indices["depth"] = [0]
            return np.zeros(1)

    @property
    def depth_dimensions(self):
        if "depth" in self.dimensions:
            data = self.dataset[self.name]
            depthsize = data.shape[-3]
            self.data_full_zdim = depthsize
            self.indices["depth"] = self.indices["depth"] if "depth" in self.indices else range(depthsize)
            if self.nolonlatindices:
                return np.empty((0, len(self.indices["depth"])) + data.shape[-2:])
            else:
                return np.empty((0, len(self.indices["depth"]), len(self.indices["lat"]), len(self.indices["lon"])))

    def _check_extend_depth(self, data, di):
        return (
            self.indices["depth"][-1] == self.data_full_zdim - 1
            and data.shape[di] == self.data_full_zdim - 1
            and self.interp_method in ["bgrid_velocity", "bgrid_w_velocity", "bgrid_tracer"]
        )

    def _apply_indices(self, data, ti):
        if len(data.shape) == 1:
            if self.indices["depth"] is not None:
                data = data[self.indices["depth"]]
        elif len(data.shape) == 2:
            if self.nolonlatindices:
                pass
            else:
                data = data[self.indices["lat"], self.indices["lon"]]
        elif len(data.shape) == 3:
            if self._check_extend_depth(data, 0):
                if self.nolonlatindices:
                    data = data[self.indices["depth"][:-1], :, :]
                else:
                    data = data[self.indices["depth"][:-1], self.indices["lat"], self.indices["lon"]]
            elif len(self.indices["depth"]) > 1:
                if self.nolonlatindices:
                    data = data[self.indices["depth"], :, :]
                else:
                    data = data[self.indices["depth"], self.indices["lat"], self.indices["lon"]]
            else:
                if self.nolonlatindices:
                    data = data[ti, :, :]
                else:
                    data = data[ti, self.indices["lat"], self.indices["lon"]]
        else:
            if self._check_extend_depth(data, 1):
                if self.nolonlatindices:
                    data = data[ti, self.indices["depth"][:-1], :, :]
                else:
                    data = data[ti, self.indices["depth"][:-1], self.indices["lat"], self.indices["lon"]]
            else:
                if self.nolonlatindices:
                    data = data[ti, self.indices["depth"], :, :]
                else:
                    data = data[ti, self.indices["depth"], self.indices["lat"], self.indices["lon"]]
        return data

    @property
    def data(self):
        return self.data_access()

    def data_access(self):
        data = self.dataset[self.name]
        ti = range(data.shape[0]) if self.ti is None else self.ti
        data = self._apply_indices(data, ti)
        return np.array(data, dtype=self.cast_data_dtype)

    @property
    def time(self):
        return self.time_access()

    def time_access(self):
        if self.timestamp is not None:
            return self.timestamp

        if "time" not in self.dimensions:
            return np.array([None])

        time_da = self.dataset[self.dimensions["time"]]
        convert_xarray_time_units(time_da, self.dimensions["time"])
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


def open_xarray_dataset(filename: Path | str, netcdf_engine: str) -> xr.Dataset:
    try:
        # Unfortunately we need to do if-else here, cause the lock-parameter is either False or a Lock-object
        # (which we would rather want to have being auto-managed).
        # If 'lock' is not specified, the Lock-object is auto-created and managed by xarray internally.
        ds = xr.open_dataset(filename, decode_cf=True, engine=netcdf_engine)
        ds["decoded"] = True
    except:
        warnings.warn(  # TODO: Is this warning necessary? What cases does this except block get triggered - is it to do with the bare except???
            f"File {filename} could not be decoded properly by xarray (version {xr.__version__}). "
            "It will be opened with no decoding. Filling values might be wrongly parsed.",
            FileWarning,
            stacklevel=2,
        )

        ds = xr.open_dataset(filename, decode_cf=False, engine=netcdf_engine)
        ds["decoded"] = False
    return ds

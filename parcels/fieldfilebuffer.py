import datetime
import warnings

import dask.array as da
import numpy as np
import xarray as xr

from parcels._typing import InterpMethodOption
from parcels.tools.converters import convert_xarray_time_units
from parcels.tools.warnings import FileWarning


class _FileBuffer:
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


class NetcdfFileBuffer(_FileBuffer):
    def __init__(self, *args, **kwargs):
        self.lib = np
        self.netcdf_engine = kwargs.pop("netcdf_engine", "netcdf4")
        super().__init__(*args, **kwargs)

    def __enter__(self):
        try:
            # Unfortunately we need to do if-else here, cause the lock-parameter is either False or a Lock-object
            # (which we would rather want to have being auto-managed).
            # If 'lock' is not specified, the Lock-object is auto-created and managed by xarray internally.
            self.dataset = xr.open_dataset(str(self.filename), decode_cf=True, engine=self.netcdf_engine)
            self.dataset["decoded"] = True
        except:
            warnings.warn(
                f"File {self.filename} could not be decoded properly by xarray (version {xr.__version__}). "
                "It will be opened with no decoding. Filling values might be wrongly parsed.",
                FileWarning,
                stacklevel=2,
            )

            self.dataset = xr.open_dataset(str(self.filename), decode_cf=False, engine=self.netcdf_engine)
            self.dataset["decoded"] = False
        for inds in self.indices.values():
            if type(inds) not in [list, range]:
                raise RuntimeError("Indices for field subsetting need to be a list")
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None

    def parse_name(self, name):
        if isinstance(name, list):
            for nm in name:
                if hasattr(self.dataset, nm):
                    name = nm
                    break
        if isinstance(name, list):
            raise OSError("None of variables in list found in file")
        return name

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


class DeferredNetcdfFileBuffer(NetcdfFileBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DaskFileBuffer(NetcdfFileBuffer):
    _static_name_maps = {
        "time": ["time", "time_count", "time_counter", "timer_count", "t"],
        "depth": [
            "depth",
            "depthu",
            "depthv",
            "depthw",
            "depths",
            "deptht",
            "depthx",
            "depthy",
            "depthz",
            "z",
            "z_u",
            "z_v",
            "z_w",
            "d",
            "k",
            "w_dep",
            "w_deps",
            "Z",
            "Zp1",
            "Zl",
            "Zu",
            "level",
        ],
        "lat": ["lat", "nav_lat", "y", "latitude", "la", "lt", "j", "YC", "YG"],
        "lon": ["lon", "nav_lon", "x", "longitude", "lo", "ln", "i", "XC", "XG"],
    }

    """ Class that encapsulates and manages deferred access to file data. """

    def __init__(self, *args, **kwargs):
        """Initializes this specific filebuffer type. As a result of using dask, the internal library is set to 'da'."""
        self.lib = da
        self.lock_file = kwargs.pop("lock_file", True)
        super().__init__(*args, **kwargs)

    def __enter__(self):
        """
        This function enters the physical file (equivalent to a 'with open(...)' statement) and returns a file object.
        In Dask, with dynamic loading, this is the point where we have access to the header-information of the file.
        """
        try:
            # Unfortunately we need to do if-else here, cause the lock-parameter is either False or a Lock-object
            # (which we would rather want to have being auto-managed).
            # If 'lock' is not specified, the Lock-object is auto-created and managed by xarray internally.
            if self.lock_file:
                self.dataset = xr.open_dataset(str(self.filename), decode_cf=True, engine=self.netcdf_engine)
            else:
                self.dataset = xr.open_dataset(
                    str(self.filename), decode_cf=True, engine=self.netcdf_engine, lock=False
                )
            self.dataset["decoded"] = True
        except:
            warnings.warn(
                f"File {self.filename} could not be decoded properly by xarray (version {xr.__version__}). "
                "It will be opened with no decoding. Filling values might be wrongly parsed.",
                FileWarning,
                stacklevel=2,
            )
            if self.lock_file:
                self.dataset = xr.open_dataset(str(self.filename), decode_cf=False, engine=self.netcdf_engine)
            else:
                self.dataset = xr.open_dataset(
                    str(self.filename), decode_cf=False, engine=self.netcdf_engine, lock=False
                )
            self.dataset["decoded"] = False

        for inds in self.indices.values():
            if type(inds) not in [list, range]:
                raise RuntimeError("Indices for field subsetting need to be a list")
        return self

    def __exit__(self, type, value, traceback):
        """Function releases the file handle.

        This function releases the file handle. Hence access to the dataset and its header-information is lost. The
        previously executed chunking is lost. Furthermore, if the file access required file locking, the lock-handle
        is freed so other processes can now access the file again.
        """
        self.close()

    def close(self):
        """Teardown FileBuffer object with dask.

        This function can be called to initialise an orderly teardown of a FileBuffer object with dask, meaning
        to release the file handle, deposing the dataset, and releasing the file lock (if required).
        """
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None

    @classmethod
    def add_to_dimension_name_map_global(cls, name_map):
        """
        [externally callable]
        This function adds entries to the name map from parcels_dim -> netcdf_dim. This is required if you want to
        use auto-chunking on large fields whose map parameters are not defined. This function must be called before
        entering the filebuffer object. Example:
        DaskFileBuffer.add_to_dimension_name_map_global({'lat': 'nydim',
                                                         'lon': 'nxdim',
                                                         'time': 'ntdim',
                                                         'depth': 'nddim'})
        [...]
        Note that not all parcels dimensions need to be present in 'name_map'.
        """
        assert isinstance(name_map, dict)
        for pcls_dim_name in name_map.keys():
            if isinstance(name_map[pcls_dim_name], list):
                for nc_dim_name in name_map[pcls_dim_name]:
                    cls._static_name_maps[pcls_dim_name].append(nc_dim_name)
            elif isinstance(name_map[pcls_dim_name], str):
                cls._static_name_maps[pcls_dim_name].append(name_map[pcls_dim_name])

    def add_to_dimension_name_map(self, name_map):
        """
        [externally callable]
        This function adds entries to the name map from parcels_dim -> netcdf_dim. This is required if you want to
        use auto-chunking on large fields whose map parameters are not defined. This function must be called after
        constructing an filebuffer object and before entering the filebuffer. Example:
        fb = DaskFileBuffer(...)
        fb.add_to_dimension_name_map({'lat': 'nydim', 'lon': 'nxdim', 'time': 'ntdim', 'depth': 'nddim'})
        with fb:
            [do_stuff}
        Note that not all parcels dimensions need to be present in 'name_map'.
        """
        assert isinstance(name_map, dict)
        for pcls_dim_name in name_map.keys():
            self._static_name_maps[pcls_dim_name].append(name_map[pcls_dim_name])

    def _get_available_dims_indices_by_request(self):
        """Returns a dict mapping 'parcels_dimname' -> [None, int32_index_data_array].

        This dictionary is based on the information provided by the requested dimensions.
        Example: {'time': 0, 'depth': None, 'lat': 1, 'lon': 2}
        """
        result = {}
        neg_offset = 0
        tpl_offset = 0
        for name in ["time", "depth", "lat", "lon"]:
            i = list(self._static_name_maps.keys()).index(name)
            if name not in self.dimensions:
                result[name] = None
                tpl_offset += 1
                neg_offset += 1
            else:
                result[name] = i - neg_offset
        return result

    def _get_available_dims_indices_by_namemap(self):
        """
        Returns a dict mapping 'parcels_dimname' -> [None, int32_index_data_array].
        This dictionary is based on the information provided by the requested dimensions.
        Example: {'time': 0, 'depth': 1, 'lat': 2, 'lon': 3}
        """
        result = {}
        for name in ["time", "depth", "lat", "lon"]:
            result[name] = list(self._static_name_maps.keys()).index(name)
        return result

    def _get_available_dims_indices_by_netcdf_file(self):
        """
        [File needs to be open (i.e. self.dataset is not None) for this to work - otherwise generating an error]
        Returns a dict mapping 'parcels_dimname' -> [None, int32_index_data_array].
        This dictionary is based on the information provided by the requested dimensions.
        Example: {'time': 0, 'depth': 5, 'lat': 3, 'lon': 1}
                 for NetCDF with dimensions:
                     timer: 1
                     x: [0 4000]
                     xr: [0 3999]
                     y: [0 2140]
                     yr: [0 2139]
                     z: [0 75]
        """
        if self.dataset is None:
            raise OSError("Trying to parse NetCDF header information before opening the file.")
        result = {}
        for pcls_dimname in ["time", "depth", "lat", "lon"]:
            for nc_dimname in self._static_name_maps[pcls_dimname]:
                if nc_dimname not in self.dataset.sizes.keys():
                    continue
                result[pcls_dimname] = list(self.dataset.sizes.keys()).index(nc_dimname)
        return result

    def _is_dimension_available(self, dimension_name):
        """
        This function returns a boolean value indicating if a certain variable (name) is available in the
        requested dimensions as well as in the actual dataset of the file. If any of the two conditions is not met,
        if returns 'False'.
        """
        if self.dimensions is None or self.dataset is None:
            return False
        return dimension_name in self.dimensions

    def _is_dimension_in_dataset(self, parcels_dimension_name, netcdf_dimension_name=None):
        """
        [File needs to be open (i.e. self.dataset is not None) for this to work - otherwise generating an error]
        This function returns the index, the name and the size of a NetCDF dimension in the file (in order: index, name, size).
        It requires as input the name of the related parcels dimension (i.e. one of ['time', 'depth', 'lat', 'lon']. If
        no hint on its mapping to a NetCDF dimension is provided, a heuristic based on the pre-defined name dictionary
        is used. If a hint is provided, a connections is made between the designated parcels-dimension and NetCDF dimension.
        """
        if self.dataset is None:
            raise OSError("Trying to parse NetCDF header information before opening the file.")
        k, dname, dvalue = (-1, "", 0)
        dimension_name = parcels_dimension_name.lower()
        dim_indices = self._get_available_dims_indices_by_request()
        i = dim_indices[dimension_name]
        if netcdf_dimension_name is not None and netcdf_dimension_name in self.dataset.sizes.keys():
            value = self.dataset.sizes[netcdf_dimension_name]
            k, dname, dvalue = i, netcdf_dimension_name, value
        elif self.dimensions is None or self.dataset is None:
            return k, dname, dvalue
        else:
            for name in self._static_name_maps[dimension_name]:
                if name in self.dataset.sizes:
                    value = self.dataset.sizes[name]
                    k, dname, dvalue = i, name, value
                    break
        return k, dname, dvalue

    @property
    def data(self):
        return self.data_access()

    def data_access(self):
        data = self.dataset[self.name]

        ti = range(data.shape[0]) if self.ti is None else self.ti
        data = self._apply_indices(data, ti)
        if isinstance(data, xr.DataArray):
            data = data.data

        return data.astype(self.cast_data_dtype)


class DeferredDaskFileBuffer(DaskFileBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

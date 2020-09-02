import dask.array as da
from dask import config as da_conf
from dask import utils as da_utils
import numpy as np
import xarray as xr
from netCDF4 import Dataset as ncDataset

import datetime
import math
import psutil

from parcels.tools.converters import convert_xarray_time_units
from parcels.tools.loggers import logger


class _FileBuffer(object):
    def __init__(self, filename, dimensions, indices, timestamp=None,
                 interp_method='linear', data_full_zdim=None, **kwargs):
        self.filename = filename
        self.dimensions = dimensions  # Dict with dimension keys for file data
        self.indices = indices
        self.dataset = None
        self.timestamp = timestamp
        self.ti = None
        self.interp_method = interp_method
        self.data_full_zdim = data_full_zdim


class XarrayFileBuffer(_FileBuffer):
    def __init__(self, *args, **kwargs):
        super(XarrayFileBuffer, self).__init__(*args, **kwargs)

    def __enter__(self):
        self.dataset = self.filename
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        pass


class NetcdfFileBuffer(_FileBuffer):
    def __init__(self, *args, **kwargs):
        self.lib = np
        self.netcdf_engine = kwargs.pop('netcdf_engine', 'netcdf4')
        super(NetcdfFileBuffer, self).__init__(*args, **kwargs)

    def __enter__(self):
        try:
            # Unfortunately we need to do if-else here, cause the lock-parameter is either False or a Lock-object
            # (which we would rather want to have being auto-managed).
            # If 'lock' is not specified, the Lock-object is auto-created and managed bz xarray internally.
            self.dataset = xr.open_dataset(str(self.filename), decode_cf=True, engine=self.netcdf_engine)
            self.dataset['decoded'] = True
        except:
            logger.warning_once("File %s could not be decoded properly by xarray (version %s).\n         "
                                "It will be opened with no decoding. Filling values might be wrongly parsed."
                                % (self.filename, xr.__version__))
            self.dataset = xr.open_dataset(str(self.filename), decode_cf=False, engine=self.netcdf_engine)
            self.dataset['decoded'] = False
        for inds in self.indices.values():
            if type(inds) not in [list, range]:
                raise RuntimeError('Indices for field subsetting need to be a list')
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
            raise IOError('None of variables in list found in file')
        return name

    @property
    def read_lonlat(self):
        lon = self.dataset[self.dimensions['lon']]
        lat = self.dataset[self.dimensions['lat']]
        xdim = lon.size if len(lon.shape) == 1 else lon.shape[-1]
        ydim = lat.size if len(lat.shape) == 1 else lat.shape[-2]
        self.indices['lon'] = self.indices['lon'] if 'lon' in self.indices else range(xdim)
        self.indices['lat'] = self.indices['lat'] if 'lat' in self.indices else range(ydim)
        if len(lon.shape) == 1:
            lon_subset = np.array(lon[self.indices['lon']])
            lat_subset = np.array(lat[self.indices['lat']])
        elif len(lon.shape) == 2:
            lon_subset = np.array(lon[self.indices['lat'], self.indices['lon']])
            lat_subset = np.array(lat[self.indices['lat'], self.indices['lon']])
        elif len(lon.shape) == 3:  # some lon, lat have a time dimension 1
            lon_subset = np.array(lon[0, self.indices['lat'], self.indices['lon']])
            lat_subset = np.array(lat[0, self.indices['lat'], self.indices['lon']])
        elif len(lon.shape) == 4:  # some lon, lat have a time and depth dimension 1
            lon_subset = np.array(lon[0, 0, self.indices['lat'], self.indices['lon']])
            lat_subset = np.array(lat[0, 0, self.indices['lat'], self.indices['lon']])
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
        return lon_subset, lat_subset

    @property
    def read_depth(self):
        if 'depth' in self.dimensions:
            depth = self.dataset[self.dimensions['depth']]
            depthsize = depth.size if len(depth.shape) == 1 else depth.shape[-3]
            self.data_full_zdim = depthsize
            self.indices['depth'] = self.indices['depth'] if 'depth' in self.indices else range(depthsize)
            if len(depth.shape) == 1:
                return np.array(depth[self.indices['depth']])
            elif len(depth.shape) == 3:
                return np.array(depth[self.indices['depth'], self.indices['lat'], self.indices['lon']])
            elif len(depth.shape) == 4:
                return np.array(depth[:, self.indices['depth'], self.indices['lat'], self.indices['lon']])
        else:
            self.indices['depth'] = [0]
            return np.zeros(1)

    @property
    def read_depth_dimensions(self):
        if 'depth' in self.dimensions:
            data = self.dataset[self.name]
            depthsize = data.shape[-3]
            self.data_full_zdim = depthsize
            self.indices['depth'] = self.indices['depth'] if 'depth' in self.indices else range(depthsize)
            return np.empty((0, len(self.indices['depth']), len(self.indices['lat']), len(self.indices['lon'])))

    def _check_extend_depth(self, data, di):
        return (self.indices['depth'][-1] == self.data_full_zdim-1
                and data.shape[di] == self.data_full_zdim-1
                and self.interp_method in ['bgrid_velocity', 'bgrid_w_velocity', 'bgrid_tracer'])

    def _extend_depth_dimension(self, data, ti):
        # Add a bottom level of zeros for B-grid if missing in the data.
        # The last level is unused by B-grid interpolator (U, V, tracer) but must be there
        # to match Parcels data shape. for W, last level must be 0 for impermeability
        for dim in ['depth', 'lat', 'lon']:
            if not isinstance(self.indices[dim], (list, range)):
                raise NotImplementedError("For B grids, indices must be provided as a range")
                # this is because da.concatenate needs data which are indexed using slices, not a range of indices
        d0 = self.indices['depth'][0]
        d1 = self.indices['depth'][-1] + 1
        lat0 = self.indices['lat'][0]
        lat1 = self.indices['lat'][-1] + 1
        lon0 = self.indices['lon'][0]
        lon1 = self.indices['lon'][-1] + 1
        if len(data.shape) == 3:
            data = self.lib.concatenate((data[d0:d1 - 1, lat0:lat1, lon0:lon1],
                                        da.zeros((1, lat1 - lat0, lon1 - lon0))), axis=0)
        else:
            if (type(ti) in [list, range]):
                t0 = ti[0]
                t1 = ti[-1] + 1
                data = self.lib.concatenate((data[t0:t1, d0:d1 - 1, lat0:lat1, lon0:lon1],
                                             self.lib.zeros((t1 - t0, 1, lat1 - lat0, lon1 - lon0))), axis=1)
            else:
                data = self.lib.concatenate((data[ti, d0:d1 - 1, lat0:lat1, lon0:lon1],
                                             self.lib.zeros((1, lat1 - lat0, lon1 - lon0))), axis=0)

        return data

    def _apply_indices(self, data, ti):
        if len(data.shape) == 2:
            data = data[self.indices['lat'], self.indices['lon']]
        elif len(data.shape) == 3:
            if self._check_extend_depth(data, 0):
                data = self._extend_depth_dimension(data, ti)
            elif len(self.indices['depth']) > 1:
                data = data[self.indices['depth'], self.indices['lat'], self.indices['lon']]
            else:
                data = data[ti, self.indices['lat'], self.indices['lon']]
        else:
            if self._check_extend_depth(data, 1):
                data = self._extend_depth_dimension(data, ti)
            else:
                data = data[ti, self.indices['depth'], self.indices['lat'], self.indices['lon']]

        return data

    @property
    def data(self):
        return self.data_access()

    def data_access(self):
        data = self.dataset[self.name]
        ti = range(data.shape[0]) if self.ti is None else self.ti
        data = self._apply_indices(data, ti)
        return np.array(data)

    @property
    def time(self):
        return self.time_access()

    def time_access(self):
        if self.timestamp is not None:
            return self.timestamp

        if 'time' not in self.dimensions:
            return np.array([None])

        time_da = self.dataset[self.dimensions['time']]
        convert_xarray_time_units(time_da, self.dimensions['time'])
        time = np.array([time_da[self.dimensions['time']]]) if len(time_da.shape) == 0 else np.array(time_da[self.dimensions['time']])
        if isinstance(time[0], datetime.datetime):
            raise NotImplementedError('Parcels currently only parses dates ranging from 1678 AD to 2262 AD, which are stored by xarray as np.datetime64. If you need a wider date range, please open an Issue on the parcels github page.')
        return time


class DeferredNetcdfFileBuffer(NetcdfFileBuffer):
    def __init__(self, *args, **kwargs):
        super(DeferredNetcdfFileBuffer, self).__init__(*args, **kwargs)


class DaskFileBuffer(NetcdfFileBuffer):
    _name_maps = {'lon': ['lon', 'nav_lon', 'x', 'longitude', 'lo', 'ln', 'i', 'XC', 'XG'],
                  'lat': ['lat', 'nav_lat', 'y', 'latitude', 'la', 'lt', 'j', 'YC', 'YG'],
                  'depth': ['depth', 'depthu', 'depthv', 'depthw', 'depths', 'deptht', 'depthx', 'depthy', 'depthz',
                            'z', 'z_u', 'z_v', 'z_w', 'd', 'k', 'w_dep', 'w_deps', 'Z', 'Zp1', 'Zl', 'Zu', 'level'],
                  'time': ['time', 'time_count', 'time_counter', 'timer_count', 't']}
    _min_dim_chunksize = 16

    """ Class that encapsulates and manages deferred access to file data. """
    def __init__(self, *args, **kwargs):
        self.lib = da
        self.field_chunksize = kwargs.pop('field_chunksize', 'auto')
        self.lock_file = kwargs.pop('lock_file', True)
        self.chunk_mapping = None
        self.rechunk_callback_fields = kwargs.pop('rechunk_callback_fields', None)
        self.chunking_finalized = False
        if "chunkdims_name_map" in kwargs.keys() and kwargs["chunkdims_name_map"] is not None and isinstance(kwargs["chunkdims_name_map"], dict):
            for key, dim_name_arr in kwargs["chunkdims_name_map"].items():
                for value in dim_name_arr:
                    if value not in self._name_maps[key]:
                        self._name_maps[key].append(value)
        super(DaskFileBuffer, self).__init__(*args, **kwargs)

    def __enter__(self):
        if self.field_chunksize not in [False, None, 'auto'] and type(self.field_chunksize) not in [list, tuple, dict]:
            raise AttributeError("'field_chunksize' is of wrong type. Parameter is expected to be a list, tuple or dict per data dimension, or be False, None or 'auto'.")
        if isinstance(self.field_chunksize, list):
            self.field_chunksize = tuple(self.field_chunksize)

        init_chunk_dict = None
        if self.field_chunksize not in [False, None]:
            init_chunk_dict = self._get_initial_chunk_dictionary()
        try:
            # Unfortunately we need to do if-else here, cause the lock-parameter is either False or a Lock-object
            # (which we would rather want to have being auto-managed).
            # If 'lock' is not specified, the Lock-object is auto-created and managed bz xarray internally.
            if self.lock_file:
                self.dataset = xr.open_dataset(str(self.filename), decode_cf=True, engine=self.netcdf_engine, chunks=init_chunk_dict)
            else:
                self.dataset = xr.open_dataset(str(self.filename), decode_cf=True, engine=self.netcdf_engine, chunks=init_chunk_dict, lock=False)
            self.dataset['decoded'] = True
        except:
            logger.warning_once("File %s could not be decoded properly by xarray (version %s).\n         It will be opened with no decoding. Filling values might be wrongly parsed."
                                % (self.filename, xr.__version__))
            if self.lock_file:
                self.dataset = xr.open_dataset(str(self.filename), decode_cf=False, engine=self.netcdf_engine, chunks=init_chunk_dict)
            else:
                self.dataset = xr.open_dataset(str(self.filename), decode_cf=False, engine=self.netcdf_engine, chunks=init_chunk_dict, lock=False)
            self.dataset['decoded'] = False

        for inds in self.indices.values():
            if type(inds) not in [list, range]:
                raise RuntimeError('Indices for field subsetting need to be a list')
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        if self.dataset is not None:
            self.dataset.close()
            self.dataset = None
        self.chunking_finalized = False
        self.chunk_mapping = None

    def _get_initial_chunk_dictionary(self):
        # ==== check-opening requested dataset to access metadata                   ==== #
        # ==== file-opening and dimension-reading does not require a decode or lock ==== #
        self.dataset = xr.open_dataset(str(self.filename), decode_cf=False, engine=self.netcdf_engine, chunks={}, lock=False)
        self.dataset['decoded'] = False
        # ==== self.dataset temporarily available ==== #
        init_chunk_dict = {}
        if isinstance(self.field_chunksize, dict):
            # init_chunk_dict = self.field_chunksize
            loni, lonname, _ = self._is_dimension_in_dataset('lon')
            lati, latname, _ = self._is_dimension_in_dataset('lat')
            depthi, depthname, _ = self._is_dimension_in_dataset('depth')
            timei, timename, _ = self._is_dimension_in_dataset('time')
            for name in self.field_chunksize.keys():
                if name in [lonname, latname, depthname, timename]:
                    init_chunk_dict[name] = self.field_chunksize[name]
        elif isinstance(self.field_chunksize, tuple):  # and (len(self.dimensions) == len(self.field_chunksize)):
            tmp_chs = [0, ] * len(self.field_chunksize)
            chunk_index = len(self.field_chunksize)-1

            loni, lonname, _ = self._is_dimension_in_dataset('lon')
            if loni >= 0 and chunk_index >= 0:
                init_chunk_dict[lonname] = self.field_chunksize[chunk_index]
                tmp_chs[chunk_index] = self.field_chunksize[chunk_index]
            else:
                logger.warning_once(self._netcdf_DimNotFound_warning_message('lon'))
            chunk_index -= 1

            lati, latname, _ = self._is_dimension_in_dataset('lat')
            if lati >= 0 and chunk_index >= 0:
                init_chunk_dict[latname] = self.field_chunksize[chunk_index]
                tmp_chs[chunk_index] = self.field_chunksize[chunk_index]
            else:
                logger.warning_once(self._netcdf_DimNotFound_warning_message('lat'))
            chunk_index -= 1

            depthi, depthname, _ = self._is_dimension_in_dataset('depth')
            if depthi >= 0 and chunk_index >= 0:
                if self._is_dimension_available('depth'):
                    init_chunk_dict[depthname] = self.field_chunksize[chunk_index]
                    tmp_chs[chunk_index] = self.field_chunksize[chunk_index]
            elif depthname:
                logger.warning_once(self._netcdf_DimNotFound_warning_message('depth'))
            chunk_index -= 1

            timei, timename, _ = self._is_dimension_in_dataset('time')
            if timei >= 0 and chunk_index >= 0:
                if self._is_dimension_available('time'):
                    init_chunk_dict[timename] = self.field_chunksize[chunk_index]
                    tmp_chs[chunk_index] = self.field_chunksize[chunk_index]
            elif timename:
                logger.warning_once(self._netcdf_DimNotFound_warning_message('time'))
            chunk_index -= 1

            # ==== re-arrange the tupe and correct for empty dimensions ==== #
            for chunk_index in range(len(self.field_chunksize)-1, -1, -1):
                if tmp_chs[chunk_index] < 1:
                    tmp_chs.pop(chunk_index)
            self.field_chunksize = tuple(tmp_chs)
        elif self.field_chunksize == 'auto':
            av_mem = psutil.virtual_memory().available
            chunk_cap = av_mem * (1/8) * (1/3)
            if 'array.chunk-size' in da_conf.config.keys():
                chunk_cap = da_utils.parse_bytes(da_conf.config.get('array.chunk-size'))
            else:
                predefined_cap = da_conf.get('array.chunk-size')
                if predefined_cap is not None:
                    chunk_cap = da_utils.parse_bytes(predefined_cap)
                else:
                    logger.info_once("Unable to locate chunking hints from dask, thus estimating the max. chunk size heuristically. Please consider defining the 'chunk-size' for 'array' in your local dask configuration file (see http://oceanparcels.org/faq.html#field_chunking_config and https://docs.dask.org).")
            loni, lonname, lonvalue = self._is_dimension_in_dataset('lon')
            lati, latname, latvalue = self._is_dimension_in_dataset('lat')
            if lati >= 0 and loni >= 0:
                pDim = int(math.floor(math.sqrt(chunk_cap/np.dtype(np.float64).itemsize)))
                init_chunk_dict[latname] = min(latvalue, pDim)
                init_chunk_dict[lonname] = min(lonvalue, pDim)
            timei, timename, _ = self._is_dimension_in_dataset('time')
            if timei >= 0:
                init_chunk_dict[timename] = 1
            depthi, depthname, depthvalue = self._is_dimension_in_dataset('depth')
            if depthi >= 0:
                init_chunk_dict[depthname] = max(1, depthvalue)
        # ==== closing check-opened requested dataset ==== #
        self.dataset.close()
        # ==== check if the chunksize reading is successful. if not, load the file ONCE really into memory and ==== #
        # ==== deduce the chunking from the array dims.                                                         ==== #
        try:
            if len(init_chunk_dict) < 3:
                raise AttributeError("Too few known chunk dimension arguments.")
            self.dataset = xr.open_dataset(str(self.filename), decode_cf=True, engine=self.netcdf_engine, chunks=init_chunk_dict, lock=False)
        except:
            # ==== fail - open it as a normal array and deduce the dimensions from the read field ==== #
            init_chunk_dict = {}
            self.dataset = ncDataset(str(self.filename))
            refdims = self.dataset.dimensions.keys()
            max_field = ""
            max_dim_names = ()
            max_overlay_dims = 0
            for vname in self.dataset.variables:
                var = self.dataset.variables[vname]
                overlay_dims = []
                for vdname in var.dimensions:
                    if vdname in refdims:
                        overlay_dims.append(vdname)
                n_overlay_dims = len(overlay_dims)
                if n_overlay_dims > max_overlay_dims:
                    max_field = vname
                    max_dim_names = tuple(overlay_dims)
                    max_overlay_dims = n_overlay_dims
            self.name = max_field
            for dname in max_dim_names:
                if isinstance(self.field_chunksize, dict):
                    if dname in self.field_chunksize.keys():
                        init_chunk_dict[dname] = min(self.field_chunksize[dname], self.dataset.dimensions[dname].size)
                        continue
                init_chunk_dict[dname] = min(self._min_dim_chunksize, self.dataset.dimensions[dname].size)
            # ==== because in this case it has shown that the requested field_chunksize setup cannot be used, ==== #
            # ==== replace the requested field_chunksize with this auto-derived version.                      ==== #
            self.field_chunksize = init_chunk_dict
        finally:
            self.dataset.close()
        self.dataset = None
        # ==== self.dataset not available ==== #
        return init_chunk_dict

    def _is_dimension_available(self, dimension_name):
        if self.dimensions is None or self.dataset is None:
            return False
        return dimension_name in self.dimensions

    def _is_dimension_in_dataset(self, dimension_name):
        k, dname, dvalue = (-1, '', 0)
        if self.dimensions is None or self.dataset is None:
            return k, dname, dvalue
        dimension_name = dimension_name.lower()
        for i, name in enumerate(self._name_maps[dimension_name]):
            if name in self.dataset.dims:
                value = self.dataset.dims[name]
                k, dname, dvalue = i, name, value
                break
        return k, dname, dvalue

    def _is_dimension_in_chunksize_request(self, dimension_name):
        k, dname, dvalue = (-1, '', 0)
        if self.dimensions is None or self.dataset is None:
            return k, dname, dvalue
        dimension_name = dimension_name.lower()
        for i, name in enumerate(self._name_maps[dimension_name]):
            if name in self.field_chunksize:
                value = self.field_chunksize[name]
                k, dname, dvalue = i, name, value
                break
        return k, dname, dvalue

    def _netcdf_DimNotFound_warning_message(self, dimension_name):
        display_name = dimension_name if (dimension_name not in self.dimensions) else self.dimensions[dimension_name]
        return "Did not find {} in NetCDF dims. Please specifiy field_chunksize as dictionary for NetCDF dimension names, e.g.\n field_chunksize={{ '{}': <number>, ... }}.".format(display_name, display_name)

    def _chunksize_to_chunkmap(self):
        if self.field_chunksize in [False, 'auto', None]:
            return
        self.chunk_mapping = {}
        if isinstance(self.field_chunksize, tuple):
            for i in range(len(self.field_chunksize)):
                self.chunk_mapping[i] = self.field_chunksize[i]
        else:
            timei, timename, timevalue = self._is_dimension_in_chunksize_request('time')
            dtimei, dtimename, dtimevalue = self._is_dimension_in_dataset('time')
            depthi, depthname, depthvalue = self._is_dimension_in_chunksize_request('depth')
            ddepthi, ddepthname, ddepthvalue = self._is_dimension_in_dataset('depth')
            lati, latname, latvalue = self._is_dimension_in_chunksize_request('lat')
            loni, lonname, lonvalue = self._is_dimension_in_chunksize_request('lon')
            dim_index = 0
            if len(self.field_chunksize) == 2:
                self.chunk_mapping[dim_index] = latvalue
                dim_index += 1
                self.chunk_mapping[dim_index] = lonvalue
                dim_index += 1
            elif len(self.field_chunksize) >= 3:
                if timei >= 0 and timevalue > 1 and dtimei >= 0 and dtimevalue > 1 and self._is_dimension_available('time'):
                    self.chunk_mapping[dim_index] = 1  # still need to make sure that we only load 1 time step at a time
                    dim_index += 1
                if depthi >= 0 and depthvalue > 1 and ddepthi >= 0 and ddepthvalue > 1 and self._is_dimension_available('depth'):
                    self.chunk_mapping[dim_index] = depthvalue
                    dim_index += 1
                self.chunk_mapping[dim_index] = latvalue
                dim_index += 1
                self.chunk_mapping[dim_index] = lonvalue
                dim_index += 1

    def _chunkmap_to_chunksize(self):
        if self.field_chunksize in [False, None]:
            return
        self.field_chunksize = {}
        chunk_map = self.chunk_mapping
        timei, _, timevalue = self._is_dimension_in_dataset('time')
        depthi, _, depthvalue = self._is_dimension_in_dataset('depth')
        if len(chunk_map) == 2:
            self.field_chunksize[self.dimensions['lat']] = chunk_map[0]
            self.field_chunksize[self.dimensions['lon']] = chunk_map[1]
        elif len(chunk_map) == 3:
            chunk_dim_index = 0
            if depthi >= 0 and depthvalue > 1 and self._is_dimension_available('depth'):
                self.field_chunksize[self.dimensions['depth']] = chunk_map[chunk_dim_index]
                chunk_dim_index += 1
            elif timei >= 0 and timevalue > 1 and self._is_dimension_available('time'):
                self.field_chunksize[self.dimensions['time']] = chunk_map[chunk_dim_index]
                chunk_dim_index += 1
            self.field_chunksize[self.dimensions['lat']] = chunk_map[chunk_dim_index]
            chunk_dim_index += 1
            self.field_chunksize[self.dimensions['lon']] = chunk_map[chunk_dim_index]
        elif len(chunk_map) >= 4:
            self.field_chunksize[self.dimensions['time']] = chunk_map[0]
            self.field_chunksize[self.dimensions['depth']] = chunk_map[1]
            self.field_chunksize[self.dimensions['lat']] = chunk_map[2]
            self.field_chunksize[self.dimensions['lon']] = chunk_map[3]
            dim_index = 4
            for dim_name in self.dimensions:
                if dim_name not in ['time', 'depth', 'lat', 'lon']:
                    self.field_chunksize[self.dimensions[dim_name]] = chunk_map[dim_index]
                    dim_index += 1

    @property
    def data(self):
        return self.data_access()

    def data_access(self):
        if self.chunk_mapping is None and self.field_chunksize not in ['auto', False, None]:
            self.chunk_mapping = {}
            if(isinstance(self.field_chunksize, tuple)):
                j = 0
                for i in range(len(self.field_chunksize)):
                    if self.field_chunksize[i] <= 1:
                        continue
                    self.chunk_mapping[j] = self.field_chunksize[i]
                    j += 1
                self.field_chunksize = tuple([self.chunk_mapping[i] for i in range(len(self.chunk_mapping))])
            else:
                self._chunksize_to_chunkmap()
        data = self.dataset[self.name]

        ti = range(data.shape[0]) if self.ti is None else self.ti
        data = self._apply_indices(data, ti)
        if isinstance(data, xr.DataArray):
            data = data.data

        if isinstance(data, da.core.Array):
            if not self.chunking_finalized:
                if self.field_chunksize == 'auto':
                    if data.shape[-2:] != data.chunksize[-2:]:
                        data = data.rechunk(self.field_chunksize)
                    self.chunk_mapping = {}
                    chunkIndex = 0
                    startblock = 0
                    for chunkDim in data.chunksize[startblock:]:
                        self.chunk_mapping[chunkIndex] = chunkDim
                        chunkIndex += 1
                    self._chunkmap_to_chunksize()
                    if self.rechunk_callback_fields is not None:
                        self.rechunk_callback_fields()
                        self.chunking_finalized = True
                else:
                    # ==== I think this can be "pass" too ==== #
                    data = data.rechunk(self.chunk_mapping)
                    self.chunking_finalized = True
        else:
            da_data = da.from_array(data, chunks=self.field_chunksize)
            if self.field_chunksize == 'auto' and da_data.shape[-2:] == da_data.chunksize[-2:]:
                data = np.array(data)
            else:
                data = da_data
            if not self.chunking_finalized and self.rechunk_callback_fields is not None:
                self.rechunk_callback_fields()
                self.chunking_finalized = True

        return data


class DeferredDaskFileBuffer(DaskFileBuffer):
    def __init__(self, *args, **kwargs):
        super(DeferredDaskFileBuffer, self).__init__(*args, **kwargs)

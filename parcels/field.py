from scipy.interpolate import RectBivariateSpline
from cachetools import cachedmethod, LRUCache
from collections import Iterable
from py import path
import numpy as np
from xray import DataArray, Dataset
import operator
from ctypes import Structure, c_int, c_float, c_double, POINTER


__all__ = ['Field']


class Field(object):
    """Class that encapsulates access to field data.

    :param name: Name of the field
    :param data: 2D array of field data
    :param lon: Longitude coordinates of the field
    :param lat: Latitude coordinates of the field
    :param transpose: Transpose data to required (lon, lat) layout
    """

    def __init__(self, name, data, lon, lat, depth=None, time=None,
                 transpose=False, vmin=None, vmax=None):
        self.name = name
        self.data = data
        self.lon = lon
        self.lat = lat
        self.depth = np.zeros(1, dtype=np.float32) if depth is None else depth
        self.time = np.zeros(1, dtype=np.float64) if time is None else time

        # Ensure that field data is the right data type
        if not self.data.dtype == np.float32:
            print("WARNING: Casting field data to np.float32")
            self.data = self.data.astype(np.float32)
        if transpose:
            # Make a copy of the transposed array to enforce
            # C-contiguous memory layout for JIT mode.
            self.data = np.transpose(self.data).copy()
        self.data = self.data.reshape((self.time.size, self.lat.size, self.lon.size))

        # Hack around the fact that NaN and ridiculously large values
        # propagate in SciPy's interpolators
        if vmin is not None:
            self.data[self.data < vmin] = 0.
        if vmax is not None:
            self.data[self.data > vmax] = 0.
        self.data[np.isnan(self.data)] = 0.

        # Variable names in JIT code
        self.ccode_data = self.name
        self.ccode_lon = self.name + "_lon"
        self.ccode_lat = self.name + "_lat"

        self.interpolator_cache = LRUCache(maxsize=2)
        self.time_index_cache = LRUCache(maxsize=2)

    @classmethod
    def from_netcdf(cls, name, varname, datasets, dimensions={}, **kwargs):
        """Create field from netCDF file using NEMO conventions

        :param name: Name of the field to create
        :param varname: Variable in the file that holds the data
        :param dataset: Single or multiple netcdf.Dataset object(s)
        containing field data. If multiple datasets are present they
        will be concatenated along the time axis
        """
        if not isinstance(datasets, Iterable):
            datasets = [datasets]
        lon = datasets[0].variables[dimensions["lon"]]
        lat = datasets[0].variables[dimensions["lat"]]
        # lat = datasets[0][dimensions["lat"]][:, 0]
        # Default depth to zeros until we implement 3D grids properly
        depth = np.zeros(1, dtype=np.float32)
        # Concatenate time variable to determine overall dimension
        # across multiple files
        timeslices = [dset[dimensions["time"]][:] for dset in datasets]
        time = np.concatenate(timeslices)

        # Pre-allocate grid data before reading files into buffer
        data = np.empty((time.size, 1, lat.size, lon.size), dtype=np.float32)
        tidx = 0
        for tslice, dset in zip(timeslices, datasets):
            data[tidx:, 0, :, :] = dset[varname][:, 0, :, :]
            tidx += tslice.size
        return cls(name, data, lon, lat, depth=depth, time=time, **kwargs)

    def __getitem__(self, key):
        return self.eval(*key)

    @cachedmethod(operator.attrgetter('interpolator_cache'))
    def interpolator(self, t_idx):
        return RectBivariateSpline(self.lat, self.lon,
                                   self.data[t_idx, :])

    @cachedmethod(operator.attrgetter('time_index_cache'))
    def time_index(self, time):
        time_index = self.time < time
        if time_index.all():
            # If given time > last known grid time, use
            # the last grid frame without interpolation
            return -1
        else:
            return time_index.argmin()

    def eval(self, time, x, y):
        idx = self.time_index(time)
        if idx > 0:
            # Return linearly interpolated field value:
            f0 = self.interpolator(idx-1).ev(y, x)
            f1 = self.interpolator(idx).ev(y, x)
            t0 = self.time[idx-1]
            t1 = self.time[idx]
            return f0 + (f1 - f0) * ((time - t0) / (t1 - t0))
        else:
            return self.interpolator(idx).ev(y, x)

    def ccode_subscript(self, t, x, y):
        ccode = "temporal_interpolation_linear(%s, %s, %s, %s, %s, %s)" \
                % (y, x, "particle->yi", "particle->xi", t, self.name)
        return ccode

    @property
    def ctypes_struct(self):
        """Returns a ctypes struct object containing all relevnt
        pointers and sizes for this field."""

        # Ctypes struct corresponding to the type definition in parcels.h
        class CField(Structure):
            _fields_ = [('xdim', c_int), ('ydim', c_int),
                        ('tdim', c_int), ('tidx', c_int),
                        ('lon', POINTER(c_float)), ('lat', POINTER(c_float)),
                        ('time', POINTER(c_double)),
                        ('data', POINTER(POINTER(c_float)))]

        # Create and populate the c-struct object
        cstruct = CField(self.lat.size, self.lon.size, self.time.size, 0,
                         self.lat.ctypes.data_as(POINTER(c_float)),
                         self.lon.ctypes.data_as(POINTER(c_float)),
                         self.time.ctypes.data_as(POINTER(c_double)),
                         self.data.ctypes.data_as(POINTER(POINTER(c_float))))
        return cstruct

    def show(self, **kwargs):
        import matplotlib.pyplot as plt
        t = kwargs.get('t', 0)
        data = np.squeeze(self.data[t, :])
        vmin = kwargs.get('vmin', data.min())
        vmax = kwargs.get('vmax', data.max())
        cs = plt.contourf(self.lon, self.lat, data,
                          levels=np.linspace(vmin, vmax, 256))
        cs.cmap.set_over('k')
        cs.cmap.set_under('w')
        cs.set_clim(vmin, vmax)
        plt.colorbar(cs)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(self.name)
        plt.show()

    def write(self, filename, varname=None):
        filepath = str(path.local('%s%s.nc' % (filename, self.name)))
        if varname is None:
            varname = self.name
        # Derive name of 'depth' variable for NEMO convention
        vname_depth = 'depth'

        # Create DataArray objects for file I/O
        t, d, x, y = (self.time.size, self.depth.size,
                      self.lon.size, self.lat.size)
        nav_lon = DataArray(self.lon + np.zeros((y, x), dtype=np.float32),
                            coords=[('y', self.lat), ('x', self.lon)])
        nav_lat = DataArray(self.lat.reshape(y, 1) + np.zeros(x, dtype=np.float32),
                            coords=[('y', self.lat), ('x', self.lon)])
        vardata = DataArray(self.data.reshape((t, d, y, x)),
                            coords=[('time_counter', self.time),
                                    (vname_depth, self.depth),
                                    ('y', self.lat), ('x', self.lon)])
        # Create xray Dataset and output to netCDF format
        dset = Dataset({varname: vardata}, coords={'nav_lon': nav_lon,
                                                   'nav_lat': nav_lat})
        dset.to_netcdf(filepath)

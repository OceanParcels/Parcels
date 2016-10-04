from scipy.interpolate import RegularGridInterpolator
from cachetools import cachedmethod, LRUCache
from collections import Iterable
from py import path
import numpy as np
import xray
import operator
from ctypes import Structure, c_int, c_float, c_double, POINTER
from netCDF4 import Dataset, num2date
from math import cos, pi
from datetime import timedelta
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib import rc
except:
    plt = None


__all__ = ['CentralDifferences', 'Field', 'Geographic', 'GeographicPolar']


class FieldSamplingError(RuntimeError):
    """Utility error class to propagate erroneous field sampling"""

    def __init__(self, x, y, field=None):
        self.field = field
        self.x = x
        self.y = y
        message = "%s sampled at (%f, %f)" % (
            field.name if field else "Grid", self.x, self.y
        )
        super(FieldSamplingError, self).__init__(message)


def CentralDifferences(field_data, lat, lon):
    r = 6.371e6  # radius of the earth
    deg2rd = np.pi / 180
    dy = r * np.diff(lat) * deg2rd
    # calculate the width of each cell, dependent on lon spacing and latitude
    dx = np.zeros([len(lon)-1, len(lat)], dtype=np.float32)
    for x in range(len(lon))[1:]:
        for y in range(len(lat)):
            dx[x-1, y] = r * np.cos(lat[y] * deg2rd) * (lon[x]-lon[x-1]) * deg2rd
    # calculate central differences for non-edge cells (with equal weighting)
    dVdx = np.zeros(shape=np.shape(field_data), dtype=np.float32)
    dVdy = np.zeros(shape=np.shape(field_data), dtype=np.float32)
    for x in range(len(lon))[1:-1]:
        for y in range(len(lat)):
            dVdx[x, y] = (field_data[x+1, y] - field_data[x-1, y]) / (2 * dx[x-1, y])
    for x in range(len(lon)):
        for y in range(len(lat))[1:-1]:
            dVdy[x, y] = (field_data[x, y+1] - field_data[x, y-1]) / (2 * dy[y-1])
    # Forward and backward difference for edges
    for x in range(len(lon)):
        dVdy[x, 0] = (field_data[x, 1] - field_data[x, 0]) / dy[0]
        dVdy[x, len(lat)-1] = (field_data[x, len(lat)-1] - field_data[x, len(lat)-2]) / dy[len(lat)-2]
    for y in range(len(lat)):
        dVdx[0, y] = (field_data[1, y] - field_data[0, y]) / dx[0, y]
        dVdx[len(lon)-1, y] = (field_data[len(lon)-1, y] - field_data[len(lon)-2, y]) / dx[len(lon)-2, y]

    return [dVdx, dVdy]


class UnitConverter(object):
    """ Interface class for spatial unit conversion during field sampling
        that performs no conversion.
    """
    source_unit = None
    target_unit = None

    def to_target(self, value, x, y):
        return value

    def ccode_to_target(self, x, y):
        return "1.0"

    def to_source(self, value, x, y):
        return value

    def ccode_to_source(self, x, y):
        return "1.0"


class Geographic(UnitConverter):
    """ Unit converter from geometric to geographic coordinates (m to degree) """
    source_unit = 'm'
    target_unit = 'degree'

    def to_target(self, value, x, y):
        return value / 1000. / 1.852 / 60.

    def ccode_to_target(self, x, y):
        return "(1.0 / (1000.0 * 1.852 * 60.0))"


class GeographicPolar(UnitConverter):
    """ Unit converter from geometric to geographic coordinates (m to degree)
        with a correction to account for narrower grid cells closer to the poles.
    """
    source_unit = 'm'
    target_unit = 'degree'

    def to_target(self, value, x, y):
        return value / 1000. / 1.852 / 60. / cos(y * pi / 180)

    def ccode_to_target(self, x, y):
        return "(1.0 / (1000. * 1.852 * 60. * cos(%s * M_PI / 180)))" % y


class Field(object):
    """Class that encapsulates access to field data.

    :param name: Name of the field
    :param data: 2D array of field data
    :param lon: Longitude coordinates of the field
    :param lat: Latitude coordinates of the field
    :param transpose: Transpose data to required (lon, lat) layout
    """

    def __init__(self, name, data, lon, lat, depth=None, time=None,
                 transpose=False, vmin=None, vmax=None, time_origin=0, units=None):
        self.name = name
        self.data = data
        self.lon = lon
        self.lat = lat
        self.depth = np.zeros(1, dtype=np.float32) if depth is None else depth
        self.time = np.zeros(1, dtype=np.float64) if time is None else time
        self.time_origin = time_origin
        self.units = units if units is not None else UnitConverter()

        # Ensure that field data is the right data type
        if not self.data.dtype == np.float32:
            print("WARNING: Casting field data to np.float32")
            self.data = self.data.astype(np.float32)
        if not self.lon.dtype == np.float32:
            print("WARNING: Casting lon data to np.float32")
            self.lon = self.lon.astype(np.float32)
        if not self.lat.dtype == np.float32:
            print("WARNING: Casting lat data to np.float32")
            self.lat = self.lat.astype(np.float32)
        if not self.time.dtype == np.float64:
            print("WARNING: Casting time data to np.float64")
            self.time = self.time.astype(np.float64)
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
    def from_netcdf(cls, name, dimensions, filenames, **kwargs):
        """Create field from netCDF file using NEMO conventions

        :param name: Name of the field to create
        :param dimensions: Variable names for the relevant dimensions
        :param dataset: Single or multiple netcdf.Dataset object(s)
        containing field data. If multiple datasets are present they
        will be concatenated along the time axis
        """
        if not isinstance(filenames, Iterable):
            filenames = [filenames]
        with FileBuffer(filenames[0], dimensions) as filebuffer:
            lon = filebuffer.lon
            lat = filebuffer.lat
            # Assign time_units if the time dimension has units and calendar
            time_units = filebuffer.time_units
            calendar = filebuffer.calendar
        # Default depth to zeros until we implement 3D grids properly
        depth = np.zeros(1, dtype=np.float32)
        # Concatenate time variable to determine overall dimension
        # across multiple files
        timeslices = []
        for fname in filenames:
            with FileBuffer(fname, dimensions) as filebuffer:
                timeslices.append(filebuffer.time)
        timeslices = np.array(timeslices)
        time = np.concatenate(timeslices)
        if time_units is None:
            time_origin = 0
        else:
            time_origin = num2date(0, time_units, calendar)

        # Pre-allocate grid data before reading files into buffer
        data = np.empty((time.size, 1, lat.size, lon.size), dtype=np.float32)
        tidx = 0
        for tslice, fname in zip(timeslices, filenames):
            with FileBuffer(fname, dimensions) as filebuffer:
                data[tidx:, 0, :, :] = filebuffer.data[:, :, :]
            tidx += tslice.size
        return cls(name, data, lon, lat, depth=depth, time=time,
                   time_origin=time_origin, **kwargs)

    def __getitem__(self, key):
        return self.eval(*key)

    def gradient(self, timerange=None, lonrange=None, latrange=None, name=None):
        if name is None:
            name = 'd' + self.name

        if timerange is None:
            time_i = range(len(self.time))
            time = self.time
        else:
            time_i = range(np.where(self.time >= timerange[0])[0][0], np.where(self.time <= timerange[1])[0][-1]+1)
            time = self.time[time_i]
        if lonrange is None:
            lon_i = range(len(self.lon))
            lon = self.lon
        else:
            lon_i = range(np.where(self.lon >= lonrange[0])[0][0], np.where(self.lon <= lonrange[1])[0][-1]+1)
            lon = self.lon[lon_i]
        if latrange is None:
            lat_i = range(len(self.lat))
            lat = self.lat
        else:
            lat_i = range(np.where(self.lat >= latrange[0])[0][0], np.where(self.lat <= latrange[1])[0][-1]+1)
            lat = self.lat[lat_i]

        dVdx = np.zeros(shape=(time.size, lat.size, lon.size), dtype=np.float32)
        dVdy = np.zeros(shape=(time.size, lat.size, lon.size), dtype=np.float32)
        for t in np.nditer(np.int32(time_i)):
            grad = CentralDifferences(np.transpose(self.data[t, :, :][np.ix_(lat_i, lon_i)]), lat, lon)
            dVdx[t, :, :] = np.array(np.transpose(grad[0]))
            dVdy[t, :, :] = np.array(np.transpose(grad[1]))

        return([Field(name + '_dx', dVdx, lon, lat, self.depth, time),
                Field(name + '_dy', dVdy, lon, lat, self.depth, time)])

    @cachedmethod(operator.attrgetter('interpolator_cache'))
    def interpolator2D(self, t_idx):
        """Provide a cached SciPy interpolator for spatial interpolation

        Note that the interpolator is configured to return NaN for
        out-of-bounds coordinates.
        """
        return RegularGridInterpolator((self.lat, self.lon), self.data[t_idx, :],
                                       bounds_error=False, fill_value=np.nan)

    def temporal_interpolate_fullfield(self, tidx, time):
        t0 = self.time[tidx-1]
        t1 = self.time[tidx]
        f0 = self.data[tidx-1, :]
        f1 = self.data[tidx, :]
        return f0 + (f1 - f0) * ((time - t0) / (t1 - t0))

    def spatial_interpolation(self, tidx, y, x):
        """Interpolate horizontal field values using a SciPy interpolator"""
        val = self.interpolator2D(tidx)((y, x))
        if np.isnan(val):
            # Detect Out-of-bounds sampling and raise exception
            raise FieldSamplingError(x, y, field=self)
        else:
            return val

    @cachedmethod(operator.attrgetter('time_index_cache'))
    def time_index(self, time):
        """Find the next index in the time array for a given time

        Note that we normalize to either the first or the last index
        if the sampled value is outside the time value range.
        """
        time_index = self.time < time
        if time_index.all():
            # If given time > last known grid time, use
            # the last grid frame without interpolation
            return len(self.time) - 1
        else:
            return time_index.argmin()

    def eval(self, time, x, y):
        """Interpolate field values in space and time.

        We interpolate linearly in time and apply implicit unit
        conversion to the result. Note that we defer to
        scipy.interpolate to perform spatial interpolation.
        """
        t_idx = self.time_index(time)
        if 0 < t_idx < len(self.time) - 1 and self.time[t_idx] != time:
            f0 = self.spatial_interpolation(t_idx - 1, y, x)
            f1 = self.spatial_interpolation(t_idx, y, x)
            t0 = self.time[t_idx-1]
            t1 = self.time[t_idx]
            value = f0 + (f1 - f0) * ((time - t0) / (t1 - t0))
        else:
            # Skip temporal interpolation if time is outside
            # of the defined time range or if we have hit an
            # excat value in the time array.
            value = self.spatial_interpolation(t_idx, y, x)

        return self.units.to_target(value, x, y)

    def ccode_subscript(self, t, x, y):
        ccode = "%s * temporal_interpolation_linear(%s, %s, %s, %s, %s, %s)" \
                % (self.units.ccode_to_target(x, y),
                   x, y, "particle->xi", "particle->yi", t, self.name)
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
        cstruct = CField(self.lon.size, self.lat.size, self.time.size, 0,
                         self.lon.ctypes.data_as(POINTER(c_float)),
                         self.lat.ctypes.data_as(POINTER(c_float)),
                         self.time.ctypes.data_as(POINTER(c_double)),
                         self.data.ctypes.data_as(POINTER(POINTER(c_float))))
        return cstruct

    def show(self, **kwargs):
        if plt is None:
            raise RuntimeError("Visualisation not possible: matplotlib not found!")

        t = kwargs.get('t', 0)
        with_particles = kwargs.get('with_particles', False)
        make_animation = kwargs.get('animation', False)
        if with_particles or (not make_animation):
            idx = self.time_index(t)
            if self.time.size > 1:
                data = np.squeeze(self.temporal_interpolate_fullfield(idx, t))
            else:
                data = np.squeeze(self.data)

            vmin = kwargs.get('vmin', data.min())
            vmax = kwargs.get('vmax', data.max())
            cs = plt.contourf(self.lon, self.lat, data,
                              levels=np.linspace(vmin, vmax, 256))
            cs.cmap.set_over('k')
            cs.cmap.set_under('w')
            cs.set_clim(vmin, vmax)
            plt.colorbar(cs)
            if not with_particles:
                plt.show()
        else:
            fig = plt.figure()
            ax = plt.axes(xlim=(self.lon[0], self.lon[-1]), ylim=(self.lat[0], self.lat[-1]))

            def animate(i):
                data = np.squeeze(self.data[i, :, :])
                vmin = kwargs.get('vmin', data.min())
                vmax = kwargs.get('vmax', data.max())
                cont = ax.contourf(self.lon, self.lat, data,
                                   levels=np.linspace(vmin, vmax, 256))
                return cont

            rc('animation', html='html5')
            anim = animation.FuncAnimation(fig, animate, frames=np.arange(1, self.data.shape[0]),
                                           interval=100, blit=False)
            plt.close()
            return anim

    def write(self, filename, varname=None):
        filepath = str(path.local('%s%s.nc' % (filename, self.name)))
        if varname is None:
            varname = self.name
        # Derive name of 'depth' variable for NEMO convention
        vname_depth = 'depth%s' % self.name.lower()

        # Create DataArray objects for file I/O
        t, d, x, y = (self.time.size, self.depth.size,
                      self.lon.size, self.lat.size)
        nav_lon = xray.DataArray(self.lon + np.zeros((y, x), dtype=np.float32),
                                 coords=[('y', self.lat), ('x', self.lon)])
        nav_lat = xray.DataArray(self.lat.reshape(y, 1) + np.zeros(x, dtype=np.float32),
                                 coords=[('y', self.lat), ('x', self.lon)])
        vardata = xray.DataArray(self.data.reshape((t, d, y, x)),
                                 coords=[('time_counter', self.time),
                                         (vname_depth, self.depth),
                                         ('y', self.lat), ('x', self.lon)])
        # Create xray Dataset and output to netCDF format
        dset = xray.Dataset({varname: vardata}, coords={'nav_lon': nav_lon,
                                                        'nav_lat': nav_lat})
        dset.to_netcdf(filepath)


class FileBuffer(object):
    """ Class that encapsulates and manages deferred access to file data. """

    def __init__(self, filename, dimensions):
        self.filename = filename
        self.dimensions = dimensions  # Dict with dimension keyes for file data
        self.dataset = None

    def __enter__(self):
        self.dataset = Dataset(str(self.filename), 'r', format="NETCDF4")
        return self

    def __exit__(self, type, value, traceback):
        self.dataset.close()

    @property
    def lon(self):
        lon = self.dataset[self.dimensions['lon']]
        return lon[0, :] if len(lon.shape) > 1 else lon[:]

    @property
    def lat(self):
        lat = self.dataset[self.dimensions['lat']]
        return lat[:, 0] if len(lat.shape) > 1 else lat[:]

    @property
    def data(self):
        if len(self.dataset[self.dimensions['data']].shape) == 3:
            return self.dataset[self.dimensions['data']][:, :, :]
        else:
            return self.dataset[self.dimensions['data']][:, 0, :, :]

    @property
    def time(self):
        if self.time_units is not None:
            dt = num2date(self.dataset[self.dimensions['time']][:],
                          self.time_units, self.calendar)
            dt -= num2date(0, self.time_units, self.calendar)
            return list(map(timedelta.total_seconds, dt))
        else:
            return self.dataset[self.dimensions['time']][:]

    @property
    def time_units(self):
        """ Derive time_units if the time dimension has units """
        try:
            return self.dataset[self.dimensions['time']].units
        except:
            try:
                return self.dataset[self.dimensions['time']].Unit
            except:
                return None

    @property
    def calendar(self):
        """ Derive calendar if the time dimension has calendar """
        try:
            return self.dataset[self.dimensions['time']].calendar
        except:
            return 'standard'

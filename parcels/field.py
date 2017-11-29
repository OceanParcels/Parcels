from parcels.loggers import logger
from scipy.interpolate import RegularGridInterpolator
from collections import Iterable
from py import path
import numpy as np
import xarray
from ctypes import Structure, c_int, c_float, POINTER, pointer
from netCDF4 import Dataset, num2date
from math import cos, pi
from datetime import timedelta, datetime
from dateutil.parser import parse
import math
from grid import RectilinearZGrid, CGrid, GridCode


__all__ = ['Field', 'Geographic', 'GeographicPolar']


class FieldSamplingError(RuntimeError):
    """Utility error class to propagate erroneous field sampling"""

    def __init__(self, x, y, z, field=None):
        self.field = field
        self.x = x
        self.y = y
        self.z = z
        message = "%s sampled at (%f, %f, %f)" % (
            field.name if field else "Field", self.x, self.y, self.z
        )
        super(FieldSamplingError, self).__init__(message)


class TimeExtrapolationError(RuntimeError):
    """Utility error class to propagate erroneous time extrapolation sampling"""

    def __init__(self, time, field=None):
        if field is not None and field.grid.time_origin != 0:
            time = field.grid.time_origin + timedelta(seconds=time)
        message = "%s sampled outside time domain at time %s." % (
            field.name if field else "Field", time)
        message += " Try setting allow_time_extrapolation to True"
        super(TimeExtrapolationError, self).__init__(message)


class UnitConverter(object):
    """ Interface class for spatial unit conversion during field sampling
        that performs no conversion.
    """
    source_unit = None
    target_unit = None

    def to_target(self, value, x, y, z):
        return value

    def ccode_to_target(self, x, y, z):
        return "1.0"

    def to_source(self, value, x, y, z):
        return value

    def ccode_to_source(self, x, y, z):
        return "1.0"


class Geographic(UnitConverter):
    """ Unit converter from geometric to geographic coordinates (m to degree) """
    source_unit = 'm'
    target_unit = 'degree'

    def to_target(self, value, x, y, z):
        return value / 1000. / 1.852 / 60.

    def to_source(self, value, x, y, z):
        return value * 1000. * 1.852 * 60.

    def ccode_to_target(self, x, y, z):
        return "(1.0 / (1000.0 * 1.852 * 60.0))"

    def ccode_to_source(self, x, y, z):
        return "(1000.0 * 1.852 * 60.0)"


class GeographicPolar(UnitConverter):
    """ Unit converter from geometric to geographic coordinates (m to degree)
        with a correction to account for narrower grid cells closer to the poles.
    """
    source_unit = 'm'
    target_unit = 'degree'

    def to_target(self, value, x, y, z):
        return value / 1000. / 1.852 / 60. / cos(y * pi / 180)

    def to_source(self, value, x, y, z):
        return value * 1000. * 1.852 * 60. * cos(y * pi / 180)

    def ccode_to_target(self, x, y, z):
        return "(1.0 / (1000. * 1.852 * 60. * cos(%s * M_PI / 180)))" % y

    def ccode_to_source(self, x, y, z):
        return "(1000. * 1.852 * 60. * cos(%s * M_PI / 180))" % y


class Field(object):
    """Class that encapsulates access to field data.

    :param name: Name of the field
    :param data: 2D, 3D or 4D array of field data
    :param lon: Longitude coordinates of the field. (only if grid is None)
    :param lat: Latitude coordinates of the field. (only if grid is None)
    :param depth: Depth coordinates of the field. (only if grid is None)
    :param time: Time coordinates of the field. (only if grid is None)
    :param mesh: Type of mesh coordinates of the field. (only if grid is None)
    :param grid: :class:`parcels.grid.Grid` object containing all the lon, lat depth, time
           mesh and time_origin information
    :param transpose: Transpose data to required (lon, lat) layout
    :param vmin: Minimum allowed value on the field.
           Data below this value are set to zero
    :param vmax: Maximum allowed value on the field
           Data above this value are set to zero
    :param time_origin: Time origin of the time axis (only if grid is None)
    :param interp_method: Method for interpolation
    :param allow_time_extrapolation: boolean whether to allow for extrapolation
    :param time_periodic: boolean whether to loop periodically over the time component of the Field
           This flag overrides the allow_time_interpolation and sets it to False
    """

    def __init__(self, name, data, lon=None, lat=None, depth=None, time=None, mesh='flat',
                 grid=None, transpose=False, vmin=None, vmax=None, time_origin=0,
                 interp_method='linear', allow_time_extrapolation=None, time_periodic=False):
        self.name = name
        self.data = data
        if grid:
            self.grid = grid
        else:
            self.grid = RectilinearZGrid('auto_gen_grid', lon, lat, depth, time, time_origin=time_origin, mesh=mesh)
        # self.lon, self.lat, self.depth and self.time are not used anymore in parcels.
        # self.grid should be used instead.
        # Those variables are still defined for backwards compatibility with users codes.
        self.lon = self.grid.lon
        self.lat = self.grid.lat
        self.depth = self.grid.depth
        self.time = self.grid.time
        if self.grid.mesh is 'flat' or (name is not 'U' and name is not 'V'):
            self.units = UnitConverter()
        elif self.grid.mesh is 'spherical' and name == 'U':
            self.units = GeographicPolar()
        elif self.grid.mesh is 'spherical' and name == 'V':
            self.units = Geographic()
        else:
            raise ValueError("Unsupported mesh type. Choose either: 'spherical' or 'flat'")
        self.interp_method = interp_method
        self.fieldset = None
        if allow_time_extrapolation is None:
            self.allow_time_extrapolation = True if time is None else False
        else:
            self.allow_time_extrapolation = allow_time_extrapolation

        self.time_periodic = time_periodic
        if self.time_periodic and self.allow_time_extrapolation:
            logger.warning_once("allow_time_extrapolation and time_periodic cannot be used together.\n \
                                 allow_time_extrapolation is set to False")
            self.allow_time_extrapolation = False

        # Ensure that field data is the right data type
        if not self.data.dtype == np.float32:
            logger.warning_once("Casting field data to np.float32")
            self.data = self.data.astype(np.float32)
        if transpose:
            # Make a copy of the transposed array to enforce
            # C-contiguous memory layout for JIT mode.
            self.data = np.transpose(self.data).copy()
        if self.grid.depth.size > 1 and len(self.grid.depth.shape) == 1:
            self.data = self.data.reshape((self.grid.time.size, self.grid.depth.size, self.grid.lat.size, self.grid.lon.size))
        elif len(self.grid.depth.shape) in [3, 4]:
            self.data = self.data.reshape((self.grid.time.size, self.grid.depth.shape[2], self.grid.lat.size, self.grid.lon.size))
        else:
            self.data = self.data.reshape((self.grid.time.size, self.grid.lat.size, self.grid.lon.size))

        # Hack around the fact that NaN and ridiculously large values
        # propagate in SciPy's interpolators
        if vmin is not None:
            self.data[self.data < vmin] = 0.
        if vmax is not None:
            self.data[self.data > vmax] = 0.
        self.data[np.isnan(self.data)] = 0.

        # Variable names in JIT code
        self.ccode_data = self.name

    @classmethod
    def from_netcdf(cls, name, dimensions, filenames, indices={},
                    allow_time_extrapolation=False, mesh='flat', **kwargs):
        """Create field from netCDF file

        :param name: Name of the field to create
        :param dimensions: Variable names for the relevant dimensions
        :param filenames: Filenames of the field
        :param indices: indices for each dimension to read from file
        :param allow_time_extrapolation: boolean whether to allow for extrapolation
        :param mesh: String indicating the type of mesh coordinates and
               units used during velocity interpolation:

               1. spherical (default): Lat and lon in degree, with a
                  correction for zonal velocity U near the poles.
               2. flat: No conversion, lat/lon are assumed to be in m.
        """

        if not isinstance(filenames, Iterable) or isinstance(filenames, str):
            filenames = [filenames]
        with FileBuffer(filenames[0], dimensions) as filebuffer:
            lon, indslon = filebuffer.read_dimension('lon', indices)
            lat, indslat = filebuffer.read_dimension('lat', indices)
            depth, indsdepth = filebuffer.read_dimension('depth', indices)
            # Assign time_units if the time dimension has units and calendar
            time_units = filebuffer.time_units
            calendar = filebuffer.calendar
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
            if type(time_origin) is not datetime:
                # num2date in some cases returns a 'phony' datetime. In that case,
                # parse it as a string.
                # See http://unidata.github.io/netcdf4-python/#netCDF4.num2date
                time_origin = parse(str(time_origin))

        # Pre-allocate data before reading files into buffer
        data = np.empty((time.size, depth.size, lat.size, lon.size), dtype=np.float32)
        tidx = 0
        for tslice, fname in zip(timeslices, filenames):
            with FileBuffer(fname, dimensions) as filebuffer:
                filebuffer.indslat = indslat
                filebuffer.indslon = indslon
                filebuffer.indsdepth = indsdepth
                if 'data' in dimensions:
                    # If Field.from_netcdf is called directly, it may not have a 'data' dimension
                    # In that case, assume that 'name' is the data dimension
                    filebuffer.name = dimensions['data']
                else:
                    filebuffer.name = name

                if len(filebuffer.dataset[filebuffer.name].shape) is 3:
                    data[tidx:tidx+len(tslice), 0, :, :] = filebuffer.data[:, :, :]
                else:
                    data[tidx:tidx+len(tslice), :, :, :] = filebuffer.data[:, :, :, :]
            tidx += len(tslice)
        # Time indexing after the fact only
        if 'time' in indices:
            time = time[indices['time']]
            data = data[indices['time'], :, :, :]
        grid = RectilinearZGrid('auto_gen_grid', lon, lat, depth, time, time_origin=time_origin, mesh=mesh)
        return cls(name, data, grid=grid,
                   allow_time_extrapolation=allow_time_extrapolation, **kwargs)

    def __getitem__(self, key):
        return self.eval(*key)

    def cell_edge_sizes(self):
        """Method to calculate cell sizes based on numpy.gradient method
                Currently only works for Rectilinear Grids"""
        dy_grid = np.zeros((self.grid.lon.size, self.grid.lat.size), dtype=np.float32)
        dx_grid = np.zeros((self.grid.lon.size, self.grid.lat.size), dtype=np.float32)

        x_conv = GeographicPolar() if self.grid.mesh is 'spherical' else UnitConverter()
        y_conv = Geographic() if self.grid.mesh is 'spherical' else UnitConverter()
        for y, (lat, dy) in enumerate(zip(self.grid.lat, np.gradient(self.grid.lat))):
            for x, (lon, dx) in enumerate(zip(self.grid.lon, np.gradient(self.grid.lon))):
                dx_grid[x, y] = x_conv.to_source(dx, lon, lat, self.grid.depth[0])
                dy_grid[x, y] = y_conv.to_source(dy, lon, lat, self.grid.depth[0])
        return dx_grid, dy_grid

    def cell_areas(self):
        """Method to calculate cell sizes based on cell_edge_sizes
                Currently only works for Rectilinear Grids"""
        dx_mesh, dy_mesh = self.cell_edge_sizes()
        return dx_mesh * dy_mesh

    def gradient(self, timerange=None, name=None):
        """Method to create gradients of Field"""
        if name is None:
            name = 'd' + self.name

        if timerange is None:
            time_i = range(len(self.grid.time))
            time = self.grid.time
        else:
            time_i = range(np.where(self.grid.time >= timerange[0])[0][0], np.where(self.grid.time <= timerange[1])[0][-1]+1)
            time = self.grid.time[time_i]

        dFdx = np.zeros_like(self.data)
        dFdy = np.zeros_like(self.data)
        celldist_lon, celldist_lat = self.cell_edge_sizes()
        for t in np.nditer(np.int32(time_i)):
            dFdy[t, :, :] = np.gradient(self.data[t, :, :], axis=0) / np.transpose(celldist_lat)
            dFdx[t, :, :] = np.gradient(self.data[t, :, :], axis=1) / np.transpose(celldist_lon)
        return([Field(name + '_dx', dFdx, lon=self.grid.lon, lat=self.grid.lat, depth=self.grid.depth, time=time,
                      interp_method=self.interp_method, allow_time_extrapolation=self.allow_time_extrapolation),
                Field(name + '_dy', dFdy, lon=self.grid.lon, lat=self.grid.lat, depth=self.grid.depth, time=time,
                      interp_method=self.interp_method, allow_time_extrapolation=self.allow_time_extrapolation)])

    def interpolator3D_rectilinear_z(self, idx, z, y, x):
        """Scipy implementation of 3D interpolation, by first interpolating
        in horizontal, then in the vertical"""

        zdx = self.depth_index(z, y, x)
        f0 = self.interpolator2D(idx, z_idx=zdx)((y, x))
        f1 = self.interpolator2D(idx, z_idx=zdx + 1)((y, x))
        z0 = self.grid.depth[zdx]
        z1 = self.grid.depth[zdx + 1]
        if z < z0 or z > z1:
            raise FieldSamplingError(x, y, z, field=self)
        if self.interp_method is 'nearest':
            return f0 if z - z0 < z1 - z else f1
        elif self.interp_method is 'linear':
            return f0 + (f1 - f0) * ((z - z0) / (z1 - z0))
        else:
            raise RuntimeError(self.interp_method+"is not implemented for 3D grids")

    def interpolator3D_rectilinear_s(self, idx, z, y, x, time):

        grid = self.grid

        if x < grid.lon[0] or x > grid.lon[-1]:
            raise FieldSamplingError(x, y, z, field=self)
        if y < grid.lat[0] or y > grid.lat[-1]:
            raise FieldSamplingError(x, y, z, field=self)

        lon_index = grid.lon <= x
        xi = yi = zi = -1
        if lon_index.all():
            xi = len(grid.lon) - 2
        else:
            xi = lon_index.argmin() - 1 if lon_index.any() else 0
        lat_index = grid.lat <= y
        if lat_index.all():
            yi = len(grid.lat) - 2
        else:
            yi = lat_index.argmin() - 1 if lat_index.any() else 0

        xsi = (x-grid.lon[xi]) / (grid.lon[xi+1]-grid.lon[xi])
        eta = (y-grid.lat[yi]) / (grid.lat[yi+1]-grid.lat[yi])
        assert xsi >= 0 and xsi <= 1
        assert eta >= 0 and eta <= 1

        if grid.z4d:
            if idx == len(self.grid.time)-1:
                depth_vector = (1-xsi)*(1-eta) * grid.depth[xi, yi, :, -1] + \
                    xsi*(1-eta) * grid.depth[xi+1, yi, :, -1] + \
                    xsi*eta * grid.depth[xi+1, yi+1, :, -1] + \
                    (1-xsi)*eta * grid.depth[xi, yi+1, :, -1]
            else:
                dv2 = (1-xsi)*(1-eta) * grid.depth[xi, yi, :, idx:idx+2] + \
                    xsi*(1-eta) * grid.depth[xi+1, yi, :, idx:idx+2] + \
                    xsi*eta * grid.depth[xi+1, yi+1, :, idx:idx+2] + \
                    (1-xsi)*eta * grid.depth[xi, yi+1, :, idx:idx+2]
                t0 = self.grid.time[idx]
                t1 = self.grid.time[idx + 1]
                depth_vector = dv2[:, 0] + (dv2[:, 1]-dv2[:, 0]) * (time - t0) / (t1 - t0)
        else:
            depth_vector = (1-xsi)*(1-eta) * grid.depth[xi, yi, :] + \
                xsi*(1-eta) * grid.depth[xi+1, yi, :] + \
                xsi*eta * grid.depth[xi+1, yi+1, :] + \
                (1-xsi)*eta * grid.depth[xi, yi+1, :]

        # depth variable is defined at np.float32 in particle.py, but as soon as
        # as there is an operation with dt which is type float, it becomes np.float64
        z = np.float32(z)
        depth_index = depth_vector <= z
        if z >= depth_vector[-1]:
            zi = len(depth_vector) - 2
        else:
            zi = depth_index.argmin() - 1 if z >= depth_vector[0] else 0
        z0 = depth_vector[zi]
        z1 = depth_vector[zi+1]
        if z < z0 or z > z1:
            raise FieldSamplingError(x, y, z, field=self)

        if self.interp_method is 'nearest':
            zii = zi if z - z0 < z1 - z else zi+1
            xii = xi if xsi <= .5 else xi+1
            yii = yi if eta <= .5 else yi+1
            return self.data[idx, zii, yii, xii]
        elif self.interp_method is 'linear':
            data = self.data[idx, zi, :, :].transpose()
            f0 = (1-xsi)*(1-eta) * data[xi, yi] + \
                xsi*(1-eta) * data[xi+1, yi] + \
                xsi*eta * data[xi+1, yi+1] + \
                    (1-xsi)*eta * data[xi, yi+1]
            data = self.data[idx, zi+1, :, :].transpose()
            f1 = (1-xsi)*(1-eta) * data[xi, yi] + \
                xsi*(1-eta) * data[xi+1, yi] + \
                xsi*eta * data[xi+1, yi+1] + \
                (1-xsi)*eta * data[xi, yi+1]
            return f0 + (f1 - f0) * ((z - z0) / (z1 - z0))
        else:
            raise RuntimeError(self.interp_method+"is not implemented for 3D grids")

    def interpolator3D(self, idx, z, y, x, time):
        """Scipy implementation of 3D interpolation, by first interpolating
        in horizontal, then in the vertical"""

        if self.grid.gtype == GridCode.RectilinearZGrid:
            return self.interpolator3D_rectilinear_z(idx, z, y, x)
        elif self.grid.gtype == GridCode.RectilinearSGrid:
            return self.interpolator3D_rectilinear_s(idx, z, y, x, time)
        else:
            print("Only RectilinearZGrid and RectilinearSGrid grids are currently implemented")
            exit(-1)

    def interpolator2D(self, t_idx, z_idx=None):
        """Provide a SciPy interpolator for spatial interpolation

        Note that the interpolator is configured to return NaN for
        out-of-bounds coordinates.
        """
        if z_idx is None:
            data = self.data[t_idx, :]
        else:
            data = self.data[t_idx, z_idx, :]
        return RegularGridInterpolator((self.grid.lat, self.grid.lon), data,
                                       bounds_error=False, fill_value=np.nan,
                                       method=self.interp_method)

    def temporal_interpolate_fullfield(self, tidx, time):
        """Calculate the data of a field between two snapshots,
        using linear interpolation

        :param tidx: Index in time array associated with time (via :func:`time_index`)
        :param time: Time to interpolate to

        :rtype: Linearly interpolated field"""
        t0 = self.grid.time[tidx]
        t1 = self.grid.time[tidx+1]
        f0 = self.data[tidx, :]
        f1 = self.data[tidx+1, :]
        return f0 + (f1 - f0) * ((time - t0) / (t1 - t0))

    def spatial_interpolation(self, tidx, z, y, x, time):
        """Interpolate horizontal field values using a SciPy interpolator"""
        if self.grid.depth.size == 1:
            val = self.interpolator2D(tidx)((y, x))
        else:
            val = self.interpolator3D(tidx, z, y, x, time)
        if np.isnan(val):
            # Detect Out-of-bounds sampling and raise exception
            raise FieldSamplingError(x, y, z, field=self)
        else:
            return val

    def time_index(self, time):
        """Find the index in the time array associated with a given time

        Note that we normalize to either the first or the last index
        if the sampled value is outside the time value range.
        """
        if not self.time_periodic and not self.allow_time_extrapolation and (time < self.grid.time[0] or time > self.grid.time[-1]):
            raise TimeExtrapolationError(time, field=self)
        time_index = self.grid.time <= time
        if self.time_periodic:
            if time_index.all() or np.logical_not(time_index).all():
                periods = math.floor((time-self.grid.time[0])/(self.grid.time[-1]-self.grid.time[0]))
                time -= periods*(self.grid.time[-1]-self.grid.time[0])
                time_index = self.grid.time <= time
                ti = time_index.argmin() - 1 if time_index.any() else 0
                return (ti, periods)
            return (time_index.argmin() - 1 if time_index.any() else 0, 0)
        if time_index.all():
            # If given time > last known field time, use
            # the last field frame without interpolation
            return (len(self.grid.time) - 1, 0)
        else:
            return (time_index.argmin() - 1 if time_index.any() else 0, 0)

    def depth_index(self, depth, lat, lon):
        """Find the index in the depth array associated with a given depth"""
        if depth > self.grid.depth[-1]:
            raise FieldSamplingError(lon, lat, depth, field=self)
        depth_index = self.grid.depth <= depth
        if depth_index.all():
            # If given depth == largest field depth, use the second-last
            # field depth (as zidx+1 needed in interpolation)
            return len(self.grid.depth) - 2
        else:
            return depth_index.argmin() - 1 if depth_index.any() else 0

    def eval(self, time, x, y, z):
        """Interpolate field values in space and time.

        We interpolate linearly in time and apply implicit unit
        conversion to the result. Note that we defer to
        scipy.interpolate to perform spatial interpolation.
        """
        (t_idx, periods) = self.time_index(time)
        time -= periods*(self.grid.time[-1]-self.grid.time[0])
        if t_idx < len(self.grid.time)-1 and time > self.grid.time[t_idx]:
            f0 = self.spatial_interpolation(t_idx, z, y, x, time)
            f1 = self.spatial_interpolation(t_idx + 1, z, y, x, time)
            t0 = self.grid.time[t_idx]
            t1 = self.grid.time[t_idx + 1]
            value = f0 + (f1 - f0) * ((time - t0) / (t1 - t0))
        else:
            # Skip temporal interpolation if time is outside
            # of the defined time range or if we have hit an
            # excat value in the time array.
            value = self.spatial_interpolation(t_idx, z, y, x, self.grid.time[t_idx-1])

        return self.units.to_target(value, x, y, z)

    def ccode_eval(self, var, t, x, y, z):
        # Casting interp_methd to int as easier to pass on in C-code
        gridset = self.fieldset.gridset
        iGrid = -1
        for i, g in enumerate(gridset.grids):
            if g.name == self.grid.name:
                iGrid = i
                break
        return "temporal_interpolation_linear(%s, %s, %s, %s, %s, %s, %s, &%s, %s)" \
            % (x, y, z, "particle->CGridIndexSet", iGrid, t, self.name, var,
               self.interp_method.upper())

    def ccode_convert(self, _, x, y, z):
        return self.units.ccode_to_target(x, y, z)

    @property
    def ctypes_struct(self):
        """Returns a ctypes struct object containing all relevant
        pointers and sizes for this field."""

        # Ctypes struct corresponding to the type definition in parcels.h
        class CField(Structure):
            _fields_ = [('xdim', c_int), ('ydim', c_int), ('zdim', c_int),
                        ('tdim', c_int),
                        ('allow_time_extrapolation', c_int),
                        ('time_periodic', c_int),
                        ('data', POINTER(POINTER(c_float))),
                        ('grid', POINTER(CGrid))]

        # Create and populate the c-struct object
        allow_time_extrapolation = 1 if self.allow_time_extrapolation else 0
        time_periodic = 1 if self.time_periodic else 0
        cstruct = CField(self.grid.lon.size, self.grid.lat.size, self.grid.depth.size,
                         self.grid.time.size, allow_time_extrapolation, time_periodic,
                         self.data.ctypes.data_as(POINTER(POINTER(c_float))),
                         pointer(self.grid.ctypes_struct))
        return cstruct

    def show(self, with_particles=False, animation=False, show_time=0, vmin=None, vmax=None):
        """Method to 'show' a :class:`Field` using matplotlib

        :param with_particles: Boolean whether particles are also plotted on Field
        :param animation: Boolean whether result is a single plot, or an animation
        :param show_time: Time at which to show the Field (only in single-plot mode)
        :param vmin: minimum colour scale (only in single-plot mode)
        :param vmax: maximum colour scale (only in single-plot mode)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation_plt
            from matplotlib import rc
        except:
            logger.info("Visualisation is not possible. Matplotlib not found.")
            return

        if with_particles or (not animation):
            (idx, periods) = self.time_index(show_time)
            show_time -= periods*(self.grid.time[-1]-self.grid.time[0])
            if self.grid.time.size > 1:
                data = np.squeeze(self.temporal_interpolate_fullfield(idx, show_time))
            else:
                data = np.squeeze(self.data)

            vmin = data.min() if vmin is None else vmin
            vmax = data.max() if vmax is None else vmax
            cs = plt.contourf(self.grid.lon, self.grid.lat, data,
                              levels=np.linspace(vmin, vmax, 256))
            cs.cmap.set_over('k')
            cs.cmap.set_under('w')
            cs.set_clim(vmin, vmax)
            plt.colorbar(cs)
            if not with_particles:
                plt.show()
        else:
            fig = plt.figure()
            ax = plt.axes(xlim=(self.grid.lon[0], self.grid.lon[-1]), ylim=(self.grid.lat[0], self.grid.lat[-1]))

            def animate(i):
                data = np.squeeze(self.data[i, :, :])
                cont = ax.contourf(self.grid.lon, self.grid.lat, data,
                                   levels=np.linspace(data.min(), data.max(), 256))
                return cont

            rc('animation', html='html5')
            anim = animation_plt.FuncAnimation(fig, animate, frames=np.arange(1, self.data.shape[0]),
                                               interval=100, blit=False)
            plt.close()
            return anim

    def add_periodic_halo(self, zonal, meridional, halosize=5):
        """Add a 'halo' to all Fields in a FieldSet, through extending the Field (and lon/lat)
        by copying a small portion of the field on one side of the domain to the other.

        :param zonal: Create a halo in zonal direction (boolean)
        :param meridional: Create a halo in meridional direction (boolean)
        :param halosize: size of the halo (in grid points). Default is 5 grid points
        """
        if zonal:
            lonshift = (self.grid.lon[-1] - 2 * self.grid.lon[0] + self.grid.lon[1])
            if len(self.data.shape) is 3:
                self.data = np.concatenate((self.data[:, :, -halosize:], self.data,
                                            self.data[:, :, 0:halosize]), axis=len(self.data.shape)-1)
            else:
                self.data = np.concatenate((self.data[:, :, :, -halosize:], self.data,
                                            self.data[:, :, :, 0:halosize]), axis=len(self.data.shape) - 1)
            self.grid.lon = np.concatenate((self.grid.lon[-halosize:] - lonshift,
                                            self.grid.lon, self.grid.lon[0:halosize] + lonshift))
            self.lon = self.grid.lon
        if meridional:
            latshift = (self.grid.lat[-1] - 2 * self.grid.lat[0] + self.grid.lat[1])
            if len(self.data.shape) is 3:
                self.data = np.concatenate((self.data[:, -halosize:, :], self.data,
                                            self.data[:, 0:halosize, :]), axis=len(self.data.shape)-2)
            else:
                self.data = np.concatenate((self.data[:, :, -halosize:, :], self.data,
                                            self.data[:, :, 0:halosize, :]), axis=len(self.data.shape) - 2)
            self.grid.lat = np.concatenate((self.grid.lat[-halosize:] - latshift,
                                            self.grid.lat, self.grid.lat[0:halosize] + latshift))
            self.lat = self.grid.lat

    def write(self, filename, varname=None):
        """Write a :class:`Field` to a netcdf file

        :param filename: Basename of the file
        :param varname: Name of the field, to be appended to the filename"""
        filepath = str(path.local('%s%s.nc' % (filename, self.name)))
        if varname is None:
            varname = self.name
        # Derive name of 'depth' variable for NEMO convention
        vname_depth = 'depth%s' % self.name.lower()

        # Create DataArray objects for file I/O
        t, d, x, y = (self.grid.time.size, self.grid.depth.size,
                      self.grid.lon.size, self.grid.lat.size)
        nav_lon = xarray.DataArray(self.grid.lon + np.zeros((y, x), dtype=np.float32),
                                   coords=[('y', self.grid.lat), ('x', self.grid.lon)])
        nav_lat = xarray.DataArray(self.grid.lat.reshape(y, 1) + np.zeros(x, dtype=np.float32),
                                   coords=[('y', self.grid.lat), ('x', self.grid.lon)])
        vardata = xarray.DataArray(self.data.reshape((t, d, y, x)),
                                   coords=[('time_counter', self.grid.time),
                                           (vname_depth, self.grid.depth),
                                           ('y', self.grid.lat), ('x', self.grid.lon)])
        # Create xarray Dataset and output to netCDF format
        dset = xarray.Dataset({varname: vardata}, coords={'nav_lon': nav_lon,
                                                          'nav_lat': nav_lat,
                                                          vname_depth: self.grid.depth})
        dset.to_netcdf(filepath)

    def advancetime(self, field_new):
        if len(field_new.grid.time) is not 1:
            raise RuntimeError('New FieldSet needs to have only one snapshot')
        if field_new.grid.time > self.grid.time[-1]:  # forward in time, so appending at end
            self.data = np.concatenate((self.data[1:, :, :], field_new.data[:, :, :]), 0)
            self.grid.time = np.concatenate((self.grid.time[1:], field_new.grid.time))
            self.time = self.grid.time
        elif field_new.grid.time < self.grid.time[0]:  # backward in time, so prepending at start
            self.data = np.concatenate((field_new.data[:, :, :], self.data[:-1, :, :]), 0)
            self.grid.time = np.concatenate((field_new.grid.time, self.grid.time[:-1]))
            self.time = self.grid.time
        else:
            raise RuntimeError("Time of field_new in Field.advancetime() overlaps with times in old Field")


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

    def read_dimension(self, dimname, indices):
        dim = getattr(self, dimname)
        inds = indices[dimname] if dimname in indices else range(dim.size)
        if not isinstance(inds, list):
            raise RuntimeError('Index for '+dimname+' needs to be a list')
        return dim[inds], inds

    @property
    def lon(self):
        lon = self.dataset[self.dimensions['lon']]
        return lon[0, :] if len(lon.shape) > 1 else lon[:]

    @property
    def lat(self):
        lat = self.dataset[self.dimensions['lat']]
        return lat[:, 0] if len(lat.shape) > 1 else lat[:]

    @property
    def depth(self):
        if 'depth' in self.dimensions:
            depth = self.dataset[self.dimensions['depth']]
            return depth[:, 0] if len(depth.shape) > 1 else depth[:]
        else:
            return np.zeros(1)

    @property
    def data(self):
        if len(self.dataset[self.name].shape) == 3:
            data = self.dataset[self.name][:, self.indslat, self.indslon]
        else:
            data = self.dataset[self.name][:, self.indsdepth, self.indslat, self.indslon]

        if np.ma.is_masked(data):  # convert masked array to ndarray
            data = np.ma.filled(data, np.nan)
        return data

    @property
    def time(self):
        if self.time_units is not None:
            dt = num2date(self.dataset[self.dimensions['time']][:],
                          self.time_units, self.calendar)
            offset = num2date(0, self.time_units, self.calendar)
            if type(offset) is datetime:
                dt -= offset
            else:
                # num2date in some cases returns a 'phony' datetime. In that case,
                # parse it as a string.
                # See http://unidata.github.io/netcdf4-python/#netCDF4.num2date
                dt -= parse(str(offset))
            return list(map(timedelta.total_seconds, dt))
        else:
            try:
                return self.dataset[self.dimensions['time']][:]
            except:
                return [None]

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
            calendar = self.dataset[self.dimensions['time']].calendar
            if calendar is ('proleptic_gregorian' or 'standard' or 'gregorian'):
                return calendar
            else:
                # Other calendars means the time can't be converted to datetime object
                # See http://unidata.github.io/netcdf4-python/#netCDF4.num2date
                return 'standard'
        except:
            return 'standard'

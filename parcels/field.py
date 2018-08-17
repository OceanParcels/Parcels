from parcels.loggers import logger
from collections import Iterable
from py import path
import numpy as np
from ctypes import Structure, c_int, c_float, POINTER, pointer
import xarray as xr
from math import cos, pi
import datetime
import math
from .grid import (RectilinearZGrid, RectilinearSGrid, CurvilinearZGrid,
                   CurvilinearSGrid, CGrid, GridCode)


__all__ = ['Field', 'VectorField', 'Geographic', 'GeographicPolar', 'GeographicSquare', 'GeographicPolarSquare']


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
        if field is not None and field.grid.time_origin:
            time = field.grid.time_origin + np.timedelta64(int(time), 's')
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


class GeographicSquare(UnitConverter):
    """ Square distance converter from geometric to geographic coordinates (m2 to degree2) """
    source_unit = 'm2'
    target_unit = 'degree2'

    def to_target(self, value, x, y, z):
        return value / pow(1000. * 1.852 * 60., 2)

    def to_source(self, value, x, y, z):
        return value * pow(1000. * 1.852 * 60., 2)

    def ccode_to_target(self, x, y, z):
        return "pow(1.0 / (1000.0 * 1.852 * 60.0), 2)"

    def ccode_to_source(self, x, y, z):
        return "pow((1000.0 * 1.852 * 60.0), 2)"


class GeographicPolarSquare(UnitConverter):
    """ Square distance converter from geometric to geographic coordinates (m2 to degree2)
        with a correction to account for narrower grid cells closer to the poles.
    """
    source_unit = 'm2'
    target_unit = 'degree2'

    def to_target(self, value, x, y, z):
        return value / pow(1000. * 1.852 * 60. * cos(y * pi / 180), 2)

    def to_source(self, value, x, y, z):
        return value * pow(1000. * 1.852 * 60. * cos(y * pi / 180), 2)

    def ccode_to_target(self, x, y, z):
        return "pow(1.0 / (1000. * 1.852 * 60. * cos(%s * M_PI / 180)), 2)" % y

    def ccode_to_source(self, x, y, z):
        return "pow((1000. * 1.852 * 60. * cos(%s * M_PI / 180)), 2)" % y


unitconverters = {'U': GeographicPolar(), 'V': Geographic(),
                  'Kh_zonal': GeographicPolarSquare(),
                  'Kh_meridional': GeographicSquare()}


class Field(object):
    """Class that encapsulates access to field data.

    :param name: Name of the field
    :param data: 2D, 3D or 4D numpy array of field data.

           1. If data shape is [xdim, ydim], [xdim, ydim, zdim], [xdim, ydim, tdim] or [xdim, ydim, zdim, tdim],
              whichever is relevant for the dataset, use the flag transpose=True
           2. If data shape is [ydim, xdim], [zdim, ydim, xdim], [tdim, ydim, xdim] or [tdim, zdim, ydim, xdim],
              use the flag transpose=False
           3. If data has any other shape, you first need to reorder it
    :param lon: Longitude coordinates (numpy vector or array) of the field (only if grid is None)
    :param lat: Latitude coordinates (numpy vector or array) of the field (only if grid is None)
    :param depth: Depth coordinates (numpy vector or array) of the field (only if grid is None)
    :param time: Time coordinates (numpy vector) of the field (only if grid is None)
    :param mesh: String indicating the type of mesh coordinates and
           units used during velocity interpolation: (only if grid is None)

           1. spherical (default): Lat and lon in degree, with a
              correction for zonal velocity U near the poles.
           2. flat: No conversion, lat/lon are assumed to be in m.
    :param grid: :class:`parcels.grid.Grid` object containing all the lon, lat depth, time
           mesh and time_origin information. Can be constructed from any of the Grid objects
    :param fieldtype: Type of Field to be used for UnitConverter when using FieldLists
           (either 'U', 'V', 'Kh_zonal', 'Kh_Meridional' or None)
    :param transpose: Transpose data to required (lon, lat) layout
    :param vmin: Minimum allowed value on the field. Data below this value are set to zero
    :param vmax: Maximum allowed value on the field. Data above this value are set to zero
    :param time_origin: Time origin (datetime or np.datetime64 object) of the time axis (only if grid is None)
    :param interp_method: Method for interpolation. Either 'linear' or 'nearest'
    :param allow_time_extrapolation: boolean whether to allow for extrapolation in time
           (i.e. beyond the last available time snapshot)
    :param time_periodic: boolean whether to loop periodically over the time component of the Field
           This flag overrides the allow_time_interpolation and sets it to False
    """

    def __init__(self, name, data, lon=None, lat=None, depth=None, time=None, grid=None, mesh='flat',
                 fieldtype=None, transpose=False, vmin=None, vmax=None, time_origin=None,
                 interp_method='linear', allow_time_extrapolation=None, time_periodic=False, **kwargs):
        self.name = name
        self.data = data
        if grid:
            self.grid = grid
        else:
            self.grid = RectilinearZGrid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh)
        self.igrid = -1
        # self.lon, self.lat, self.depth and self.time are not used anymore in parcels.
        # self.grid should be used instead.
        # Those variables are still defined for backwards compatibility with users codes.
        self.lon = self.grid.lon
        self.lat = self.grid.lat
        self.depth = self.grid.depth
        self.time = self.grid.time
        fieldtype = self.name if fieldtype is None else fieldtype
        if self.grid.mesh == 'flat' or (fieldtype not in unitconverters.keys()):
            self.units = UnitConverter()
        elif self.grid.mesh == 'spherical':
            self.units = unitconverters[fieldtype]
        else:
            raise ValueError("Unsupported mesh type. Choose either: 'spherical' or 'flat'")
        if type(interp_method) is dict:
            if name in interp_method:
                self.interp_method = interp_method[name]
            else:
                raise RuntimeError('interp_method is a dictionary but %s is not in it' % name)
        else:
            self.interp_method = interp_method
        self.fieldset = None
        if allow_time_extrapolation is None:
            self.allow_time_extrapolation = True if len(self.grid.time) == 1 else False
        else:
            self.allow_time_extrapolation = allow_time_extrapolation

        self.time_periodic = time_periodic
        if self.time_periodic and self.allow_time_extrapolation:
            logger.warning_once("allow_time_extrapolation and time_periodic cannot be used together.\n \
                                 allow_time_extrapolation is set to False")
            self.allow_time_extrapolation = False

        self.vmin = vmin
        self.vmax = vmax

        if not self.grid.defer_load:
            self.data = self.reshape(self.data, transpose)

            # Hack around the fact that NaN and ridiculously large values
            # propagate in SciPy's interpolators
            self.data[np.isnan(self.data)] = 0.
            if self.vmin is not None:
                self.data[self.data < self.vmin] = 0.
            if self.vmax is not None:
                self.data[self.data > self.vmax] = 0.

        self._scaling_factor = None
        (self.gradientx, self.gradienty) = (None, None)  # to store if Field is a gradient() of another field
        self.is_gradient = False

        # Variable names in JIT code
        self.ccode_data = self.name
        self.dimensions = kwargs.pop('dimensions', None)
        self.indices = kwargs.pop('indices', None)
        self.dataFiles = kwargs.pop('dataFiles', None)

    @classmethod
    def from_netcdf(cls, filenames, variable, dimensions, indices=None, grid=None,
                    mesh='spherical', allow_time_extrapolation=None, time_periodic=False,
                    full_load=False, dimension_filename=None, **kwargs):
        """Create field from netCDF file

        :param filenames: list of filenames to read for the field.
               Note that wildcards ('*') are also allowed
        :param variable: Name of the field to create. Note that this has to be a string
        :param dimensions: Dictionary mapping variable names for the relevant dimensions in the NetCDF file
        :param indices: dictionary mapping indices for each dimension to read from file.
               This can be used for reading in only a subregion of the NetCDF file
        :param mesh: String indicating the type of mesh coordinates and
               units used during velocity interpolation:

               1. spherical (default): Lat and lon in degree, with a
                  correction for zonal velocity U near the poles.
               2. flat: No conversion, lat/lon are assumed to be in m.
        :param allow_time_extrapolation: boolean whether to allow for extrapolation in time
               (i.e. beyond the last available time snapshot)
               Default is False if dimensions includes time, else True
        :param time_periodic: boolean whether to loop periodically over the time component of the FieldSet
               This flag overrides the allow_time_interpolation and sets it to False
        :param full_load: boolean whether to fully load the data or only pre-load them. (default: False)
               It is advised not to fully load the data, since in that case Parcels deals with
               a better memory management during particle set execution.
               full_load is however sometimes necessary for plotting the fields.
        """

        if not isinstance(filenames, Iterable) or isinstance(filenames, str):
            filenames = [filenames]
        dimension_filename = dimension_filename if dimension_filename else filenames[0]
        if indices is None:
            indices = {}
        with NetcdfFileBuffer(dimension_filename, dimensions, indices) as filebuffer:
            lon, lat = filebuffer.read_lonlat
            depth = filebuffer.read_depth
            indices = filebuffer.indices
            # Check if parcels_mesh has been explicitly set in file
            if 'parcels_mesh' in filebuffer.dataset.attrs:
                mesh = filebuffer.dataset.attrs['parcels_mesh']

        if len(filenames) > 1 and 'time' not in dimensions:
            raise RuntimeError('Multiple files given but no time dimension specified')

        if grid is None:
            # Concatenate time variable to determine overall dimension
            # across multiple files
            timeslices = []
            dataFiles = []
            for fname in filenames:
                with NetcdfFileBuffer(fname, dimensions, indices) as filebuffer:
                    ftime = filebuffer.time
                    timeslices.append(ftime)
                    dataFiles.append([fname] * len(ftime))
            timeslices = np.array(timeslices)
            time = np.concatenate(timeslices)
            dataFiles = np.concatenate(np.array(dataFiles))
            if isinstance(time[0], np.datetime64):
                time_origin = time[0]
                time = (time - time_origin) / np.timedelta64(1, 's')
            else:
                time_origin = None
            assert(np.all((time[1:]-time[:-1]) > 0))

            if time.size == 1 and time[0] is None:
                time[0] = 0
            if len(lon.shape) == 1:
                if len(depth.shape) == 1:
                    grid = RectilinearZGrid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh)
                else:
                    grid = RectilinearSGrid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh)
            else:
                if len(depth.shape) == 1:
                    grid = CurvilinearZGrid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh)
                else:
                    grid = CurvilinearSGrid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh)
            grid.timeslices = timeslices
            kwargs['dataFiles'] = dataFiles

        if 'time' in indices:
            logger.warning_once('time dimension in indices is not necessary anymore. It is then ignored.')

        if grid.time.size <= 3 or full_load:
            # Pre-allocate data before reading files into buffer
            data = np.empty((grid.tdim, grid.zdim, grid.ydim, grid.xdim), dtype=np.float32)
            ti = 0
            for tslice, fname in zip(grid.timeslices, filenames):
                with NetcdfFileBuffer(fname, dimensions, indices) as filebuffer:
                    # If Field.from_netcdf is called directly, it may not have a 'data' dimension
                    # In that case, assume that 'name' is the data dimension
                    filebuffer.name = dimensions['data'] if 'data' in dimensions else variable

                    if len(filebuffer.dataset[filebuffer.name].shape) == 2:
                        data[ti:ti+len(tslice), 0, :, :] = filebuffer.data[:, :]
                    elif len(filebuffer.dataset[filebuffer.name].shape) == 3:
                        if len(filebuffer.indices['depth']) > 1:
                            data[ti:ti+len(tslice), :, :, :] = filebuffer.data[:, :, :]
                        else:
                            data[ti:ti+len(tslice), 0, :, :] = filebuffer.data[:, :, :]
                    else:
                        data[ti:ti+len(tslice), :, :, :] = filebuffer.data[:, :, :, :]
                ti += len(tslice)
        else:
            grid.defer_load = True
            grid.time_full = grid.time
            grid.ti = -1
            data = None

        if allow_time_extrapolation is None:
            allow_time_extrapolation = False if 'time' in dimensions else True

        kwargs['dimensions'] = dimensions.copy()
        kwargs['indices'] = indices
        kwargs['time_periodic'] = time_periodic

        return cls(variable, data, grid=grid,
                   allow_time_extrapolation=allow_time_extrapolation, **kwargs)

    def reshape(self, data, transpose=False):

        # Ensure that field data is the right data type
        if not data.dtype == np.float32:
            logger.warning_once("Casting field data to np.float32")
            data = data.astype(np.float32)
        if transpose:
            data = np.transpose(data)
        if self.grid.lat_flipped:
            data = np.flip(data, axis=-2)

        if self.grid.tdim == 1:
            if len(data.shape) < 4:
                data = data.reshape(sum(((1,), data.shape), ()))
        if self.grid.zdim == 1:
            if len(data.shape) == 4:
                data = data.reshape(sum(((data.shape[0],), data.shape[2:]), ()))
        if len(data.shape) == 4:
            assert data.shape == (self.grid.tdim, self.grid.zdim, self.grid.ydim-2*self.grid.meridional_halo, self.grid.xdim-2*self.grid.zonal_halo), \
                                 ('Field %s expecting a data shape of a [ydim, xdim], [zdim, ydim, xdim], [tdim, ydim, xdim] or [tdim, zdim, ydim, xdim]. Flag transpose=True could help to reorder the data.')
        else:
            assert data.shape == (self.grid.tdim, self.grid.ydim-2*self.grid.meridional_halo, self.grid.xdim-2*self.grid.zonal_halo), \
                                 ('Field %s expecting a data shape of a [ydim, xdim], [zdim, ydim, xdim], [tdim, ydim, xdim] or [tdim, zdim, ydim, xdim]. Flag transpose=True could help to reorder the data.')
        if self.grid.meridional_halo > 0 or self.grid.zonal_halo > 0:
            data = self.add_periodic_halo(zonal=self.grid.zonal_halo > 0, meridional=self.grid.meridional_halo > 0, halosize=max(self.grid.meridional_halo, self.grid.zonal_halo), data=data)
        return data

    def set_scaling_factor(self, factor):
        """Scales the field data by some constant factor.

        :param factor: scaling factor
        """

        if self._scaling_factor:
            raise NotImplementedError(('Scaling factor for field %s already defined.' % self.name))
        self._scaling_factor = factor
        if not self.grid.defer_load:
            self.data *= factor

    def __getitem__(self, key):
        return self.eval(*key)

    def calc_cell_edge_sizes(self):
        """Method to calculate cell sizes based on numpy.gradient method
                Currently only works for Rectilinear Grids"""
        if not self.grid.cell_edge_sizes:
            if self.grid.gtype in (GridCode.RectilinearZGrid, GridCode.RectilinearSGrid):
                self.grid.cell_edge_sizes['x'] = np.zeros((self.grid.ydim, self.grid.xdim), dtype=np.float32)
                self.grid.cell_edge_sizes['y'] = np.zeros((self.grid.ydim, self.grid.xdim), dtype=np.float32)

                x_conv = GeographicPolar() if self.grid.mesh is 'spherical' else UnitConverter()
                y_conv = Geographic() if self.grid.mesh is 'spherical' else UnitConverter()
                for y, (lat, dy) in enumerate(zip(self.grid.lat, np.gradient(self.grid.lat))):
                    for x, (lon, dx) in enumerate(zip(self.grid.lon, np.gradient(self.grid.lon))):
                        self.grid.cell_edge_sizes['x'][y, x] = x_conv.to_source(dx, lon, lat, self.grid.depth[0])
                        self.grid.cell_edge_sizes['y'][y, x] = y_conv.to_source(dy, lon, lat, self.grid.depth[0])
                self.cell_edge_sizes = self.grid.cell_edge_sizes
            else:
                logger.error(('Field.cell_edge_sizes() not implemented for ', self.grid.gtype, 'grids.',
                              'You can provide Field.grid.cell_edge_sizes yourself',
                              'by in e.g. NEMO using the e1u fields etc from the mesh_mask.nc file'))
                exit(-1)

    def cell_areas(self):
        """Method to calculate cell sizes based on cell_edge_sizes
                Currently only works for Rectilinear Grids"""
        if not self.grid.cell_edge_sizes:
            self.calc_cell_edge_sizes()
        return self.grid.cell_edge_sizes['x'] * self.grid.cell_edge_sizes['y']

    def gradient(self, update=False):
        """Method to calculate horizontal gradients of Field.
                Returns two Fields: the zonal and meridional gradients,
                on the same Grid as the original Field, using numpy.gradient() method
                Names of these grids are dNAME_dx and dNAME_dy, where NAME is the name
                of the original Field"""
        if not self.grid.cell_edge_sizes:
            self.calc_cell_edge_sizes()
        if self.grid.defer_load and self.data is None:
            (dFdx, dFdy) = (None, None)
        else:
            dFdy = np.gradient(self.data, axis=-2) / self.grid.cell_edge_sizes['y']
            dFdx = np.gradient(self.data, axis=-1) / self.grid.cell_edge_sizes['x']
        if update:
            self.gradientx.data = dFdx
            self.gradienty.data = dFdy
        else:
            dFdx_fld = Field('d%s_dx' % self.name, dFdx, grid=self.grid)
            dFdy_fld = Field('d%s_dy' % self.name, dFdy, grid=self.grid)
            dFdx_fld.is_gradient = True
            dFdy_fld.is_gradient = True
            (self.gradientx, self.gradienty) = (dFdx_fld, dFdy_fld)
            return (dFdx_fld, dFdy_fld)

    def search_indices_vertical_z(self, z):
        grid = self.grid
        z = np.float32(z)
        depth_index = grid.depth <= z
        if z >= grid.depth[-1]:
            zi = len(grid.depth) - 2
        else:
            zi = depth_index.argmin() - 1 if z >= grid.depth[0] else 0
        zeta = (z-grid.depth[zi]) / (grid.depth[zi+1]-grid.depth[zi])
        return (zi, zeta)

    def search_indices_vertical_s(self, x, y, z, xi, yi, xsi, eta, ti, time):
        grid = self.grid
        if time < grid.time[ti]:
            ti -= 1
        if grid.z4d:
            if ti == len(grid.time)-1:
                depth_vector = (1-xsi)*(1-eta) * grid.depth[-1, :, yi, xi] + \
                    xsi*(1-eta) * grid.depth[-1, :, yi, xi+1] + \
                    xsi*eta * grid.depth[-1, :, yi+1, xi+1] + \
                    (1-xsi)*eta * grid.depth[-1, :, yi+1, xi]
            else:
                dv2 = (1-xsi)*(1-eta) * grid.depth[ti:ti+2, :, yi, xi] + \
                    xsi*(1-eta) * grid.depth[ti:ti+2, :, yi, xi+1] + \
                    xsi*eta * grid.depth[ti:ti+2, :, yi+1, xi+1] + \
                    (1-xsi)*eta * grid.depth[ti:ti+2, :, yi+1, xi]
                tt = (time-grid.time[ti]) / (grid.time[ti+1]-grid.time[ti])
                assert tt >= 0 and tt <= 1, 'Vertical s grid is being wrongly interpolated in time'
                depth_vector = dv2[0, :] * (1-tt) + dv2[1, :] * tt
        else:
            depth_vector = (1-xsi)*(1-eta) * grid.depth[:, yi, xi] + \
                xsi*(1-eta) * grid.depth[:, yi, xi+1] + \
                xsi*eta * grid.depth[:, yi+1, xi+1] + \
                (1-xsi)*eta * grid.depth[:, yi+1, xi]
        z = np.float32(z)
        depth_index = depth_vector <= z
        if z >= depth_vector[-1]:
            zi = len(depth_vector) - 2
        else:
            zi = depth_index.argmin() - 1 if z >= depth_vector[0] else 0
        if z < depth_vector[zi] or z > depth_vector[zi+1]:
            raise FieldSamplingError(x, y, z, field=self)
        zeta = (z - depth_vector[zi]) / (depth_vector[zi+1]-depth_vector[zi])
        return (zi, zeta)

    def reconnect_bnd_indices(self, xi, yi, xdim, ydim, sphere_mesh):
        if xi < 0:
            if sphere_mesh:
                xi = xdim-2
            else:
                xi = 0
        if xi > xdim-2:
            if sphere_mesh:
                xi = 0
            else:
                xi = xdim-2
        if yi < 0:
            yi = 0
        if yi > ydim-2:
            yi = ydim-2
            if sphere_mesh:
                xi = xdim - xi
        return xi, yi

    def search_indices_rectilinear(self, x, y, z, ti=-1, time=-1, search2D=False):
        grid = self.grid
        xi = yi = -1

        if grid.mesh is not 'spherical':
            if x < grid.lon[0] or x > grid.lon[-1]:
                raise FieldSamplingError(x, y, z, field=self)
            lon_index = grid.lon < x
            if lon_index.all():
                xi = len(grid.lon) - 2
            else:
                xi = lon_index.argmin() - 1 if lon_index.any() else 0
            xsi = (x-grid.lon[xi]) / (grid.lon[xi+1]-grid.lon[xi])
            if xsi < 0:
                xi -= 1
                xsi = (x-grid.lon[xi]) / (grid.lon[xi+1]-grid.lon[xi])
            elif xsi > 1:
                xi += 1
                xsi = (x-grid.lon[xi]) / (grid.lon[xi+1]-grid.lon[xi])
        else:
            lon_fixed = grid.lon.copy()
            indices = lon_fixed >= lon_fixed[0]
            if not indices.all():
                lon_fixed[indices.argmin():] += 360
            if x < lon_fixed[0]:
                lon_fixed -= 360
            if not grid.zonal_periodic:
                if (grid.lon[0] < grid.lon[-1]) and (x < grid.lon[0] or x > grid.lon[-1]):
                    raise FieldSamplingError(x, y, z, field=self)
                elif (grid.lon[0] >= grid.lon[-1]) and (x < grid.lon[0] and x > grid.lon[-1]):
                    raise FieldSamplingError(x, y, z, field=self)

            lon_index = lon_fixed < x
            if lon_index.all():
                xi = len(lon_fixed) - 2
            else:
                xi = lon_index.argmin() - 1 if lon_index.any() else 0
            xsi = (x-lon_fixed[xi]) / (lon_fixed[xi+1]-lon_fixed[xi])
            if xsi < 0:
                xi -= 1
                xsi = (x-lon_fixed[xi]) / (lon_fixed[xi+1]-lon_fixed[xi])
            elif xsi > 1:
                xi += 1
                xsi = (x-lon_fixed[xi]) / (lon_fixed[xi+1]-lon_fixed[xi])

        if y < grid.lat[0] or y > grid.lat[-1]:
            raise FieldSamplingError(x, y, z, field=self)
        lat_index = grid.lat < y
        if lat_index.all():
            yi = len(grid.lat) - 2
        else:
            yi = lat_index.argmin() - 1 if lat_index.any() else 0

        eta = (y-grid.lat[yi]) / (grid.lat[yi+1]-grid.lat[yi])
        if eta < 0:
            yi -= 1
            eta = (y-grid.lat[yi]) / (grid.lat[yi+1]-grid.lat[yi])
        elif eta > 1:
            yi += 1
            eta = (y-grid.lat[yi]) / (grid.lat[yi+1]-grid.lat[yi])

        if grid.zdim > 1 and not search2D:
            if grid.gtype == GridCode.RectilinearZGrid:
                # Never passes here, because in this case, we work with scipy
                (zi, zeta) = self.search_indices_vertical_z(z)
            elif grid.gtype == GridCode.RectilinearSGrid:
                (zi, zeta) = self.search_indices_vertical_s(x, y, z, xi, yi, xsi, eta, ti, time)
        else:
            zi = -1
            zeta = 0

        assert(xsi >= 0 and xsi <= 1)
        assert(eta >= 0 and eta <= 1)
        assert(zeta >= 0 and zeta <= 1)

        return (xsi, eta, zeta, xi, yi, zi)

    def search_indices_curvilinear(self, x, y, z, xi, yi, ti=-1, time=-1, search2D=False):
        xsi = eta = -1
        grid = self.grid
        invA = np.array([[1, 0, 0, 0],
                         [-1, 1, 0, 0],
                         [-1, 0, 0, 1],
                         [1, -1, 1, -1]])
        maxIterSearch = 1e6
        it = 0
        if (not grid.zonal_periodic) or grid.mesh == 'flat':
            if (grid.lon[0, 0] < grid.lon[0, -1]) and (x < grid.lon[0, 0] or x > grid.lon[0, -1]):
                raise FieldSamplingError(x, y, z, field=self)
            elif (grid.lon[0, 0] >= grid.lon[0, -1]) and (x < grid.lon[0, 0] and x > grid.lon[0, -1]):
                raise FieldSamplingError(x, y, z, field=self)
        if y < np.min(grid.lat) or y > np.max(grid.lat):
            raise FieldSamplingError(x, y, z, field=self)

        while xsi < 0 or xsi > 1 or eta < 0 or eta > 1:
            px = np.array([grid.lon[yi, xi], grid.lon[yi, xi+1], grid.lon[yi+1, xi+1], grid.lon[yi+1, xi]])
            if grid.mesh == 'spherical':
                px[0] = px[0]+360 if px[0] < x-225 else px[0]
                px[0] = px[0]-360 if px[0] > x+225 else px[0]
                px[1:] = np.where(px[1:] - px[0] > 180, px[1:]-360, px[1:])
                px[1:] = np.where(-px[1:] + px[0] > 180, px[1:]+360, px[1:])
            py = np.array([grid.lat[yi, xi], grid.lat[yi, xi+1], grid.lat[yi+1, xi+1], grid.lat[yi+1, xi]])
            a = np.dot(invA, px)
            b = np.dot(invA, py)

            aa = a[3]*b[2] - a[2]*b[3]
            bb = a[3]*b[0] - a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + x*b[3] - y*a[3]
            cc = a[1]*b[0] - a[0]*b[1] + x*b[1] - y*a[1]
            if abs(aa) < 1e-12:  # Rectilinear cell, or quasi
                eta = -cc / bb
            else:
                det2 = bb*bb-4*aa*cc
                if det2 > 0:  # so, if det is nan we keep the xsi, eta from previous iter
                    det = np.sqrt(det2)
                    eta = (-bb+det)/(2*aa)
            xsi = (x-a[0]-a[2]*eta) / (a[1]+a[3]*eta)
            if xsi < 0 and eta < 0 and xi == 0 and yi == 0:
                raise FieldSamplingError(x, y, 0, field=self)
            if xsi > 1 and eta > 1 and xi == grid.xdim-1 and yi == grid.ydim-1:
                raise FieldSamplingError(x, y, 0, field=self)
            if xsi < 0:
                xi -= 1
            elif xsi > 1:
                xi += 1
            if eta < 0:
                yi -= 1
            elif eta > 1:
                yi += 1
            (xi, yi) = self.reconnect_bnd_indices(xi, yi, grid.xdim, grid.ydim, grid.mesh)
            it += 1
            if it > maxIterSearch:
                print('Correct cell not found after %d iterations' % maxIterSearch)
                raise FieldSamplingError(x, y, 0, field=self)

        if grid.zdim > 1 and not search2D:
            if grid.gtype == GridCode.CurvilinearZGrid:
                (zi, zeta) = self.search_indices_vertical_z(z)
            elif grid.gtype == GridCode.CurvilinearSGrid:
                (zi, zeta) = self.search_indices_vertical_s(x, y, z, xi, yi, xsi, eta, ti, time)
        else:
            zi = -1
            zeta = 0

        assert(xsi >= 0 and xsi <= 1)
        assert(eta >= 0 and eta <= 1)
        assert(zeta >= 0 and zeta <= 1)

        return (xsi, eta, zeta, xi, yi, zi)

    def search_indices(self, x, y, z, xi, yi, ti=-1, time=-1, search2D=False):
        if self.grid.gtype in [GridCode.RectilinearSGrid, GridCode.RectilinearZGrid]:
            return self.search_indices_rectilinear(x, y, z, ti, time, search2D=search2D)
        else:
            return self.search_indices_curvilinear(x, y, z, xi, yi, ti, time, search2D=search2D)

    def interpolator2D(self, ti, z, y, x):
        xi = 0
        yi = 0
        (xsi, eta, _, xi, yi, _) = self.search_indices(x, y, z, xi, yi)
        if self.interp_method is 'nearest':
            xii = xi if xsi <= .5 else xi+1
            yii = yi if eta <= .5 else yi+1
            return self.data[ti, yii, xii]
        elif self.interp_method is 'linear':
            val = (1-xsi)*(1-eta) * self.data[ti, yi, xi] + \
                xsi*(1-eta) * self.data[ti, yi, xi+1] + \
                xsi*eta * self.data[ti, yi+1, xi+1] + \
                (1-xsi)*eta * self.data[ti, yi+1, xi]
            return val
        else:
            raise RuntimeError(self.interp_method+" is not implemented for 2D grids")

    def interpolator3D(self, ti, z, y, x, time):
        xi = int(self.grid.xdim / 2)
        yi = int(self.grid.ydim / 2)
        (xsi, eta, zeta, xi, yi, zi) = self.search_indices(x, y, z, xi, yi, ti, time)
        if self.interp_method is 'nearest':
            xii = xi if xsi <= .5 else xi+1
            yii = yi if eta <= .5 else yi+1
            zii = zi if zeta <= .5 else zi+1
            return self.data[ti, zii, yii, xii]
        elif self.interp_method is 'cgrid_linear':
            # evaluating W velocity in c_grid
            f0 = self.data[ti, zi, yi, xi]
            f1 = self.data[ti, zi+1, yi, xi]
            return (1-zeta) * f0 + zeta * f1
        elif self.interp_method is 'linear':
            data = self.data[ti, zi, :, :]
            f0 = (1-xsi)*(1-eta) * data[yi, xi] + \
                xsi*(1-eta) * data[yi, xi+1] + \
                xsi*eta * data[yi+1, xi+1] + \
                (1-xsi)*eta * data[yi+1, xi]
            data = self.data[ti, zi+1, :, :]
            f1 = (1-xsi)*(1-eta) * data[yi, xi] + \
                xsi*(1-eta) * data[yi, xi+1] + \
                xsi*eta * data[yi+1, xi+1] + \
                (1-xsi)*eta * data[yi+1, xi]
            return (1-zeta) * f0 + zeta * f1
        else:
            raise RuntimeError(self.interp_method+" is not implemented for 3D grids")

    def temporal_interpolate_fullfield(self, ti, time):
        """Calculate the data of a field between two snapshots,
        using linear interpolation

        :param ti: Index in time array associated with time (via :func:`time_index`)
        :param time: Time to interpolate to

        :rtype: Linearly interpolated field"""
        t0 = self.grid.time[ti]
        t1 = self.grid.time[ti+1]
        f0 = self.data[ti, :]
        f1 = self.data[ti+1, :]
        return f0 + (f1 - f0) * ((time - t0) / (t1 - t0))

    def spatial_interpolation(self, ti, z, y, x, time):
        """Interpolate horizontal field values using a SciPy interpolator"""

        if self.grid.zdim == 1:
            val = self.interpolator2D(ti, z, y, x)
        else:
            val = self.interpolator3D(ti, z, y, x, time)
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

    def eval(self, time, x, y, z, applyConversion=True):
        """Interpolate field values in space and time.

        We interpolate linearly in time and apply implicit unit
        conversion to the result. Note that we defer to
        scipy.interpolate to perform spatial interpolation.
        """
        (ti, periods) = self.time_index(time)
        time -= periods*(self.grid.time[-1]-self.grid.time[0])
        if ti < self.grid.tdim-1 and time > self.grid.time[ti]:
            f0 = self.spatial_interpolation(ti, z, y, x, time)
            f1 = self.spatial_interpolation(ti + 1, z, y, x, time)
            t0 = self.grid.time[ti]
            t1 = self.grid.time[ti + 1]
            value = f0 + (f1 - f0) * ((time - t0) / (t1 - t0))
        else:
            # Skip temporal interpolation if time is outside
            # of the defined time range or if we have hit an
            # excat value in the time array.
            value = self.spatial_interpolation(ti, z, y, x, self.grid.time[ti])

        if applyConversion:
            return self.units.to_target(value, x, y, z)
        else:
            return value

    def ccode_eval(self, var, t, x, y, z):
        # Casting interp_methd to int as easier to pass on in C-code
        return "temporal_interpolation(%s, %s, %s, %s, %s, particle->cxi, particle->cyi, particle->czi, particle->cti, &%s, %s)" \
            % (x, y, z, t, self.name, var, self.interp_method.upper())

    def ccode_convert(self, _, x, y, z):
        return self.units.ccode_to_target(x, y, z)

    @property
    def ctypes_struct(self):
        """Returns a ctypes struct object containing all relevant
        pointers and sizes for this field."""

        # Ctypes struct corresponding to the type definition in parcels.h
        class CField(Structure):
            _fields_ = [('xdim', c_int), ('ydim', c_int), ('zdim', c_int),
                        ('tdim', c_int), ('igrid', c_int),
                        ('allow_time_extrapolation', c_int),
                        ('time_periodic', c_int),
                        ('data', POINTER(POINTER(c_float))),
                        ('grid', POINTER(CGrid))]

        # Create and populate the c-struct object
        allow_time_extrapolation = 1 if self.allow_time_extrapolation else 0
        time_periodic = 1 if self.time_periodic else 0
        cstruct = CField(self.grid.xdim, self.grid.ydim, self.grid.zdim,
                         self.grid.tdim, self.igrid, allow_time_extrapolation, time_periodic,
                         self.data.ctypes.data_as(POINTER(POINTER(c_float))),
                         pointer(self.grid.ctypes_struct))
        return cstruct

    def show(self, animation=False, show_time=None, domain=None, projection=None, land=None,
             vmin=None, vmax=None, savefile=None, **kwargs):
        """Method to 'show' a Parcels Field

        :param animation: Boolean whether result is a single plot, or an animation
        :param show_time: Time at which to show the Field (only in single-plot mode)
        :param domain: Four-vector (latN, latS, lonE, lonW) defining domain to show
        :param projection: type of cartopy projection to use (default PlateCarree)
        :param land: Boolean whether to show land
        :param vmin: minimum colour scale (only in single-plot mode)
        :param vmax: maximum colour scale (only in single-plot mode)
        :param savefile: Name of a file to save the plot to
        """
        from parcels.plotting import plotfield
        plt, _, _, _ = plotfield(self, animation=animation, show_time=show_time, domain=domain, projection=projection,
                                 land=land, vmin=vmin, vmax=vmax, savefile=savefile, **kwargs)
        if plt:
            plt.show()

    def add_periodic_halo(self, zonal, meridional, halosize=5, data=None):
        """Add a 'halo' to all Fields in a FieldSet, through extending the Field (and lon/lat)
        by copying a small portion of the field on one side of the domain to the other.
        Before adding a periodic halo to the Field, it has to be added to the Grid on which the Field depends

        :param zonal: Create a halo in zonal direction (boolean)
        :param meridional: Create a halo in meridional direction (boolean)
        :param halosize: size of the halo (in grid points). Default is 5 grid points
        :param data: if data is not None, the periodic halo will be achieved on data instead of self.data and data will be returned
        """
        dataNone = not isinstance(data, np.ndarray)
        if self.grid.defer_load and dataNone:
            return
        data = self.data if dataNone else data
        if zonal:
            if len(data.shape) is 3:
                data = np.concatenate((data[:, :, -halosize:], data,
                                       data[:, :, 0:halosize]), axis=len(data.shape)-1)
                assert data.shape[2] == self.grid.xdim
            else:
                data = np.concatenate((data[:, :, :, -halosize:], data,
                                       data[:, :, :, 0:halosize]), axis=len(data.shape) - 1)
                assert data.shape[3] == self.grid.xdim
            self.lon = self.grid.lon
            self.lat = self.grid.lat
        if meridional:
            if len(data.shape) is 3:
                data = np.concatenate((data[:, -halosize:, :], data,
                                       data[:, 0:halosize, :]), axis=len(data.shape)-2)
                assert data.shape[1] == self.grid.ydim
            else:
                data = np.concatenate((data[:, :, -halosize:, :], data,
                                       data[:, :, 0:halosize, :]), axis=len(data.shape) - 2)
                assert data.shape[2] == self.grid.ydim
            self.lat = self.grid.lat
        if dataNone:
            self.data = data
        else:
            return data

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
        if type(self.grid) is RectilinearZGrid:
            nav_lon = xr.DataArray(self.grid.lon + np.zeros((self.grid.ydim, self.grid.xdim), dtype=np.float32),
                                   coords=[('y', self.grid.lat), ('x', self.grid.lon)])
            nav_lat = xr.DataArray(self.grid.lat.reshape(self.grid.ydim, 1) + np.zeros(self.grid.xdim, dtype=np.float32),
                                   coords=[('y', self.grid.lat), ('x', self.grid.lon)])
        elif type(self.grid) is CurvilinearZGrid:
            nav_lon = xr.DataArray(self.grid.lon, coords=[('y', range(self.grid.ydim)),
                                                          ('x', range(self.grid.xdim))])
            nav_lat = xr.DataArray(self.grid.lat, coords=[('y', range(self.grid.ydim)),
                                                          ('x', range(self.grid.xdim))])
        else:
            raise NotImplementedError('Field.write only implemented for RectilinearZGrid and CurvilinearZGrid')

        attrs = {'units': 'seconds since ' + str(self.grid.time_origin)} if self.grid.time_origin else {}
        time_counter = xr.DataArray(self.grid.time,
                                    dims=['time_counter'],
                                    attrs=attrs)
        vardata = xr.DataArray(self.data.reshape((self.grid.tdim, self.grid.zdim, self.grid.ydim, self.grid.xdim)),
                               dims=['time_counter', vname_depth, 'y', 'x'])
        # Create xarray Dataset and output to netCDF format
        attrs = {'parcels_mesh': self.grid.mesh}
        dset = xr.Dataset({varname: vardata}, coords={'nav_lon': nav_lon,
                                                      'nav_lat': nav_lat,
                                                      'time_counter': time_counter,
                                                      vname_depth: self.grid.depth}, attrs=attrs)
        dset.to_netcdf(filepath)

    def advancetime(self, field_new, advanceForward):
        if advanceForward == 1:  # forward in time, so appending at end
            self.data = np.concatenate((self.data[1:, :, :], field_new.data[:, :, :]), 0)
            self.time = self.grid.time
        else:  # backward in time, so prepending at start
            self.data = np.concatenate((field_new.data[:, :, :], self.data[:-1, :, :]), 0)
            self.time = self.grid.time

    def computeTimeChunk(self, data, tindex):
        g = self.grid
        with NetcdfFileBuffer(self.dataFiles[g.ti+tindex], self.dimensions, self.indices) as filebuffer:
            filebuffer.name = self.dimensions['data'] if 'data' in self.dimensions else self.name
            time_data = filebuffer.time
            if isinstance(time_data[0], np.datetime64):
                assert isinstance(time_data[0], type(g.time_origin)), ('Field %s stores times as dates, but time_origin is not defined ' % self.name)
                time_data = (time_data - g.time_origin) / np.timedelta64(1, 's')
            ti = (time_data <= g.time[tindex]).argmin() - 1
            if len(filebuffer.dataset[filebuffer.name].shape) == 2:
                data[tindex, 0, :, :] = filebuffer.data[:, :]
            elif len(filebuffer.dataset[filebuffer.name].shape) == 3:
                if g.zdim > 1:
                    data[tindex, :, :, :] = filebuffer.data[:, :, :]
                else:
                    data[tindex, 0, :, :] = filebuffer.data[ti, :, :]
            else:
                data[tindex, :, :, :] = filebuffer.data[ti, :, :, :]
        data[np.isnan(data)] = 0.
        if self.vmin is not None:
            data[data < self.vmin] = 0.
        if self.vmax is not None:
            data[data > self.vmax] = 0.

        return data


class VectorField(object):
    """Class VectorField stores 2 or 3 fields which defines together a vector field.
    This enables to interpolate them as one single vector field in the kernels.

    :param name: Name of the vector field
    :param U: field defining the zonal component
    :param V: field defining the meridional component
    :param W: field defining the vertical component (default: None)
    """
    def __init__(self, name, U, V, W=None):
        self.name = name
        self.U = U
        self.V = V
        self.W = W
        if self.U.interp_method == 'cgrid_linear':
            assert self.V.interp_method == 'cgrid_linear'
            assert self.U.grid is self.V.grid
            if W:
                assert self.W.interp_method == 'cgrid_linear'
                assert self.U.grid is self.W.grid

    def dist(self, lon1, lon2, lat1, lat2, mesh):
        if mesh == 'spherical':
            r = 360*60*1852/2/np.pi
            rad = np.pi/180.
            x1 = r*np.cos(rad*lon1) * np.cos(rad*lat1)
            y1 = r*np.sin(rad*lon1) * np.cos(rad*lat1)
            z1 = r*np.sin(rad*lat1)
            x2 = r*np.cos(rad*lon2) * np.cos(rad*lat2)
            y2 = r*np.sin(rad*lon2) * np.cos(rad*lat2)
            z2 = r*np.sin(rad*lat2)
            return np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        else:
            return np.sqrt((lon2-lon1)**2 + (lat2-lat1)**2)

    def jacobian(self, xsi, eta, px, py):
        dphidxsi = [eta-1, 1-eta, eta, -eta]
        dphideta = [xsi-1, -xsi, xsi, 1-xsi]

        dxdxsi = np.dot(px, dphidxsi)
        dxdeta = np.dot(px, dphideta)
        dydxsi = np.dot(py, dphidxsi)
        dydeta = np.dot(py, dphideta)
        jac = dxdxsi*dydeta - dxdeta*dydxsi
        return jac

    def spatial_c_grid_interpolation2D(self, ti, z, y, x, time):
        grid = self.U.grid
        xi = int(grid.xdim / 2)
        yi = int(grid.ydim / 2)
        (xsi, eta, zeta, xi, yi, zi) = self.U.search_indices(x, y, z, xi, yi, ti, time)

        if grid.gtype in [GridCode.RectilinearSGrid, GridCode.RectilinearZGrid]:
            px = np.array([grid.lon[xi], grid.lon[xi+1], grid.lon[xi+1], grid.lon[xi]])
            py = np.array([grid.lat[yi], grid.lat[yi], grid.lat[yi+1], grid.lat[yi+1]])
        else:
            px = np.array([grid.lon[yi, xi], grid.lon[yi, xi+1], grid.lon[yi+1, xi+1], grid.lon[yi+1, xi]])
            py = np.array([grid.lat[yi, xi], grid.lat[yi, xi+1], grid.lat[yi+1, xi+1], grid.lat[yi+1, xi]])

        if grid.mesh == 'spherical':
            px[0] = px[0]+360 if px[0] < x-225 else px[0]
            px[0] = px[0]-360 if px[0] > x+225 else px[0]
            px[1:] = np.where(px[1:] - px[0] > 180, px[1:]-360, px[1:])
            px[1:] = np.where(-px[1:] + px[0] > 180, px[1:]+360, px[1:])
        xx = (1-xsi)*(1-eta) * px[0] + xsi*(1-eta) * px[1] + xsi*eta * px[2] + (1-xsi)*eta * px[3]
        assert abs(xx-x) < 1e-4
        c1 = self.dist(px[0], px[1], py[0], py[1], grid.mesh)
        c2 = self.dist(px[1], px[2], py[1], py[2], grid.mesh)
        c3 = self.dist(px[2], px[3], py[2], py[3], grid.mesh)
        c4 = self.dist(px[3], px[0], py[3], py[0], grid.mesh)
        if grid.zdim == 1:
            U0 = self.U.data[ti, yi+1, xi] * c4
            U1 = self.U.data[ti, yi+1, xi+1] * c2
            V0 = self.V.data[ti, yi, xi+1] * c1
            V1 = self.V.data[ti, yi+1, xi+1] * c3
        else:
            U0 = self.U.data[ti, zi, yi+1, xi] * c4  # zi here??
            U1 = self.U.data[ti, zi, yi+1, xi+1] * c2
            V0 = self.V.data[ti, zi, yi, xi+1] * c1
            V1 = self.V.data[ti, zi, yi+1, xi+1] * c3
        U = (1-xsi) * U0 + xsi * U1
        V = (1-eta) * V0 + eta * V1
        rad = np.pi/180.
        deg2m = 1852 * 60.
        meshJac = (deg2m * deg2m * cos(rad * y)) if grid.mesh == 'spherical' else 1
        jac = self.jacobian(xsi, eta, px, py) * meshJac

        u = ((-(1-eta) * U - (1-xsi) * V) * px[0] +
             ((1-eta) * U - xsi * V) * px[1] +
             (eta * U + xsi * V) * px[2] +
             (-eta * U + (1-xsi) * V) * px[3]) / jac
        v = ((-(1-eta) * U - (1-xsi) * V) * py[0] +
             ((1-eta) * U - xsi * V) * py[1] +
             (eta * U + xsi * V) * py[2] +
             (-eta * U + (1-xsi) * V) * py[3]) / jac
        return (u, v)

    def spatial_c_grid_interpolation3D(self, ti, z, y, x, time):
        """
          __ V1 __
        |          |
        U0         U1
        | __ V0 __ |
        The interpolation is done in the following by
        interpolating linearly U depending on the longitude coordinate and
        interpolating linearly V depending on the latitude coordinate.
        Curvilinear grids are treated properly, since the element is projected to a rectilinear parent element.
        """
        if self.U.grid.gtype in [GridCode.RectilinearSGrid, GridCode.CurvilinearSGrid]:
            raise NotImplementedError('C staggered grid with a s vertical discretisation are not available')
        (u, v) = self.spatial_c_grid_interpolation2D(ti, z, y, x, time)
        w = self.W.eval(time, x, y, z, False)
        w = self.W.units.to_target(w, x, y, z)
        return (u, v, w)

    def eval(self, time, x, y, z):
        if self.U.interp_method != 'cgrid_linear':
            u = self.U.eval(time, x, y, z, False)
            v = self.V.eval(time, x, y, z, False)
            u = self.U.units.to_target(u, x, y, z)
            v = self.V.units.to_target(v, x, y, z)
            if self.W is not None:
                w = self.W.eval(time, x, y, z, False)
                w = self.W.units.to_target(w, x, y, z)
                return (u, v, w)
            else:
                return (u, v)
        else:
            grid = self.U.grid
            (ti, periods) = self.U.time_index(time)
            time -= periods*(grid.time[-1]-grid.time[0])
            if ti < grid.tdim-1 and time > self.grid.time[ti]:
                t0 = grid.time[ti]
                t1 = grid.time[ti + 1]
                if self.W:
                    (u0, v0, w0) = self.spatial_c_grid_interpolation3D(ti, z, y, x, time)
                    (u1, v1, w1) = self.spatial_c_grid_interpolation3D(ti + 1, z, y, x, time)
                    w = w0 + (w1 - w0) * ((time - t0) / (t1 - t0))
                else:
                    (u0, v0) = self.spatial_c_grid_interpolation2D(ti, z, y, x, time)
                    (u1, v1) = self.spatial_c_grid_interpolation2D(ti + 1, z, y, x, time)
                u = u0 + (u1 - u0) * ((time - t0) / (t1 - t0))
                v = v0 + (v1 - v0) * ((time - t0) / (t1 - t0))
                if self.W:
                    return (u, v, w)
                else:
                    return (u, v)
            else:
                # Skip temporal interpolation if time is outside
                # of the defined time range or if we have hit an
                # excat value in the time array.
                if self.W:
                    return self.spatial_c_grid_interpolation3D(ti, z, y, x, grid.time[ti])
                else:
                    return self.spatial_c_grid_interpolation2D(ti, z, y, x, grid.time[ti])

    def __getitem__(self, key):
        return self.eval(*key)

    def ccode_eval(self, varU, varV, varW, U, V, W, t, x, y, z):
        # Casting interp_methd to int as easier to pass on in C-code
        if varW:
            return "temporal_interpolationUVW(%s, %s, %s, %s, %s, %s, %s, " \
                   % (x, y, z, t, U.name, V.name, W.name) + \
                   "particle->cxi, particle->cyi, particle->czi, particle->cti, &%s, &%s, &%s, %s)" \
                   % (varU, varV, varW, U.interp_method.upper())
        else:
            return "temporal_interpolationUV(%s, %s, %s, %s, %s, %s, " \
                   % (x, y, z, t, U.name, V.name) + \
                   "particle->cxi, particle->cyi, particle->czi, particle->cti, &%s, &%s, %s)" \
                   % (varU, varV, U.interp_method.upper())


class NetcdfFileBuffer(object):
    """ Class that encapsulates and manages deferred access to file data. """

    def __init__(self, filename, dimensions, indices):
        self.filename = filename
        self.dimensions = dimensions  # Dict with dimension keyes for file data
        self.indices = indices
        self.dataset = None

    def __enter__(self):
        try:
            self.dataset = xr.open_dataset(str(self.filename), decode_cf=True)
            self.dataset['decoded'] = True
        except:
            self.dataset = xr.open_dataset(str(self.filename), decode_cf=False)
            self.dataset['decoded'] = False
        for inds in self.indices.values():
            if type(inds) not in [list, range]:
                raise RuntimeError('Indices for field subsetting need to be a list')
        return self

    def __exit__(self, type, value, traceback):
        self.dataset.close()

    @property
    def read_lonlat(self):
        lon = getattr(self.dataset, self.dimensions['lon'])
        lat = getattr(self.dataset, self.dimensions['lat'])
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
        if len(lon.shape) > 1:  # if lon, lat are rectilinear but were stored in arrays
            xdim = lon_subset.shape[0]
            ydim = lat_subset.shape[1]
            if np.allclose(lon_subset[0, :], lon_subset[int(xdim/2), :]) and np.allclose(lat_subset[:, 0], lat_subset[:, int(ydim/2)]):
                lon_subset = lon_subset[0, :]
                lat_subset = lat_subset[:, 0]
        return lon_subset, lat_subset

    @property
    def read_depth(self):
        if 'depth' in self.dimensions:
            depth = getattr(self.dataset, self.dimensions['depth'])
            depthsize = depth.size if len(depth.shape) == 1 else depth.shape[-3]
            self.indices['depth'] = self.indices['depth'] if 'depth' in self.indices else range(depthsize)
            if len(depth.shape) == 1:
                return np.array(depth[self.indices['depth']])
            elif len(depth.shape) == 3:
                return np.array(depth[self.indices['depth'], self.indices['lat'], self.indices['lon']])
            elif len(depth.shape) == 4:
                raise NotImplementedError('Time varying depth data cannot be read in netcdf files yet')
                return np.array(depth[:, self.indices['depth'], self.indices['lat'], self.indices['lon']])
        else:
            self.indices['depth'] = [0]
            return np.zeros(1)

    @property
    def data(self):
        data = getattr(self.dataset, self.name)
        if len(data.shape) == 2:
            data = data[self.indices['lat'], self.indices['lon']]
        elif len(data.shape) == 3:
            if len(self.indices['depth']) > 1:
                data = data[self.indices['depth'], self.indices['lat'], self.indices['lon']]
            else:
                data = data[:, self.indices['lat'], self.indices['lon']]
        else:
            data = data[:, self.indices['depth'], self.indices['lat'], self.indices['lon']]

        if np.ma.is_masked(data):  # convert masked array to ndarray
            data = np.ma.filled(data, np.nan)
        return data

    @property
    def time(self):
        try:
            time_da = getattr(self.dataset, self.dimensions['time'])
            if self.dataset['decoded'] and 'Unit' not in time_da.attrs:
                time = np.array(time_da)
            else:
                if 'units' not in time_da.attrs and 'Unit' in time_da.attrs:
                    time_da.attrs['units'] = time_da.attrs['Unit']
                ds = xr.Dataset({self.dimensions['time']: time_da})
                ds = xr.decode_cf(ds)
                time = np.array(getattr(ds, self.dimensions['time']))
            if isinstance(time[0], datetime.datetime):
                raise NotImplementedError('Parcels currently only parses dates ranging from 1678 AD to 2262 AD, which are stored by xarray as np.datetime64. If you need a wider date range, please open an Issue on the parcels github page.')
            return time
        except:
            return np.array([None])

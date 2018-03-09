from parcels.loggers import logger
from scipy.interpolate import RegularGridInterpolator
from collections import Iterable
from py import path
import numpy as np
from ctypes import Structure, c_int, c_float, POINTER, pointer
import xarray as xr
from math import cos, pi
from datetime import timedelta
import math
from grid import (RectilinearZGrid, RectilinearSGrid, CurvilinearZGrid,
                  CurvilinearSGrid, CGrid, GridCode)


__all__ = ['Field', 'Geographic', 'GeographicPolar', 'GeographicSquare', 'GeographicPolarSquare']


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
    :param transpose: Transpose data to required (lon, lat) layout
    :param vmin: Minimum allowed value on the field. Data below this value are set to zero
    :param vmax: Maximum allowed value on the field. Data above this value are set to zero
    :param time_origin: Time origin (datetime object) of the time axis (only if grid is None)
    :param interp_method: Method for interpolation. Either 'linear' or 'nearest'
    :param allow_time_extrapolation: boolean whether to allow for extrapolation in time
           (i.e. beyond the last available time snapshot)
    :param time_periodic: boolean whether to loop periodically over the time component of the Field
           This flag overrides the allow_time_interpolation and sets it to False
    """

    unitconverters = {'U': GeographicPolar(), 'V': Geographic(),
                      'Kh_zonal': GeographicPolarSquare(),
                      'Kh_meridional': GeographicSquare()}

    def __init__(self, name, data, lon=None, lat=None, depth=None, time=None, grid=None, mesh='flat',
                 transpose=False, vmin=None, vmax=None, time_origin=0,
                 interp_method='linear', allow_time_extrapolation=None, time_periodic=False):
        self.name = name
        if self.name == 'UV':
            return
        self.data = data
        if grid:
            self.grid = grid
        else:
            self.grid = RectilinearZGrid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh)
        # self.lon, self.lat, self.depth and self.time are not used anymore in parcels.
        # self.grid should be used instead.
        # Those variables are still defined for backwards compatibility with users codes.
        self.lon = self.grid.lon
        self.lat = self.grid.lat
        self.depth = self.grid.depth
        self.time = self.grid.time
        if self.grid.mesh is 'flat' or (name not in self.unitconverters.keys()):
            self.units = UnitConverter()
        elif self.grid.mesh is 'spherical':
            self.units = self.unitconverters[name]
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
            self.data = np.transpose(self.data)

        if self.grid.lat_flipped:
            self.data = np.flip(self.data, axis=-2)

        if self.grid.tdim == 1:
            if len(self.data.shape) < 4:
                self.data = self.data.reshape(sum(((1,), self.data.shape), ()))
        if self.grid.zdim == 1:
            if len(self.data.shape) == 4:
                self.data = self.data.reshape(sum(((self.data.shape[0],), self.data.shape[2:]), ()))
        if len(self.data.shape) == 4:
            assert self.data.shape == (self.grid.tdim, self.grid.zdim, self.grid.ydim, self.grid.xdim), \
                                      ('Field %s expecting a data shape of a [ydim, xdim], [zdim, ydim, xdim], [tdim, ydim, xdim] or [tdim, zdim, ydim, xdim]. Flag transpose=True could help to reorder the data.')
        else:
            assert self.data.shape == (self.grid.tdim, self.grid.ydim, self.grid.xdim), \
                                      ('Field %s expecting a data shape of a [ydim, xdim], [zdim, ydim, xdim], [tdim, ydim, xdim] or [tdim, zdim, ydim, xdim]. Flag transpose=True could help to reorder the data.')

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
        :param dimensions: Dictionary mapping variable names for the relevant dimensions in the NetCDF file
        :param filenames: list of filenames to read for the field.
               Note that wildcards ('*') are also allowed
        :param indices: dictionary mapping indices for each dimension to read from file.
               This can be used for reading in only a subregion of the NetCDF file
        :param allow_time_extrapolation: boolean whether to allow for extrapolation in time
               (i.e. beyond the last available time snapshot
        :param mesh: String indicating the type of mesh coordinates and
               units used during velocity interpolation:

               1. spherical (default): Lat and lon in degree, with a
                  correction for zonal velocity U near the poles.
               2. flat: No conversion, lat/lon are assumed to be in m.
        """

        if not isinstance(filenames, Iterable) or isinstance(filenames, str):
            filenames = [filenames]
        with FileBuffer(filenames[0], dimensions, indices) as filebuffer:
            lon, lat = filebuffer.read_lonlat
            depth = filebuffer.read_depth
            if name in ['cosU', 'sinU', 'cosV', 'sinV']:
                warning = False
                try:
                    source = filebuffer.dataset.source
                    if source != 'parcels_compute_curvilinearGrid_rotationAngles':
                        warning = True
                except:
                    warning = True
                if warning:
                    logger.warning_once("You are defining a field name 'cosU', 'sinU', 'cosV' or 'sinV' which was not generated by Parcels. This field will be used to rotate UV velocity at interpolation")

        # Concatenate time variable to determine overall dimension
        # across multiple files
        timeslices = []
        for fname in filenames:
            with FileBuffer(fname, dimensions, indices) as filebuffer:
                timeslices.append(filebuffer.time)
        timeslices = np.array(timeslices)
        time = np.concatenate(timeslices)
        if isinstance(time[0], np.datetime64):
            time_origin = time[0]
            time = (time - time_origin) / np.timedelta64(1, 's')
        else:
            time_origin = 0
        assert(np.all((time[1:]-time[:-1]) > 0))

        # Pre-allocate data before reading files into buffer
        depthdim = depth.size if len(depth.shape) == 1 else depth.shape[-3]
        latdim = lat.size if len(lat.shape) == 1 else lat.shape[-2]
        londim = lon.size if len(lon.shape) == 1 else lon.shape[-1]
        data = np.empty((time.size, depthdim, latdim, londim), dtype=np.float32)
        ti = 0
        for tslice, fname in zip(timeslices, filenames):
            with FileBuffer(fname, dimensions, indices) as filebuffer:
                depthsize = depth.size if len(depth.shape) == 1 else depth.shape[-3]
                latsize = lat.size if len(lat.shape) == 1 else lat.shape[-2]
                lonsize = lon.size if len(lon.shape) == 1 else lon.shape[-1]
                filebuffer.indslat = indices['lat'] if 'lat' in indices else range(latsize)
                filebuffer.indslon = indices['lon'] if 'lon' in indices else range(lonsize)
                filebuffer.indsdepth = indices['depth'] if 'depth' in indices else range(depthsize)
                for inds in [filebuffer.indslat, filebuffer.indslon, filebuffer.indsdepth]:
                    if not isinstance(inds, list):
                        raise RuntimeError('Indices sur field subsetting need to be a list')
                if 'data' in dimensions:
                    # If Field.from_netcdf is called directly, it may not have a 'data' dimension
                    # In that case, assume that 'name' is the data dimension
                    filebuffer.name = dimensions['data']
                else:
                    filebuffer.name = name

                if len(filebuffer.dataset[filebuffer.name].shape) == 2:
                    data[ti:ti+len(tslice), 0, :, :] = filebuffer.data[:, :]
                elif len(filebuffer.dataset[filebuffer.name].shape) == 3:
                    data[ti:ti+len(tslice), 0, :, :] = filebuffer.data[:, :, :]
                else:
                    data[ti:ti+len(tslice), :, :, :] = filebuffer.data[:, :, :, :]
            ti += len(tslice)
        # Time indexing after the fact only
        if 'time' in indices:
            time = time[indices['time']]
            data = data[indices['time'], :, :, :]
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
        if name in ['cosU', 'sinU', 'cosV', 'sinV']:
            allow_time_extrapolation = True
        return cls(name, data, grid=grid,
                   allow_time_extrapolation=allow_time_extrapolation, **kwargs)

    def getUV(self, time, x, y, z):
        fieldset = self.fieldset
        U = fieldset.U.eval(time, x, y, z, False)
        V = fieldset.V.eval(time, x, y, z, False)
        if fieldset.U.grid.gtype in [GridCode.RectilinearZGrid, GridCode.RectilinearSGrid]:
            zonal = U
            meridional = V
        else:
            cosU = fieldset.cosU.eval(time, x, y, z, False)
            sinU = fieldset.sinU.eval(time, x, y, z, False)
            cosV = fieldset.cosV.eval(time, x, y, z, False)
            sinV = fieldset.sinV.eval(time, x, y, z, False)
            zonal = U * cosU - V * sinV
            meridional = U * sinU + V * cosV
        zonal = fieldset.U.units.to_target(zonal, x, y, z)
        meridional = fieldset.V.units.to_target(meridional, x, y, z)
        return (zonal, meridional)

    def __getitem__(self, key):
        if self.name == 'UV':
            return self.getUV(*key)
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

    def gradient(self):
        """Method to calculate horizontal gradients of Field.
                Returns two numpy arrays: the zonal and meridional gradients,
                on the same Grid as the original Field, using numpy.gradient() method"""
        if not self.grid.cell_edge_sizes:
            self.calc_cell_edge_sizes()
        dFdy = np.gradient(self.data, axis=-2) / self.grid.cell_edge_sizes['y']
        dFdx = np.gradient(self.data, axis=-1) / self.grid.cell_edge_sizes['x']
        return dFdx, dFdy

    def interpolator2D_scipy(self, ti, z_idx=None):
        """Provide a SciPy interpolator for spatial interpolation

        Note that the interpolator is configured to return NaN for
        out-of-bounds coordinates.
        """
        if z_idx is None:
            data = self.data[ti, :]
        else:
            data = self.data[ti, z_idx, :]
        return RegularGridInterpolator((self.grid.lat, self.grid.lon), data,
                                       bounds_error=False, fill_value=np.nan,
                                       method=self.interp_method)

    def interpolator3D_rectilinear_z(self, idx, z, y, x):
        """Scipy implementation of 3D interpolation, by first interpolating
        in horizontal, then in the vertical"""

        zdx = self.depth_index(z, y, x)
        f0 = self.interpolator2D_scipy(idx, z_idx=zdx)((y, x))
        f1 = self.interpolator2D_scipy(idx, z_idx=zdx + 1)((y, x))
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
                t0 = grid.time[ti]
                t1 = grid.time[ti + 1]
                depth_vector = dv2[0, :] + (dv2[1, :]-dv2[0, :]) * (time - t0) / (t1 - t0)
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

    def fix_i_index(self, xi, dim, sphere_mesh):
        if xi < 0:
            if sphere_mesh:
                xi = dim-2
            else:
                xi = 0
        if xi > dim-2:
            if sphere_mesh:
                xi = 0
            else:
                xi = dim-2
        return xi

    def search_indices_rectilinear(self, x, y, z, ti=-1, time=-1, search2D=False):
        grid = self.grid
        xi = yi = -1
        lon_index = grid.lon <= x

        if grid.mesh is not 'spherical':
            if x < grid.lon[0] or x > grid.lon[-1]:
                raise FieldSamplingError(x, y, z, field=self)
            lon_index = grid.lon <= x
            if lon_index.all():
                xi = len(grid.lon) - 2
            else:
                xi = lon_index.argmin() - 1 if lon_index.any() else 0
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

            lon_index = lon_fixed <= x
            if lon_index.all():
                xi = len(lon_fixed) - 2
            else:
                xi = lon_index.argmin() - 1 if lon_index.any() else 0
            xsi = (x-lon_fixed[xi]) / (lon_fixed[xi+1]-lon_fixed[xi])

        if y < grid.lat[0] or y > grid.lat[-1]:
            raise FieldSamplingError(x, y, z, field=self)
        lat_index = grid.lat <= y
        if lat_index.all():
            yi = len(grid.lat) - 2
        else:
            yi = lat_index.argmin() - 1 if lat_index.any() else 0

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
                px[1:] = np.where(px[1:] - x > 180, px[1:]-360, px[1:])
                px[1:] = np.where(-px[1:] + x > 180, px[1:]+360, px[1:])
            py = np.array([grid.lat[yi, xi], grid.lat[yi, xi+1], grid.lat[yi+1, xi+1], grid.lat[yi+1, xi]])
            a = np.dot(invA, px)
            b = np.dot(invA, py)

            aa = a[3]*b[2] - a[2]*b[3]
            if abs(aa) < 1e-12:  # Rectilinear cell, or quasi
                xsi = ((x-px[0]) / (px[1]-px[0])
                       + (x-px[3]) / (px[2]-px[3])) * .5
                eta = ((y-grid.lat[yi, xi]) / (grid.lat[yi+1, xi]-grid.lat[yi, xi])
                       + (y-grid.lat[yi, xi+1]) / (grid.lat[yi+1, xi+1]-grid.lat[yi, xi+1])) * .5
            else:
                bb = a[3]*b[0] - a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + x*b[3] - y*a[3]
                cc = a[1]*b[0] - a[0]*b[1] + x*b[1] - y*a[1]
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
            xi = self.fix_i_index(xi, grid.xdim, grid.mesh == 'spherical')
            yi = self.fix_i_index(yi, grid.ydim, False)
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
        (xsi, eta, trash, xi, yi, trash) = self.search_indices(x, y, z, xi, yi)
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
            raise RuntimeError(self.interp_method+"is not implemented for 3D grids")

    def interpolator3D(self, ti, z, y, x, time):
        xi = int(self.grid.xdim / 2)
        yi = int(self.grid.ydim / 2)
        (xsi, eta, zeta, xi, yi, zi) = self.search_indices(x, y, z, xi, yi, ti, time)
        if self.interp_method is 'nearest':
            xii = xi if xsi <= .5 else xi+1
            yii = yi if eta <= .5 else yi+1
            zii = zi if zeta <= .5 else zi+1
            return self.data[ti, zii, yii, xii]
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
            raise RuntimeError(self.interp_method+"is not implemented for 3D grids")

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

        if self.grid.gtype is GridCode.RectilinearZGrid:  # The only case where we use scipy interpolation
            if self.grid.zdim == 1:
                val = self.interpolator2D_scipy(ti)((y, x))
            else:
                val = self.interpolator3D_rectilinear_z(ti, z, y, x)
        elif self.grid.gtype in [GridCode.RectilinearSGrid, GridCode.CurvilinearZGrid, GridCode.CurvilinearSGrid]:
            if self.grid.zdim == 1:
                val = self.interpolator2D(ti, z, y, x)
            else:
                val = self.interpolator3D(ti, z, y, x, time)
        else:
            raise RuntimeError("Only RectilinearZGrid, RectilinearSGrid and CRectilinearGrid grids are currently implemented")
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
            value = self.spatial_interpolation(ti, z, y, x, self.grid.time[ti-1])

        if applyConversion:
            return self.units.to_target(value, x, y, z)
        else:
            return value

    def ccode_evalUV(self, varU, varV, t, x, y, z):
        # Casting interp_methd to int as easier to pass on in C-code

        gridset = self.fieldset.gridset
        uiGrid = gridset.grids.index(self.fieldset.U.grid)
        viGrid = gridset.grids.index(self.fieldset.V.grid)
        if self.fieldset.U.grid.gtype in [GridCode.RectilinearZGrid, GridCode.RectilinearSGrid]:
            return "temporal_interpolationUV(%s, %s, %s, %s, U, V, particle->CGridIndexSet, %s, %s, &%s, &%s, %s)" \
                % (x, y, z, t,
                   uiGrid, viGrid, varU, varV, self.fieldset.U.interp_method.upper())
        else:
            cosuiGrid = gridset.grids.index(self.fieldset.cosU.grid)
            sinuiGrid = gridset.grids.index(self.fieldset.sinU.grid)
            cosviGrid = gridset.grids.index(self.fieldset.cosV.grid)
            sinviGrid = gridset.grids.index(self.fieldset.sinV.grid)
            return "temporal_interpolationUVrotation(%s, %s, %s, %s, U, V, cosU, sinU, cosV, sinV, particle->CGridIndexSet, %s, %s, %s, %s, %s, %s, &%s, &%s, %s)" \
                % (x, y, z, t,
                   uiGrid, viGrid, cosuiGrid, sinuiGrid, cosviGrid, sinviGrid,
                   varU, varV, self.fieldset.U.interp_method.upper())

    def ccode_eval(self, var, t, x, y, z):
        # Casting interp_methd to int as easier to pass on in C-code
        gridset = self.fieldset.gridset
        iGrid = gridset.grids.index(self.grid)
        return "temporal_interpolation(%s, %s, %s, %s, %s, %s, %s, &%s, %s)" \
            % (x, y, z, t, self.name, "particle->CGridIndexSet", iGrid, var,
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
        cstruct = CField(self.grid.xdim, self.grid.ydim, self.grid.zdim,
                         self.grid.tdim, allow_time_extrapolation, time_periodic,
                         self.data.ctypes.data_as(POINTER(POINTER(c_float))),
                         pointer(self.grid.ctypes_struct))
        return cstruct

    def show(self, with_particles=False, animation=False, show_time=None, vmin=None, vmax=None):
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
            show_time = self.grid.time[0] if show_time is None else show_time
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
        Before adding a periodic halo to the Field, it has to be added to the Grid on which the Field depends

        :param zonal: Create a halo in zonal direction (boolean)
        :param meridional: Create a halo in meridional direction (boolean)
        :param halosize: size of the halo (in grid points). Default is 5 grid points
        """
        if self.name == 'UV':
            return
        if zonal:
            if len(self.data.shape) is 3:
                self.data = np.concatenate((self.data[:, :, -halosize:], self.data,
                                            self.data[:, :, 0:halosize]), axis=len(self.data.shape)-1)
                assert self.data.shape[2] == self.grid.xdim
            else:
                self.data = np.concatenate((self.data[:, :, :, -halosize:], self.data,
                                            self.data[:, :, :, 0:halosize]), axis=len(self.data.shape) - 1)
                assert self.data.shape[3] == self.grid.xdim
            self.lon = self.grid.lon
            self.lat = self.grid.lat
        if meridional:
            if len(self.data.shape) is 3:
                self.data = np.concatenate((self.data[:, -halosize:, :], self.data,
                                            self.data[:, 0:halosize, :]), axis=len(self.data.shape)-2)
                assert self.data.shape[1] == self.grid.ydim
            else:
                self.data = np.concatenate((self.data[:, :, -halosize:, :], self.data,
                                            self.data[:, :, 0:halosize, :]), axis=len(self.data.shape) - 2)
                assert self.data.shape[2] == self.grid.ydim
            self.lat = self.grid.lat

    def write(self, filename, varname=None):
        """Write a :class:`Field` to a netcdf file

        :param filename: Basename of the file
        :param varname: Name of the field, to be appended to the filename"""
        if self.name == 'UV':
            return
        filepath = str(path.local('%s%s.nc' % (filename, self.name)))
        if varname is None:
            varname = self.name
        # Derive name of 'depth' variable for NEMO convention
        vname_depth = 'depth%s' % self.name.lower()

        # Create DataArray objects for file I/O
        t, d, x, y = (self.grid.time.size, self.grid.depth.size,
                      self.grid.lon.size, self.grid.lat.size)
        nav_lon = xr.DataArray(self.grid.lon + np.zeros((y, x), dtype=np.float32),
                               coords=[('y', self.grid.lat), ('x', self.grid.lon)])
        nav_lat = xr.DataArray(self.grid.lat.reshape(y, 1) + np.zeros(x, dtype=np.float32),
                               coords=[('y', self.grid.lat), ('x', self.grid.lon)])
        vardata = xr.DataArray(self.data.reshape((t, d, y, x)),
                               coords=[('time_counter', self.grid.time),
                                       (vname_depth, self.grid.depth),
                                       ('y', self.grid.lat), ('x', self.grid.lon)])
        # Create xarray Dataset and output to netCDF format
        dset = xr.Dataset({varname: vardata}, coords={'nav_lon': nav_lon,
                                                      'nav_lat': nav_lat,
                                                      vname_depth: self.grid.depth})
        dset.to_netcdf(filepath)

    def advancetime(self, field_new, advanceForward):
        if advanceForward == 1:  # forward in time, so appending at end
            self.data = np.concatenate((self.data[1:, :, :], field_new.data[:, :, :]), 0)
            self.time = self.grid.time
        else:  # backward in time, so prepending at start
            self.data = np.concatenate((field_new.data[:, :, :], self.data[:-1, :, :]), 0)
            self.time = self.grid.time


class FileBuffer(object):
    """ Class that encapsulates and manages deferred access to file data. """

    def __init__(self, filename, dimensions, indices):
        self.filename = filename
        self.dimensions = dimensions  # Dict with dimension keyes for file data
        self.indices = indices
        self.dataset = None

    def __enter__(self):
        self.dataset = xr.open_dataset(str(self.filename))
        lon = getattr(self.dataset, self.dimensions['lon'])
        lat = getattr(self.dataset, self.dimensions['lat'])
        latsize = lat.size if len(lat.shape) == 1 else lat.shape[-2]
        lonsize = lon.size if len(lon.shape) == 1 else lon.shape[-1]
        self.indslon = self.indices['lon'] if 'lon' in self.indices else range(lonsize)
        self.indslat = self.indices['lat'] if 'lat' in self.indices else range(latsize)
        if 'depth' in self.dimensions:
            depth = getattr(self.dataset, self.dimensions['depth'])
            depthsize = depth.size if len(depth.shape) == 1 else depth.shape[-3]
            self.indsdepth = self.indices['depth'] if 'depth' in self.indices else range(depthsize)
        return self

    def __exit__(self, type, value, traceback):
        self.dataset.close()

    @property
    def read_lonlat(self):
        lon = getattr(self.dataset, self.dimensions['lon'])
        lat = getattr(self.dataset, self.dimensions['lat'])
        if len(lon.shape) == 1:
            lon_subset = np.array(lon[self.indslon])
            lat_subset = np.array(lat[self.indslat])
        elif len(lon.shape) == 2:
            lon_subset = np.array(lon[self.indslat, self.indslon])
            lat_subset = np.array(lat[self.indslat, self.indslon])
        elif len(lon.shape) == 3:  # some lon, lat have a time dimension 1
            lon_subset = np.array(lon[0, self.indslat, self.indslon])
            lat_subset = np.array(lat[0, self.indslat, self.indslon])
        if len(lon.shape) > 1:  # if lon, lat are rectilinear but were stored in arrays
            londim = lon_subset.shape[0]
            latdim = lat_subset.shape[1]
            if np.allclose(lon_subset[0, :], lon_subset[int(londim/2), :]) and np.allclose(lat_subset[:, 0], lat_subset[:, int(latdim/2)]):
                lon_subset = lon_subset[0, :]
                lat_subset = lat_subset[:, 0]
        return lon_subset, lat_subset

    @property
    def read_depth(self):
        if 'depth' in self.dimensions:
            depth = getattr(self.dataset, self.dimensions['depth'])
            if len(depth.shape) == 1:
                return np.array(depth[self.indsdepth])
            elif len(depth.shape) == 3:
                return np.array(depth[self.indsdepth, self.indslat, self.indslon])
            elif len(depth.shape) == 4:
                raise NotImplementedError('Time varying depth data cannot be read in netcdf files yet')
                return np.array(depth[:, self.indsdepth, self.indslat, self.indslon])
        else:
            return np.zeros(1)

    @property
    def data(self):
        data = getattr(self.dataset, self.name)
        if len(data.shape) == 2:
            data = data[self.indslat, self.indslon]
        elif len(data.shape) == 3:
            data = data[:, self.indslat, self.indslon]
        else:
            data = data[:, self.indsdepth, self.indslat, self.indslon]

        if np.ma.is_masked(data):  # convert masked array to ndarray
            data = np.ma.filled(data, np.nan)
        return data

    @property
    def time(self):
        try:
            time = getattr(self.dataset, self.dimensions['time'])
            if isinstance(time[0], np.datetime64) or 'Unit' not in time.attrs:
                return np.array(time)
            time.attrs['units'] = time.attrs['Unit']
            ds = xr.decode_cf(self.dataset)
            time = getattr(ds, self.dimensions['time'])
            time_arr = np.array(time)
            return time_arr
        except:
            return np.array([None])

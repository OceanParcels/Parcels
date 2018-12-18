from parcels.tools.loggers import logger
from parcels.tools.converters import unitconverters_map, UnitConverter, Geographic, GeographicPolar
from parcels.tools.converters import TimeConverter
from parcels.tools.error import FieldSamplingError, TimeExtrapolationError
import parcels.tools.interpolation_utils as i_u
from collections import Iterable
from py import path
import numpy as np
from ctypes import Structure, c_int, c_float, POINTER, pointer
import xarray as xr
import datetime
import math
from .grid import (RectilinearZGrid, RectilinearSGrid, CurvilinearZGrid,
                   CurvilinearSGrid, CGrid, GridCode)


__all__ = ['Field', 'VectorField', 'SummedField', 'SummedVectorField', 'NestedField']


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
    :param fieldtype: Type of Field to be used for UnitConverter when using SummedFields
           (either 'U', 'V', 'Kh_zonal', 'Kh_Meridional' or None)
    :param transpose: Transpose data to required (lon, lat) layout
    :param vmin: Minimum allowed value on the field. Data below this value are set to zero
    :param vmax: Maximum allowed value on the field. Data above this value are set to zero
    :param time_origin: Time origin (TimeConverter object) of the time axis (only if grid is None)
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
        time_origin = TimeConverter(0) if time_origin is None else time_origin
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
        self.fieldtype = self.name if fieldtype is None else fieldtype
        if self.grid.mesh == 'flat' or (self.fieldtype not in unitconverters_map.keys()):
            self.units = UnitConverter()
        elif self.grid.mesh == 'spherical':
            self.units = unitconverters_map[self.fieldtype]
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
        self.netcdf_engine = kwargs.pop('netcdf_engine', 'netcdf4')
        self.loaded_time_indices = []

    @classmethod
    def from_netcdf(cls, filenames, variable, dimensions, indices=None, grid=None,
                    mesh='spherical', allow_time_extrapolation=None, time_periodic=False,
                    full_load=False, **kwargs):
        """Create field from netCDF file

        :param filenames: list of filenames to read for the field.
               Note that wildcards ('*') are also allowed
               filenames can be a list [files]
               or a dictionary {dim:[files]} (if lon, lat, depth and/or data not stored in same files as data)
               time values are in filenames[data]
        :param variable: Name of the field to create. Note that this has to be a string
        :param dimensions: Dictionary mapping variable names for the relevant dimensions in the NetCDF file
        :param indices: dictionary mapping indices for each dimension to read from file.
               This can be used for reading in only a subregion of the NetCDF file.
               Note that negative indices are not allowed.
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
        :param netcdf_engine: engine to use for netcdf reading in xarray. Default is 'netcdf',
               but in cases where this doesn't work, setting netcdf_engine='scipy' could help
        """
        if isinstance(variable, xr.core.dataarray.DataArray):
            lonlat_filename = variable
            depth_filename = variable
            data_filenames = variable
            netcdf_engine = 'xarray'
        else:
            if not isinstance(filenames, Iterable) or isinstance(filenames, str):
                filenames = [filenames]

            data_filenames = filenames['data'] if type(filenames) is dict else filenames
            if type(filenames) == dict:
                for k in filenames.keys():
                    assert k in ['lon', 'lat', 'depth', 'data'], \
                        'filename dimension keys must be lon, lat, depth or data'
                assert len(filenames['lon']) == 1
                if filenames['lon'] != filenames['lat']:
                    raise NotImplementedError('longitude and latitude dimensions are currently processed together from one single file')
                lonlat_filename = filenames['lon'][0]
                if 'depth' in dimensions:
                    if not isinstance(filenames['depth'], Iterable) or isinstance(filenames['depth'], str):
                        filenames['depth'] = [filenames['depth']]
                    if len(filenames['depth']) != 1:
                        raise NotImplementedError('Vertically adaptive meshes not implemented for from_netcdf()')
                    depth_filename = filenames['depth'][0]
            else:
                lonlat_filename = filenames[0]
                depth_filename = filenames[0]
            netcdf_engine = kwargs.pop('netcdf_engine', 'netcdf4')

        indices = {} if indices is None else indices.copy()
        for ind in indices.values():
            assert np.min(ind) >= 0, \
                ('Negative indices are currently not allowed in Parcels. '
                 + 'This is related to the non-increasing dimension it could generate '
                 + 'if the domain goes from lon[-4] to lon[6] for example. '
                 + 'Please raise an issue on https://github.com/OceanParcels/parcels/issues '
                 + 'if you would need such feature implemented.')

        with NetcdfFileBuffer(lonlat_filename, dimensions, indices, netcdf_engine) as filebuffer:
            lon, lat = filebuffer.read_lonlat
            indices = filebuffer.indices
            # Check if parcels_mesh has been explicitly set in file
            if 'parcels_mesh' in filebuffer.dataset.attrs:
                mesh = filebuffer.dataset.attrs['parcels_mesh']

        if 'depth' in dimensions:
            with NetcdfFileBuffer(depth_filename, dimensions, indices, netcdf_engine) as filebuffer:
                depth = filebuffer.read_depth
        else:
            indices['depth'] = [0]
            depth = np.zeros(1)

        if len(data_filenames) > 1 and 'time' not in dimensions:
            raise RuntimeError('Multiple files given but no time dimension specified')

        if grid is None:
            # Concatenate time variable to determine overall dimension
            # across multiple files
            if netcdf_engine == 'xarray':
                with NetcdfFileBuffer(data_filenames, dimensions, indices, netcdf_engine) as filebuffer:
                    time = filebuffer.time
                    timeslices = time if isinstance(time, (list, np.ndarray)) else [time]
                    dataFiles = data_filenames if isinstance(data_filenames, (list, np.ndarray)) else [data_filenames] * len(time)
            else:
                timeslices = []
                dataFiles = []
                for fname in data_filenames:
                    with NetcdfFileBuffer(fname, dimensions, indices, netcdf_engine) as filebuffer:
                        ftime = filebuffer.time
                        timeslices.append(ftime)
                        dataFiles.append([fname] * len(ftime))
                timeslices = np.array(timeslices)
                time = np.concatenate(timeslices)
                dataFiles = np.concatenate(np.array(dataFiles))
            if time.size == 1 and time[0] is None:
                time[0] = 0
            time_origin = TimeConverter(time[0])
            time = time_origin.reltime(time)
            assert(np.all((time[1:]-time[:-1]) > 0))

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
            for tslice, fname in zip(grid.timeslices, data_filenames):
                with NetcdfFileBuffer(fname, dimensions, indices, netcdf_engine) as filebuffer:
                    # If Field.from_netcdf is called directly, it may not have a 'data' dimension
                    # In that case, assume that 'name' is the data dimension
                    if netcdf_engine == 'xarray':
                        tslice = [tslice]
                    else:
                        filebuffer.name = filebuffer.parse_name(dimensions, variable)

                    if len(filebuffer.data.shape) == 2:
                        data[ti:ti+len(tslice), 0, :, :] = filebuffer.data
                    elif len(filebuffer.data.shape) == 3:
                        if len(filebuffer.indices['depth']) > 1:
                            data[ti:ti+len(tslice), :, :, :] = filebuffer.data
                        else:
                            data[ti:ti+len(tslice), 0, :, :] = filebuffer.data
                    else:
                        data[ti:ti+len(tslice), :, :, :] = filebuffer.data
                ti += len(tslice)
        else:
            grid.defer_load = True
            grid.ti = -1
            data = DeferredArray()

        if allow_time_extrapolation is None:
            allow_time_extrapolation = False if 'time' in dimensions else True

        kwargs['dimensions'] = dimensions.copy()
        kwargs['indices'] = indices
        kwargs['time_periodic'] = time_periodic
        kwargs['netcdf_engine'] = netcdf_engine

        variable = kwargs['var_name'] if netcdf_engine == 'xarray' else variable
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

    def gradient(self, update=False, tindex=None):
        """Method to calculate horizontal gradients of Field.
                Returns two Fields: the zonal and meridional gradients,
                on the same Grid as the original Field, using numpy.gradient() method
                Names of these grids are dNAME_dx and dNAME_dy, where NAME is the name
                of the original Field"""
        tindex = range(self.grid.tdim) if tindex is None else tindex
        if not self.grid.cell_edge_sizes:
            self.calc_cell_edge_sizes()
        if self.grid.defer_load and isinstance(self.data, DeferredArray):
            (dFdx, dFdy) = (None, None)
        else:
            dFdy = np.gradient(self.data[tindex, :], axis=-2) / self.grid.cell_edge_sizes['y']
            dFdx = np.gradient(self.data[tindex, :], axis=-1) / self.grid.cell_edge_sizes['x']
        if update:
            if self.gradientx.data is None:
                self.gradientx.data = np.zeros_like(self.data)
                self.gradienty.data = np.zeros_like(self.data)
            self.gradientx.data[tindex, :] = dFdx
            self.gradienty.data[tindex, :] = dFdy
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

        if not grid.zonal_periodic:
            if x < grid.lonlat_minmax[0] or x > grid.lonlat_minmax[1]:
                raise FieldSamplingError(x, y, z, field=self)
        if y < grid.lonlat_minmax[2] or y > grid.lonlat_minmax[3]:
            raise FieldSamplingError(x, y, z, field=self)

        if grid.mesh is not 'spherical':
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
        if not grid.zonal_periodic:
            if x < grid.lonlat_minmax[0] or x > grid.lonlat_minmax[1]:
                if grid.lon[0, 0] < grid.lon[0, -1]:
                    raise FieldSamplingError(x, y, z, field=self)
                elif x < grid.lon[0, 0] and x > grid.lon[0, -1]:  # This prevents from crashing in [160, -160]
                    raise FieldSamplingError(x, y, z, field=self)
        if y < grid.lonlat_minmax[2] or y > grid.lonlat_minmax[3]:
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
            if abs(a[1]+a[3]*eta) < 1e-12:  # this happens when recti cell rotated of 90deg
                xsi = ((y-py[0])/(py[1]-py[0]) + (y-py[3])/(py[2]-py[3])) * .5
            else:
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
        elif self.interp_method is 'cgrid_tracer':
            return self.data[ti, yi+1, xi+1]
        else:
            raise RuntimeError(self.interp_method+" is not implemented for 2D grids")

    def interpolator3D(self, ti, z, y, x, time):
        xi = int(self.grid.xdim / 2) - 1
        yi = int(self.grid.ydim / 2) - 1
        (xsi, eta, zeta, xi, yi, zi) = self.search_indices(x, y, z, xi, yi, ti, time)
        if self.interp_method is 'nearest':
            xii = xi if xsi <= .5 else xi+1
            yii = yi if eta <= .5 else yi+1
            zii = zi if zeta <= .5 else zi+1
            return self.data[ti, zii, yii, xii]
        elif self.interp_method is 'cgrid_velocity':
            # evaluating W velocity in c_grid
            f0 = self.data[ti, zi, yi+1, xi+1]
            f1 = self.data[ti, zi+1, yi+1, xi+1]
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
        elif self.interp_method is 'cgrid_tracer':
            return self.data[ti, zi, yi+1, xi+1]
        else:
            raise RuntimeError(self.interp_method+" is not implemented for 3D grids")

    def temporal_interpolate_fullfield(self, ti, time):
        """Calculate the data of a field between two snapshots,
        using linear interpolation

        :param ti: Index in time array associated with time (via :func:`time_index`)
        :param time: Time to interpolate to

        :rtype: Linearly interpolated field"""
        t0 = self.grid.time[ti]
        if time == t0:
            return self.data[ti, :]
        elif ti+1 >= len(self.grid.time):
            raise TimeExtrapolationError(time, field=self, msg='show_time')
        else:
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

    def eval(self, time, z, y, x, applyConversion=True):
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

    def ccode_eval(self, var, t, z, y, x):
        # Casting interp_methd to int as easier to pass on in C-code
        return "temporal_interpolation(%s, %s, %s, %s, %s, particle->cxi, particle->cyi, particle->czi, particle->cti, &%s, %s)" \
            % (x, y, z, t, self.name, var, self.interp_method.upper())

    def ccode_convert(self, _, z, y, x):
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

    def show(self, animation=False, show_time=None, domain=None, depth_level=0, projection=None, land=True,
             vmin=None, vmax=None, savefile=None, **kwargs):
        """Method to 'show' a Parcels Field

        :param animation: Boolean whether result is a single plot, or an animation
        :param show_time: Time at which to show the Field (only in single-plot mode)
        :param domain: Four-vector (latN, latS, lonE, lonW) defining domain to show
        :param depth_level: depth level to be plotted (default 0)
        :param projection: type of cartopy projection to use (default PlateCarree)
        :param land: Boolean whether to show land. This is ignored for flat meshes
        :param vmin: minimum colour scale (only in single-plot mode)
        :param vmax: maximum colour scale (only in single-plot mode)
        :param savefile: Name of a file to save the plot to
        """
        from parcels.plotting import plotfield
        plt, _, _, _ = plotfield(self, animation=animation, show_time=show_time, domain=domain, depth_level=depth_level,
                                 projection=projection, land=land, vmin=vmin, vmax=vmax, savefile=savefile, **kwargs)
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

        attrs = {'units': 'seconds since ' + str(self.grid.time_origin)} if self.grid.time_origin.calendar else {}
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
        with NetcdfFileBuffer(self.dataFiles[g.ti+tindex], self.dimensions, self.indices, self.netcdf_engine) as filebuffer:
            time_data = filebuffer.time
            time_data = g.time_origin.reltime(time_data)
            filebuffer.ti = (time_data <= g.time[tindex]).argmin() - 1
            if self.netcdf_engine != 'xarray':
                filebuffer.name = filebuffer.parse_name(self.dimensions, self.name)
            if len(filebuffer.data.shape) == 2:
                data[tindex, 0, :, :] = filebuffer.data
            elif len(filebuffer.data.shape) == 3:
                if g.zdim > 1:
                    data[tindex, :, :, :] = filebuffer.data
                else:
                    data[tindex, 0, :, :] = filebuffer.data
            else:
                data[tindex, :, :, :] = filebuffer.data

    def __add__(self, field):
        if isinstance(self, Field) and isinstance(field, Field):
            return SummedField([self, field])
        elif isinstance(field, SummedField):
            field.insert(0, self)
            return field


class SummedField(list):
    """Class SummedField is a list of Fields over which Field interpolation
    is summed. This can e.g. be used when combining multiple flow fields,
    where the total flow is the sum of all the individual flows.
    Note that the individual Fields can be on different Grids.
    Also note that, since SummedFields are lists, the individual Fields can
    still be queried through their list index (e.g. SummedField[1]).
    """
    def eval(self, time, z, y, x, applyConversion=True):
        tmp = 0
        for fld in self:
            tmp += fld.eval(time, z, y, x, applyConversion)
        return tmp

    def __getitem__(self, key):
        if isinstance(key, int):
            return list.__getitem__(self, key)
        else:
            return self.eval(*key)

    def __add__(self, field):
        if isinstance(field, Field):
            self.append(field)
        elif isinstance(field, SummedField):
            for fld in field:
                self.append(fld)
        return self


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
        if self.U.interp_method == 'cgrid_velocity':
            assert self.V.interp_method == 'cgrid_velocity'
            assert self.U.grid is self.V.grid
            if W:
                assert self.W.interp_method == 'cgrid_velocity'

    def dist(self, lon1, lon2, lat1, lat2, mesh, lat):
        if mesh == 'spherical':
            rad = np.pi/180.
            deg2m = 1852 * 60.
            return np.sqrt(((lon2-lon1)*deg2m*math.cos(rad * lat))**2 + ((lat2-lat1)*deg2m)**2)
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
        xi = int(grid.xdim / 2) - 1
        yi = int(grid.ydim / 2) - 1
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
        c1 = self.dist(px[0], px[1], py[0], py[1], grid.mesh, np.dot(i_u.phi2D_lin(xsi, 0.), py))
        c2 = self.dist(px[1], px[2], py[1], py[2], grid.mesh, np.dot(i_u.phi2D_lin(1., eta), py))
        c3 = self.dist(px[2], px[3], py[2], py[3], grid.mesh, np.dot(i_u.phi2D_lin(xsi, 1.), py))
        c4 = self.dist(px[3], px[0], py[3], py[0], grid.mesh, np.dot(i_u.phi2D_lin(0., eta), py))
        if grid.zdim == 1:
            U0 = self.U.data[ti, yi+1, xi] * c4
            U1 = self.U.data[ti, yi+1, xi+1] * c2
            V0 = self.V.data[ti, yi, xi+1] * c1
            V1 = self.V.data[ti, yi+1, xi+1] * c3
        else:
            U0 = self.U.data[ti, zi, yi+1, xi] * c4
            U1 = self.U.data[ti, zi, yi+1, xi+1] * c2
            V0 = self.V.data[ti, zi, yi, xi+1] * c1
            V1 = self.V.data[ti, zi, yi+1, xi+1] * c3
        U = (1-xsi) * U0 + xsi * U1
        V = (1-eta) * V0 + eta * V1
        rad = np.pi/180.
        deg2m = 1852 * 60.
        meshJac = (deg2m * deg2m * math.cos(rad * y)) if grid.mesh == 'spherical' else 1
        jac = self.jacobian(xsi, eta, px, py) * meshJac

        u = ((-(1-eta) * U - (1-xsi) * V) * px[0]
             + ((1-eta) * U - xsi * V) * px[1]
             + (eta * U + xsi * V) * px[2]
             + (-eta * U + (1-xsi) * V) * px[3]) / jac
        v = ((-(1-eta) * U - (1-xsi) * V) * py[0]
             + ((1-eta) * U - xsi * V) * py[1]
             + (eta * U + xsi * V) * py[2]
             + (-eta * U + (1-xsi) * V) * py[3]) / jac
        return (u, v)

    def spatial_c_grid_interpolation3D_full(self, ti, z, y, x, time):
        grid = self.U.grid
        xi = int(grid.xdim / 2) - 1
        yi = int(grid.ydim / 2) - 1
        (xsi, eta, zet, xi, yi, zi) = self.U.search_indices(x, y, z, xi, yi, ti, time)

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

        px = np.concatenate((px, px))
        py = np.concatenate((py, py))
        if grid.z4d:
            pz = np.array([grid.depth[0, zi, yi, xi], grid.depth[0, zi, yi, xi+1], grid.depth[0, zi, yi+1, xi+1], grid.depth[0, zi, yi+1, xi],
                           grid.depth[0, zi+1, yi, xi], grid.depth[0, zi+1, yi, xi+1], grid.depth[0, zi+1, yi+1, xi+1], grid.depth[0, zi+1, yi+1, xi]])
        else:
            pz = np.array([grid.depth[zi, yi, xi], grid.depth[zi, yi, xi+1], grid.depth[zi, yi+1, xi+1], grid.depth[zi, yi+1, xi],
                           grid.depth[zi+1, yi, xi], grid.depth[zi+1, yi, xi+1], grid.depth[zi+1, yi+1, xi+1], grid.depth[zi+1, yi+1, xi]])

        u0 = self.U.data[ti, zi, yi+1, xi]
        u1 = self.U.data[ti, zi, yi+1, xi+1]
        v0 = self.V.data[ti, zi, yi, xi+1]
        v1 = self.V.data[ti, zi, yi+1, xi+1]
        w0 = self.W.data[ti, zi, yi+1, xi+1]
        w1 = self.W.data[ti, zi+1, yi+1, xi+1]

        U0 = u0 * i_u.jacobian3D_lin_face(px, py, pz, 0, eta, zet, 'zonal', grid.mesh)
        U1 = u1 * i_u.jacobian3D_lin_face(px, py, pz, 1, eta, zet, 'zonal', grid.mesh)
        V0 = v0 * i_u.jacobian3D_lin_face(px, py, pz, xsi, 0, zet, 'meridional', grid.mesh)
        V1 = v1 * i_u.jacobian3D_lin_face(px, py, pz, xsi, 1, zet, 'meridional', grid.mesh)
        W0 = w0 * i_u.jacobian3D_lin_face(px, py, pz, xsi, eta, 0, 'vertical', grid.mesh)
        W1 = w1 * i_u.jacobian3D_lin_face(px, py, pz, xsi, eta, 1, 'vertical', grid.mesh)

        # Computing fluxes in half left hexahedron -> flux_u05
        xx = [px[0], (px[0]+px[1])/2, (px[2]+px[3])/2, px[3], px[4], (px[4]+px[5])/2, (px[6]+px[7])/2, px[7]]
        yy = [py[0], (py[0]+py[1])/2, (py[2]+py[3])/2, py[3], py[4], (py[4]+py[5])/2, (py[6]+py[7])/2, py[7]]
        zz = [pz[0], (pz[0]+pz[1])/2, (pz[2]+pz[3])/2, pz[3], pz[4], (pz[4]+pz[5])/2, (pz[6]+pz[7])/2, pz[7]]
        flux_u0 = u0 * i_u.jacobian3D_lin_face(xx, yy, zz, 0, .5, .5, 'zonal', grid.mesh)
        flux_v0_halfx = v0 * i_u.jacobian3D_lin_face(xx, yy, zz, .5, 0, .5, 'meridional', grid.mesh)
        flux_v1_halfx = v1 * i_u.jacobian3D_lin_face(xx, yy, zz, .5, 1, .5, 'meridional', grid.mesh)
        flux_w0_halfx = w0 * i_u.jacobian3D_lin_face(xx, yy, zz, .5, .5, 0, 'vertical', grid.mesh)
        flux_w1_halfx = w1 * i_u.jacobian3D_lin_face(xx, yy, zz, .5, .5, 1, 'vertical', grid.mesh)
        flux_u05 = flux_u0 + flux_v0_halfx - flux_v1_halfx + flux_w0_halfx - flux_w1_halfx

        # Computing fluxes in half front hexahedron -> flux_v05
        xx = [px[0], px[1], (px[1]+px[2])/2, (px[0]+px[3])/2, px[4], px[5], (px[5]+px[6])/2, (px[4]+px[7])/2]
        yy = [py[0], py[1], (py[1]+py[2])/2, (py[0]+py[3])/2, py[4], py[5], (py[5]+py[6])/2, (py[4]+py[7])/2]
        zz = [pz[0], pz[1], (pz[1]+pz[2])/2, (pz[0]+pz[3])/2, pz[4], pz[5], (pz[5]+pz[6])/2, (pz[4]+pz[7])/2]
        flux_u0_halfy = u0 * i_u.jacobian3D_lin_face(xx, yy, zz, 0, .5, .5, 'zonal', grid.mesh)
        flux_u1_halfy = u1 * i_u.jacobian3D_lin_face(xx, yy, zz, 1, .5, .5, 'zonal', grid.mesh)
        flux_v0 = v0 * i_u.jacobian3D_lin_face(xx, yy, zz, .5, 0, .5, 'meridional', grid.mesh)
        flux_w0_halfy = w0 * i_u.jacobian3D_lin_face(xx, yy, zz, .5, .5, 0, 'vertical', grid.mesh)
        flux_w1_halfy = w1 * i_u.jacobian3D_lin_face(xx, yy, zz, .5, .5, 1, 'vertical', grid.mesh)
        flux_v05 = flux_u0_halfy - flux_u1_halfy + flux_v0 + flux_w0_halfy - flux_w1_halfy

        # Computing fluxes in half lower hexahedron -> flux_w05
        xx = [px[0], px[1], px[2], px[3], (px[0]+px[4])/2, (px[1]+px[5])/2, (px[2]+px[6])/2, (px[3]+px[7])/2]
        yy = [py[0], py[1], py[2], py[3], (py[0]+py[4])/2, (py[1]+py[5])/2, (py[2]+py[6])/2, (py[3]+py[7])/2]
        zz = [pz[0], pz[1], pz[2], pz[3], (pz[0]+pz[4])/2, (pz[1]+pz[5])/2, (pz[2]+pz[6])/2, (pz[3]+pz[7])/2]
        flux_u0_halfz = u0 * i_u.jacobian3D_lin_face(xx, yy, zz, 0, .5, .5, 'zonal', grid.mesh)
        flux_u1_halfz = u1 * i_u.jacobian3D_lin_face(xx, yy, zz, 1, .5, .5, 'zonal', grid.mesh)
        flux_v0_halfz = v0 * i_u.jacobian3D_lin_face(xx, yy, zz, .5, 0, .5, 'meridional', grid.mesh)
        flux_v1_halfz = v1 * i_u.jacobian3D_lin_face(xx, yy, zz, .5, 1, .5, 'meridional', grid.mesh)
        flux_w0 = w0 * i_u.jacobian3D_lin_face(xx, yy, zz, .5, .5, 0, 'vertical', grid.mesh)
        flux_w05 = flux_u0_halfz - flux_u1_halfz + flux_v0_halfz - flux_v1_halfz + flux_w0

        surf_u05 = i_u.jacobian3D_lin_face(px, py, pz, .5, .5, .5, 'zonal', grid.mesh)
        jac_u05 = i_u.jacobian3D_lin_face(px, py, pz, .5, eta, zet, 'zonal', grid.mesh)
        U05 = flux_u05 / surf_u05 * jac_u05

        surf_v05 = i_u.jacobian3D_lin_face(px, py, pz, .5, .5, .5, 'meridional', grid.mesh)
        jac_v05 = i_u.jacobian3D_lin_face(px, py, pz, xsi, .5, zet, 'meridional', grid.mesh)
        V05 = flux_v05 / surf_v05 * jac_v05

        surf_w05 = i_u.jacobian3D_lin_face(px, py, pz, .5, .5, .5, 'vertical', grid.mesh)
        jac_w05 = i_u.jacobian3D_lin_face(px, py, pz, xsi, eta, .5, 'vertical', grid.mesh)
        W05 = flux_w05 / surf_w05 * jac_w05

        jac = i_u.jacobian3D_lin(px, py, pz, xsi, eta, zet, grid.mesh)
        dxsidt = i_u.interpolate(i_u.phi1D_quad, [U0, U05, U1], xsi) / jac
        detadt = i_u.interpolate(i_u.phi1D_quad, [V0, V05, V1], eta) / jac
        dzetdt = i_u.interpolate(i_u.phi1D_quad, [W0, W05, W1], zet) / jac

        dphidxsi, dphideta, dphidzet = i_u.dphidxsi3D_lin(xsi, eta, zet)

        u = np.dot(dphidxsi, px) * dxsidt + np.dot(dphideta, px) * detadt + np.dot(dphidzet, px) * dzetdt
        v = np.dot(dphidxsi, py) * dxsidt + np.dot(dphideta, py) * detadt + np.dot(dphidzet, py) * dzetdt
        w = np.dot(dphidxsi, pz) * dxsidt + np.dot(dphideta, pz) * detadt + np.dot(dphidzet, pz) * dzetdt
        return (u, v, w)

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
            (u, v, w) = self.spatial_c_grid_interpolation3D_full(ti, z, y, x, time)
        else:
            (u, v) = self.spatial_c_grid_interpolation2D(ti, z, y, x, time)
            w = self.W.eval(time, z, y, x, False)
            w = self.W.units.to_target(w, x, y, z)
        return (u, v, w)

    def eval(self, time, z, y, x):
        if self.U.interp_method != 'cgrid_velocity':
            u = self.U.eval(time, z, y, x, False)
            v = self.V.eval(time, z, y, x, False)
            u = self.U.units.to_target(u, x, y, z)
            v = self.V.units.to_target(v, x, y, z)
            if self.W is not None:
                w = self.W.eval(time, z, y, x, False)
                w = self.W.units.to_target(w, x, y, z)
                return (u, v, w)
            else:
                return (u, v)
        else:
            grid = self.U.grid
            (ti, periods) = self.U.time_index(time)
            time -= periods*(grid.time[-1]-grid.time[0])
            if ti < grid.tdim-1 and time > grid.time[ti]:
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

    def ccode_eval(self, varU, varV, varW, U, V, W, t, z, y, x):
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


class DeferredArray():
    """Class used for throwing error when Field.data is not read in deferred loading mode"""
    def __getitem__(self, key):
        raise RuntimeError('Field is in deferred_load mode, so can''t be accessed. Use .computeTimeChunk() method to force loading of  data')


class SummedVectorField(list):
    """Class SummedVectorField stores 2 or 3 SummedFields which defines together a vector field.
    This enables to interpolate them as one single vector SummedField in the kernels.

    :param name: Name of the vector field
    :param U: SummedField defining the zonal component
    :param V: SummedField defining the meridional component
    :param W: SummedField defining the vertical component (default: None)
    """

    def __init__(self, name, U, V, W=None):
        self.name = name
        self.U = U
        self.V = V
        self.W = W

    def eval(self, time, z, y, x):
        zonal = meridional = vertical = 0
        if self.W is not None:
            for (U, V, W) in zip(self.U, self.V, self.W):
                vfld = VectorField(self.name, U, V, W)
                vfld.fieldset = self.fieldset
                (tmp1, tmp2, tmp3) = vfld.eval(time, z, y, x)
                zonal += tmp1
                meridional += tmp2
                vertical += tmp3
            return (zonal, meridional, vertical)
        else:
            for (U, V) in zip(self.U, self.V):
                vfld = VectorField(self.name, U, V)
                vfld.fieldset = self.fieldset
                (tmp1, tmp2) = vfld.eval(time, z, y, x)
                zonal += tmp1
                meridional += tmp2
            return (zonal, meridional)

    def __getitem__(self, key):
        if isinstance(key, int):
            return list.__getitem__(self, key)
        else:
            return self.eval(*key)


class NestedField(list):
    """Class NestedField is a list of Fields from which the first one to be not declared out-of-boundaries
    at particle position is interpolated. This induces that the order of the fields in the list matters.
    Each one it its turn, a field is interpolated: if the interpolation succeeds or if an error other
    than `ErrorOutOfBounds` is thrown, the function is stopped. Otherwise, next field is interpolated.
    NestedField returns an `ErrorOutOfBounds` only if last field is as well out of boundaries.
    NestedField is composed of either Fields or VectorFields.

    :param name: Name of the Nested field
    :param U: List of fields (order matters). U can be a scalar Field, a VectorField, or the zonal component of the VectorField
    :param V: List of fields defining the meridional component (default: None)
    :param W: List of fields defining the vertical component (default: None)
    """

    def __init__(self, name, U, V=None, W=None):
        if V is None:
            for Ui in U:
                self.append(Ui)
        elif W is None:
            for (i, Ui, Vi) in zip(range(len(U)), U, V):
                self.append(VectorField(name+'_%d' % i, Ui, Vi))
        else:
            for (i, Ui, Vi, Wi) in zip(range(len(U)), U, V, W):
                self.append(VectorField(name+'_%d' % i, Ui, Vi, Wi))
        self.name = name

    def __getitem__(self, key):
        if isinstance(key, int):
            return list.__getitem__(self, key)
        else:
            for iField in range(len(self)):
                try:
                    val = list.__getitem__(self, iField).eval(*key)
                    break
                except FieldSamplingError:
                    if iField == len(self)-1:
                        raise
                    else:
                        pass
            return val


class NetcdfFileBuffer(object):
    """ Class that encapsulates and manages deferred access to file data. """

    def __init__(self, filename, dimensions, indices, netcdf_engine):
        self.filename = filename
        self.dimensions = dimensions  # Dict with dimension keys for file data
        self.indices = indices
        self.dataset = None
        self.netcdf_engine = netcdf_engine
        self.ti = None

    def __enter__(self):
        if self.netcdf_engine == 'xarray':
            self.dataset = self.filename
            return self
        try:
            self.dataset = xr.open_dataset(str(self.filename), decode_cf=True, engine=self.netcdf_engine)
            self.dataset['decoded'] = True
        except:
            logger.warning_once("File %s could not be decoded properly by xarray (version %s).\n         It will be opened with no decoding. Filling values might be wrongly parsed."
                                % (self.filename, xr.__version__))
            self.dataset = xr.open_dataset(str(self.filename), decode_cf=False, engine=self.netcdf_engine)
            self.dataset['decoded'] = False
        for inds in self.indices.values():
            if type(inds) not in [list, range]:
                raise RuntimeError('Indices for field subsetting need to be a list')
        return self

    def __exit__(self, type, value, traceback):
        if self.netcdf_engine == 'xarray':
            pass
        else:
            self.dataset.close()

    def parse_name(self, dimensions, variable):
        name = dimensions['data'] if 'data' in dimensions else variable
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
        if self.netcdf_engine == 'xarray':
            data = self.dataset
        else:
            data = self.dataset[self.name]
        ti = range(data.shape[0]) if self.ti is None else self.ti
        if len(data.shape) == 2:
            data = data[self.indices['lat'], self.indices['lon']]
        elif len(data.shape) == 3:
            if len(self.indices['depth']) > 1:
                data = data[self.indices['depth'], self.indices['lat'], self.indices['lon']]
            else:
                data = data[ti, self.indices['lat'], self.indices['lon']]
        else:
            data = data[ti, self.indices['depth'], self.indices['lat'], self.indices['lon']]

        if np.ma.is_masked(data):  # convert masked array to ndarray
            data = np.ma.filled(data, np.nan)
        return data

    @property
    def time(self):
        try:
            time_da = self.dataset[self.dimensions['time']]
            if self.netcdf_engine != 'xarray' and (self.dataset['decoded'] and 'Unit' not in time_da.attrs):
                time = np.array([time_da]) if len(time_da.shape) == 0 else np.array(time_da)
            else:
                if 'units' not in time_da.attrs and 'Unit' in time_da.attrs:
                    time_da.attrs['units'] = time_da.attrs['Unit']
                ds = xr.Dataset({self.dimensions['time']: time_da})
                ds = xr.decode_cf(ds)
                da = ds[self.dimensions['time']]
                time = np.array([da]) if len(da.shape) == 0 else np.array(da)
            if isinstance(time[0], datetime.datetime):
                raise NotImplementedError('Parcels currently only parses dates ranging from 1678 AD to 2262 AD, which are stored by xarray as np.datetime64. If you need a wider date range, please open an Issue on the parcels github page.')
            return time
        except:
            return np.array([None])

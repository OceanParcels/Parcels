import math

import numpy as np

import parcels.tools.interpolation_utils as i_u

# from .grid import Grid
# from .grid import GridCode
from parcels.numba.grid_all import BaseGrid
from parcels.numba.grid_all import GridCode
from parcels.tools.converters import Geographic
from parcels.tools.converters import GeographicPolar
from parcels.tools.converters import TimeConverter
from parcels.tools.converters import UnitConverter
from parcels.tools.converters import unitconverters_map
from parcels.tools.statuscodes import FieldOutOfBoundError
from parcels.tools.statuscodes import FieldOutOfBoundSurfaceError
from parcels.tools.statuscodes import FieldSamplingError
from parcels.tools.statuscodes import TimeExtrapolationError


__all__ = ['Field', 'VectorField', 'SummedField', 'NestedField']


# def _isParticle(key):
#     if hasattr(key, '_next_dt'):
#         return True
#     else:
#         return False


class NumbaField():
    def __init__(
            self, data, grid, fieldtype=None, transpose=False, vmin=None,
            vmax=None, time_origin=None, interp_method='linear',
            allow_time_extrapolation=None, time_periodic=False,
            gridindexingtype='nemo',
            to_write=False, **kwargs):
        self.data = data
#         time_origin = TimeConverter(0) if time_origin is None else time_origin
        self.grid = grid
        self.igrid = -1
        # self.lon, self.lat, self.depth and self.time are not used anymore in parcels.
        # self.grid should be used instead.
        # Those variables are still defined for backwards compatibility with users codes.
#         self.lon = self.grid.lon
#         self.lat = self.grid.lat
#         self.depth = self.grid.depth
        self.fieldtype = self.name if fieldtype is None else fieldtype
        self.to_write = to_write
        if self.grid.mesh == 'flat' or (self.fieldtype not in unitconverters_map.keys()):
            self.units = UnitConverter()
        elif self.grid.mesh == 'spherical':
            self.units = unitconverters_map[self.fieldtype]
        else:
            raise ValueError("Unsupported mesh type. Choose either: 'spherical' or 'flat'")
#         self.timestamps = timestamps
#         if type(interp_method) is dict:
#             if self.name in interp_method:
#                 self.interp_method = interp_method[self.name]
#             else:
#                 raise RuntimeError('interp_method is a dictionary but %s is not in it' % name)
#         else:
        self.interp_method = interp_method
        self.gridindexingtype = gridindexingtype
#         if self.interp_method in ['bgrid_velocity', 'bgrid_w_velocity', 'bgrid_tracer'] and \
#            self.grid.gtype in [GridCode.RectilinearSGrid, GridCode.CurvilinearSGrid]:
#             logger.warning_once('General s-levels are not supported in B-grid. RectilinearSGrid and CurvilinearSGrid can still be used to deal with shaved cells, but the levels must be horizontal.')
    
#         self.fieldset = None
        if allow_time_extrapolation is None:
            self.allow_time_extrapolation = True if len(self.grid.time) == 1 else False
        else:
            self.allow_time_extrapolation = allow_time_extrapolation
    
        self.time_periodic = time_periodic
        if self.time_periodic is not False and self.allow_time_extrapolation:
#             logger.warning_once("allow_time_extrapolation and time_periodic cannot be used together.\n \
#                                  allow_time_extrapolation is set to False")
            self.allow_time_extrapolation = False
        if self.time_periodic is True:
            raise ValueError("Unsupported time_periodic=True. time_periodic must now be either False or the length of the period (either float in seconds or datetime.timedelta object.")
        if self.time_periodic is not False:
            if isinstance(self.time_periodic, datetime.timedelta):
                self.time_periodic = self.time_periodic.total_seconds()
            if not np.isclose(self.grid.time[-1] - self.grid.time[0], self.time_periodic):
                if self.grid.time[-1] - self.grid.time[0] > self.time_periodic:
                    raise ValueError("Time series provided is longer than the time_periodic parameter")
                self.grid._add_last_periodic_data_timestep = True
                self.grid.time = np.append(self.grid.time, self.grid.time[0] + self.time_periodic)
                self.grid.time_full = self.grid.time

        self.vmin = vmin
        self.vmax = vmax
#         self.cast_data_dtype = cast_data_dtype
#         if self.cast_data_dtype == 'float32':
#             self.cast_data_dtype = np.float32
#         elif self.cast_data_dtype == 'float64':
#             self.cast_data_dtype = np.float64

        if not self.grid.defer_load:
            self.data = self.reshape(self.data, transpose)

            # Hack around the fact that NaN and ridiculously large values
            # propagate in SciPy's interpolators
#             lib = np if isinstance(self.data, np.ndarray) else da
            self.data[np.isnan(self.data)] = 0.
            if self.vmin is not None:
                self.data[self.data < self.vmin] = 0.
            if self.vmax is not None:
                self.data[self.data > self.vmax] = 0.

            if self.grid._add_last_periodic_data_timestep:
                self.data = np.concatenate((self.data, self.data[:1, :]), axis=0)

        self._scaling_factor = None

        # Variable names in JIT code
#         self.dimensions = kwargs.pop('dimensions', None)
#         self.indices = kwargs.pop('indices', None)
#         self.dataFiles = kwargs.pop('dataFiles', None)
#         if self.grid._add_last_periodic_data_timestep and self.dataFiles is not None:
#             self.dataFiles = np.append(self.dataFiles, self.dataFiles[0])
#         self._field_fb_class = kwargs.pop('FieldFileBuffer', None)
#         self.netcdf_engine = kwargs.pop('netcdf_engine', 'netcdf4')
#         self.loaded_time_indices = []
#         self.creation_log = kwargs.pop('creation_log', '')
#         self.chunksize = kwargs.pop('chunksize', None)
#         self.netcdf_chunkdims_name_map = kwargs.pop('chunkdims_name_map', None)
    #         self.grid.depth_field = kwargs.pop('depth_field', None)
    
    #         if self.grid.depth_field == 'not_yet_set':
    #             assert self.grid.z4d, 'Providing the depth dimensions from another field data is only available for 4d S grids'
    
        # data_full_zdim is the vertical dimension of the complete field data, ignoring the indices.
        # (data_full_zdim = grid.zdim if no indices are used, for A- and C-grids and for some B-grids). It is used for the B-grid,
        # since some datasets do not provide the deeper level of data (which is ignored by the interpolation).
#         self.data_full_zdim = kwargs.pop('data_full_zdim', None)
#         self.data_chunks = []
#         self.c_data_chunks = []
#         self.nchunks = []
#         self.chunk_set = False
#         self.filebuffers = [None] * 2
#         if len(kwargs) > 0:
#             raise SyntaxError('Field received an unexpected keyword argument "%s"' % list(kwargs.keys())[0])
    def reshape(self, data, transpose=False):
        # Ensure that field data is the right data type
#         if not isinstance(data, (np.ndarray, da.core.Array)):
#             data = np.array(data)
#         if (self.cast_data_dtype == np.float32) and (data.dtype != np.float32):
#             data = data.astype(np.float32)
#         elif (self.cast_data_dtype == np.float64) and (data.dtype != np.float64):
#             data = data.astype(np.float64)
#         lib = np if isinstance(data, np.ndarray) else da
        if transpose:
            data = np.transpose(data)
        if self.grid.lat_flipped:
            data = np.flip(data, axis=-2)

        if self.grid.xdim == 1 or self.grid.ydim == 1:
            data = np.squeeze(data)  # First remove all length-1 dimensions in data, so that we can add them below
        if self.grid.xdim == 1 and len(data.shape) < 4:
#             if lib == da:
#                 raise NotImplementedError('Length-one dimensions with field chunking not implemented, as dask does not have an `expand_dims` method. Use chunksize=None')
            data = np.expand_dims(data, axis=-1)
        if self.grid.ydim == 1 and len(data.shape) < 4:
#             if lib == da:
#                 raise NotImplementedError('Length-one dimensions with field chunking not implemented, as dask does not have an `expand_dims` method. Use chunksize=None')
            data = np.expand_dims(data, axis=-2)
        if self.grid.tdim == 1:
            if len(data.shape) < 4:
                data = data.reshape(sum(((1,), data.shape), ()))
        if self.grid.zdim == 1:
            if len(data.shape) == 4:
                data = data.reshape(sum(((data.shape[0],), data.shape[2:]), ()))
        if len(data.shape) == 4:
            errormessage = ('Field %s expecting a data shape of [tdim, zdim, ydim, xdim]. '
                            'Flag transpose=True could help to reorder the data.' % self.name)
            assert data.shape[0] == self.grid.tdim, errormessage
            assert data.shape[2] == self.grid.ydim - 2 * self.grid.meridional_halo, errormessage
            assert data.shape[3] == self.grid.xdim - 2 * self.grid.zonal_halo, errormessage
            if self.gridindexingtype == 'pop':
                assert data.shape[1] == self.grid.zdim or data.shape[1] == self.grid.zdim-1, errormessage
            else:
                assert data.shape[1] == self.grid.zdim, errormessage
        else:
            assert (data.shape == (self.grid.tdim,
                                   self.grid.ydim - 2 * self.grid.meridional_halo,
                                   self.grid.xdim - 2 * self.grid.zonal_halo)), \
                ('Field %s expecting a data shape of [tdim, ydim, xdim]. '
                 'Flag transpose=True could help to reorder the data.' % self.name)
        if self.grid.meridional_halo > 0 or self.grid.zonal_halo > 0:
            data = self.add_periodic_halo(zonal=self.grid.zonal_halo > 0, meridional=self.grid.meridional_halo > 0, halosize=max(self.grid.meridional_halo, self.grid.zonal_halo), data=data)
        return data

    def set_scaling_factor(self, factor):
        """Scales the field data by some constant factor.

        :param factor: scaling factor

        For usage examples see the following tutorial:

        * `Unit converters <https://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_unitconverters.ipynb>`_
        """
        if self._scaling_factor:
            raise NotImplementedError(('Scaling factor for field %s already defined.' % self.name))
        self._scaling_factor = factor
        if not self.grid.defer_load:
            self.data *= factor

    def set_depth_from_field(self, field):
        """Define the depth dimensions from another (time-varying) field

        See `this tutorial <https://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_timevaryingdepthdimensions.ipynb>`_
        for a detailed explanation on how to set up time-evolving depth dimensions

        """
        self.grid.depth_field = field
        if self.grid != field.grid:
            field.grid.depth_field = field

    def calc_cell_edge_sizes(self):
        """Method to calculate cell sizes based on numpy.gradient method
                Currently only works for Rectilinear Grids"""
        if not self.grid.cell_edge_sizes:
            if self.grid.gtype in (GridCode.RectilinearZGrid, GridCode.RectilinearSGrid):
                self.grid.cell_edge_sizes['x'] = np.zeros((self.grid.ydim, self.grid.xdim), dtype=np.float32)
                self.grid.cell_edge_sizes['y'] = np.zeros((self.grid.ydim, self.grid.xdim), dtype=np.float32)

                x_conv = GeographicPolar() if self.grid.mesh == 'spherical' else UnitConverter()
                y_conv = Geographic() if self.grid.mesh == 'spherical' else UnitConverter()
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

    def search_indices(self, x, y, z, ti=-1, time=-1, particle=None, search2D=False):
        if self.grid.gtype in [GridCode.RectilinearSGrid, GridCode.RectilinearZGrid]:
            return self.search_indices_rectilinear(x, y, z, ti, time, particle=particle, search2D=search2D)
        else:
            return self.search_indices_curvilinear(x, y, z, ti, time, particle=particle, search2D=search2D)

    def interpolator2D(self, ti, z, y, x, particle=None):
        (xsi, eta, _, xi, yi, _) = self.search_indices(x, y, z, particle=particle)
        if self.interp_method == 'nearest':
            xii = xi if xsi <= .5 else xi+1
            yii = yi if eta <= .5 else yi+1
            return self.data[ti, yii, xii]
        elif self.interp_method in ['linear', 'bgrid_velocity', 'partialslip', 'freeslip']:
            val = (1-xsi)*(1-eta) * self.data[ti, yi, xi] + \
                xsi*(1-eta) * self.data[ti, yi, xi+1] + \
                xsi*eta * self.data[ti, yi+1, xi+1] + \
                (1-xsi)*eta * self.data[ti, yi+1, xi]
            return val
        elif self.interp_method == 'linear_invdist_land_tracer':
            land = np.isclose(self.data[ti, yi:yi+2, xi:xi+2], 0.)
            nb_land = np.sum(land)
            if nb_land == 4:
                return 0
            elif nb_land > 0:
                val = 0
                w_sum = 0
                for j in range(2):
                    for i in range(2):
                        distance = pow((eta - j), 2) + pow((xsi - i), 2)
                        if np.isclose(distance, 0):
                            if land[j][i] == 1:  # index search led us directly onto land
                                return 0
                            else:
                                return self.data[ti, yi+j, xi+i]
                        elif land[i][j] == 0:
                            val += self.data[ti, yi+j, xi+i] / distance
                            w_sum += 1 / distance
                return val / w_sum
            else:
                val = (1 - xsi) * (1 - eta) * self.data[ti, yi, xi] + \
                    xsi * (1 - eta) * self.data[ti, yi, xi + 1] + \
                    xsi * eta * self.data[ti, yi + 1, xi + 1] + \
                    (1 - xsi) * eta * self.data[ti, yi + 1, xi]
                return val
        elif self.interp_method in ['cgrid_tracer', 'bgrid_tracer']:
            return self.data[ti, yi+1, xi+1]
        elif self.interp_method == 'cgrid_velocity':
            raise RuntimeError("%s is a scalar field. cgrid_velocity interpolation method should be used for vector fields (e.g. FieldSet.UV)" % self.name)
        else:
            raise RuntimeError(self.interp_method+" is not implemented for 2D grids")

    def interpolator3D(self, ti, z, y, x, time, particle=None):
        (xsi, eta, zeta, xi, yi, zi) = self.search_indices(x, y, z, ti, time, particle=particle)
        if self.interp_method == 'nearest':
            xii = xi if xsi <= .5 else xi+1
            yii = yi if eta <= .5 else yi+1
            zii = zi if zeta <= .5 else zi+1
            return self.data[ti, zii, yii, xii]
        elif self.interp_method == 'cgrid_velocity':
            # evaluating W velocity in c_grid
            if self.gridindexingtype == 'nemo':
                f0 = self.data[ti, zi, yi+1, xi+1]
                f1 = self.data[ti, zi+1, yi+1, xi+1]
            elif self.gridindexingtype == 'mitgcm':
                f0 = self.data[ti, zi, yi, xi]
                f1 = self.data[ti, zi+1, yi, xi]
            return (1-zeta) * f0 + zeta * f1
        elif self.interp_method == 'linear_invdist_land_tracer':
            land = np.isclose(self.data[ti, zi:zi+2, yi:yi+2, xi:xi+2], 0.)
            nb_land = np.sum(land)
            if nb_land == 8:
                return 0
            elif nb_land > 0:
                val = 0
                w_sum = 0
                for k in range(2):
                    for j in range(2):
                        for i in range(2):
                            distance = pow((zeta - k), 2) + pow((eta - j), 2) + pow((xsi - i), 2)
                            if np.isclose(distance, 0):
                                if land[k][j][i] == 1:  # index search led us directly onto land
                                    return 0
                                else:
                                    return self.data[ti, zi+i, yi+j, xi+k]
                            elif land[k][j][i] == 0:
                                val += self.data[ti, zi+k, yi+j, xi+i] / distance
                                w_sum += 1 / distance
                return val / w_sum
            else:
                data = self.data[ti, zi, :, :]
                f0 = (1 - xsi) * (1 - eta) * data[yi, xi] + \
                    xsi * (1 - eta) * data[yi, xi + 1] + \
                    xsi * eta * data[yi + 1, xi + 1] + \
                    (1 - xsi) * eta * data[yi + 1, xi]
                data = self.data[ti, zi + 1, :, :]
                f1 = (1 - xsi) * (1 - eta) * data[yi, xi] + \
                    xsi * (1 - eta) * data[yi, xi + 1] + \
                    xsi * eta * data[yi + 1, xi + 1] + \
                    (1 - xsi) * eta * data[yi + 1, xi]
                return (1 - zeta) * f0 + zeta * f1
        elif self.interp_method in ['linear', 'bgrid_velocity', 'bgrid_w_velocity', 'partialslip', 'freeslip']:
            if self.interp_method == 'bgrid_velocity':
                if self.gridindexingtype == 'mom5':
                    zeta = 1.
                else:
                    zeta = 0.
            elif self.interp_method == 'bgrid_w_velocity':
                eta = 1.
                xsi = 1.
            data = self.data[ti, zi, :, :]
            f0 = (1-xsi)*(1-eta) * data[yi, xi] + \
                xsi*(1-eta) * data[yi, xi+1] + \
                xsi*eta * data[yi+1, xi+1] + \
                (1-xsi)*eta * data[yi+1, xi]
            if self.gridindexingtype == 'pop' and zi >= self.grid.zdim-2:
                # Since POP is indexed at cell top, allow linear interpolation of W to zero in lowest cell
                return (1-zeta) * f0
            data = self.data[ti, zi+1, :, :]
            f1 = (1-xsi)*(1-eta) * data[yi, xi] + \
                xsi*(1-eta) * data[yi, xi+1] + \
                xsi*eta * data[yi+1, xi+1] + \
                (1-xsi)*eta * data[yi+1, xi]
            if self.interp_method == 'bgrid_w_velocity' and self.gridindexingtype == 'mom5' and zi == -1:
                # Since MOM5 is indexed at cell bottom, allow linear interpolation of W to zero in uppermost cell
                return zeta * f1
            else:
                return (1-zeta) * f0 + zeta * f1
        elif self.interp_method in ['cgrid_tracer', 'bgrid_tracer']:
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

    def spatial_interpolation(self, ti, z, y, x, time, particle=None):
        """Interpolate horizontal field values using a SciPy interpolator"""

        if self.grid.zdim == 1:
            val = self.interpolator2D(ti, z, y, x, particle=particle)
        else:
            val = self.interpolator3D(ti, z, y, x, time, particle=particle)
        if np.isnan(val):
            # Detect Out-of-bounds sampling and raise exception
            raise FieldOutOfBoundError(x, y, z, field=self)
        else:
#             if isinstance(val, da.core.Array):
#                 val = val.compute()
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
                periods = int(math.floor((time-self.grid.time_full[0])/(self.grid.time_full[-1]-self.grid.time_full[0])))
                if isinstance(self.grid.periods, c_int):
                    self.grid.periods.value = periods
                else:
                    self.grid.periods = periods
                time -= periods*(self.grid.time_full[-1]-self.grid.time_full[0])
                time_index = self.grid.time <= time
                ti = time_index.argmin() - 1 if time_index.any() else 0
                return (ti, periods)
            return (time_index.argmin() - 1 if time_index.any() else 0, 0)
        if time_index.all():
            # If given time > last known field time, use
            # the last field frame without interpolation
            return (len(self.grid.time) - 1, 0)
        elif np.logical_not(time_index).all():
            # If given time < any time in the field, use
            # the first field frame without interpolation
            return (0, 0)
        else:
            return (time_index.argmin() - 1 if time_index.any() else 0, 0)

    def __getitem__(self, key):
        if _isParticle(key):
            return self.eval(key.time, key.depth, key.lat, key.lon, key)
        else:
            return self.eval(*key)

    def eval(self, time, z, y, x, particle=None, applyConversion=True):
        """Interpolate field values in space and time.

        We interpolate linearly in time and apply implicit unit
        conversion to the result. Note that we defer to
        scipy.interpolate to perform spatial interpolation.
        """
        (ti, periods) = self.time_index(time)
        time -= periods*(self.grid.time_full[-1]-self.grid.time_full[0])
        if ti < self.grid.tdim-1 and time > self.grid.time[ti]:
            f0 = self.spatial_interpolation(ti, z, y, x, time, particle=particle)
            f1 = self.spatial_interpolation(ti + 1, z, y, x, time, particle=particle)
            t0 = self.grid.time[ti]
            t1 = self.grid.time[ti + 1]
            value = f0 + (f1 - f0) * ((time - t0) / (t1 - t0))
        else:
            # Skip temporal interpolation if time is outside
            # of the defined time range or if we have hit an
            # excat value in the time array.
            value = self.spatial_interpolation(ti, z, y, x, self.grid.time[ti], particle=particle)

        if applyConversion:
            return self.units.to_target(value, x, y, z)
        else:
            return value

    def get_block_id(self, block):
        return np.ravel_multi_index(block, self.nchunks)

    def get_block(self, bid):
        return np.unravel_index(bid, self.nchunks[1:])

    def chunk_setup(self):
        if isinstance(self.data, da.core.Array):
            chunks = self.data.chunks
            self.nchunks = self.data.numblocks
            npartitions = 1
            for n in self.nchunks[1:]:
                npartitions *= n
        elif isinstance(self.data, np.ndarray):
            chunks = tuple((t,) for t in self.data.shape)
            self.nchunks = (1,) * len(self.data.shape)
            npartitions = 1
        elif isinstance(self.data, DeferredArray):
            self.nchunks = (1,) * len(self.data.data_shape)
            return
        else:
            return

        self.data_chunks = [None] * npartitions
        self.c_data_chunks = [None] * npartitions
        self.grid.load_chunk = np.zeros(npartitions, dtype=c_int)
        # self.grid.chunk_info format: number of dimensions (without tdim); number of chunks per dimensions;
        #      chunksizes (the 0th dim sizes for all chunk of dim[0], then so on for next dims
        self.grid.chunk_info = [[len(self.nchunks)-1], list(self.nchunks[1:]), sum(list(list(ci) for ci in chunks[1:]), [])]
        self.grid.chunk_info = sum(self.grid.chunk_info, [])
        self.chunk_set = True

    def chunk_data(self):
        if not self.chunk_set:
            self.chunk_setup()
        g = self.grid
        if isinstance(self.data, da.core.Array):
            for block_id in range(len(self.grid.load_chunk)):
                if g.load_chunk[block_id] == g.chunk_loading_requested \
                        or g.load_chunk[block_id] in g.chunk_loaded and self.data_chunks[block_id] is None:
                    block = self.get_block(block_id)
                    self.data_chunks[block_id] = np.array(self.data.blocks[(slice(self.grid.tdim),) + block])
                elif g.load_chunk[block_id] == g.chunk_not_loaded:
                    if isinstance(self.data_chunks, list):
                        self.data_chunks[block_id] = None
                    else:
                        self.data_chunks[block_id, :] = None
                    self.c_data_chunks[block_id] = None
        else:
            if isinstance(self.data_chunks, list):
                self.data_chunks[0] = None
            else:
                self.data_chunks[0, :] = None
            self.c_data_chunks[0] = None
            self.grid.load_chunk[0] = g.chunk_loaded_touched
            self.data_chunks[0] = np.array(self.data)

    def add_periodic_halo(self, zonal, meridional, halosize=5, data=None):
        """Add a 'halo' to all Fields in a FieldSet, through extending the Field (and lon/lat)
        by copying a small portion of the field on one side of the domain to the other.
        Before adding a periodic halo to the Field, it has to be added to the Grid on which the Field depends

        See `this tutorial <https://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_periodic_boundaries.ipynb>`_
        for a detailed explanation on how to set up periodic boundaries

        :param zonal: Create a halo in zonal direction (boolean)
        :param meridional: Create a halo in meridional direction (boolean)
        :param halosize: size of the halo (in grid points). Default is 5 grid points
        :param data: if data is not None, the periodic halo will be achieved on data instead of self.data and data will be returned
        """
        lib = np
        if zonal:
            if len(data.shape) == 3:
                data = lib.concatenate((data[:, :, -halosize:], data,
                                       data[:, :, 0:halosize]), axis=len(data.shape)-1)
                assert data.shape[2] == self.grid.xdim, "Third dim must be x."
            else:
                data = lib.concatenate((data[:, :, :, -halosize:], data,
                                       data[:, :, :, 0:halosize]), axis=len(data.shape) - 1)
                assert data.shape[3] == self.grid.xdim, "Fourth dim must be x."
#             self.lon = self.grid.lon
#             self.lat = self.grid.lat
        if meridional:
            if len(data.shape) == 3:
                data = lib.concatenate((data[:, -halosize:, :], data,
                                       data[:, 0:halosize, :]), axis=len(data.shape)-2)
                assert data.shape[1] == self.grid.ydim, "Second dim must be y."
            else:
                data = lib.concatenate((data[:, :, -halosize:, :], data,
                                       data[:, :, 0:halosize, :]), axis=len(data.shape) - 2)
                assert data.shape[2] == self.grid.ydim, "Third dim must be y."
#             self.lat = self.grid.lat
        return data

    def rescale_and_set_minmax(self, data):
        data[np.isnan(data)] = 0
        if self._scaling_factor:
            data *= self._scaling_factor
        if self.vmin is not None:
            data[data < self.vmin] = 0
        if self.vmax is not None:
            data[data > self.vmax] = 0
        return data

    def data_concatenate(self, data, data_to_concat, tindex):
#         if data[tindex] is not None:
#             if isinstance(data, np.ndarray):
#                 data[tindex] = None
#             elif isinstance(data, list):
#                 del data[tindex]
        lib = np
        if tindex == 0:
            data = lib.concatenate([data_to_concat, data[tindex+1:, :]], axis=0)
        elif tindex == 1:
            data = lib.concatenate([data[:tindex, :], data_to_concat], axis=0)
        else:
            raise ValueError("data_concatenate is used for computeTimeChunk, with tindex in [0, 1]")
        return data

    def advancetime(self, field_new, advanceForward):
#         if isinstance(self.data) is not isinstance(field_new):
#             logger.warning("[Field.advancetime] New field data and persistent field data have different types - time advance not possible.")
#             return
        lib = np 
        if advanceForward == 1:  # forward in time, so appending at end
            self.data = lib.concatenate((self.data[1:, :, :], field_new.data[:, :, :]), 0)
            self.time = self.grid.time
        else:  # backward in time, so prepending at start
            self.data = lib.concatenate((field_new.data[:, :, :], self.data[:-1, :, :]), 0)
            self.time = self.grid.time


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
        self.vector_type = '3D' if W else '2D'
        self.gridindexingtype = U.gridindexingtype
        if self.U.interp_method == 'cgrid_velocity':
            assert self.V.interp_method == 'cgrid_velocity', (
                'Interpolation methods of U and V are not the same.')
            assert self._check_grid_dimensions(U.grid, V.grid), (
                'Dimensions of U and V are not the same.')
            if self.vector_type == '3D':
                assert self.W.interp_method == 'cgrid_velocity', (
                    'Interpolation methods of U and W are not the same.')
                assert self._check_grid_dimensions(U.grid, W.grid), (
                    'Dimensions of U and W are not the same.')

    @staticmethod
    def _check_grid_dimensions(grid1, grid2):
        return (np.allclose(grid1.lon, grid2.lon) and np.allclose(grid1.lat, grid2.lat)
                and np.allclose(grid1.depth, grid2.depth) and np.allclose(grid1.time_full, grid2.time_full))

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

    def spatial_c_grid_interpolation2D(self, ti, z, y, x, time, particle=None):
        grid = self.U.grid
        (xsi, eta, zeta, xi, yi, zi) = self.U.search_indices(x, y, z, ti, time, particle=particle)

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
            if self.gridindexingtype == 'nemo':
                U0 = self.U.data[ti, yi+1, xi] * c4
                U1 = self.U.data[ti, yi+1, xi+1] * c2
                V0 = self.V.data[ti, yi, xi+1] * c1
                V1 = self.V.data[ti, yi+1, xi+1] * c3
            elif self.gridindexingtype == 'mitgcm':
                U0 = self.U.data[ti, yi, xi] * c4
                U1 = self.U.data[ti, yi, xi + 1] * c2
                V0 = self.V.data[ti, yi, xi] * c1
                V1 = self.V.data[ti, yi + 1, xi] * c3
        else:
            if self.gridindexingtype == 'nemo':
                U0 = self.U.data[ti, zi, yi+1, xi] * c4
                U1 = self.U.data[ti, zi, yi+1, xi+1] * c2
                V0 = self.V.data[ti, zi, yi, xi+1] * c1
                V1 = self.V.data[ti, zi, yi+1, xi+1] * c3
            elif self.gridindexingtype == 'mitgcm':
                U0 = self.U.data[ti, zi, yi, xi] * c4
                U1 = self.U.data[ti, zi, yi, xi + 1] * c2
                V0 = self.V.data[ti, zi, yi, xi] * c1
                V1 = self.V.data[ti, zi, yi + 1, xi] * c3
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
        if isinstance(u, da.core.Array):
            u = u.compute()
            v = v.compute()
        return (u, v)

    def spatial_c_grid_interpolation3D_full(self, ti, z, y, x, time, particle=None):
        grid = self.U.grid
        (xsi, eta, zet, xi, yi, zi) = self.U.search_indices(x, y, z, ti, time, particle=particle)

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

#         if isinstance(u, da.core.Array):
#             u = u.compute()
#             v = v.compute()
#             w = w.compute()
        return (u, v, w)

    def spatial_c_grid_interpolation3D(self, ti, z, y, x, time, particle=None):
        """
        +---+---+---+
        |   |V1 |   |
        +---+---+---+
        |U0 |   |U1 |
        +---+---+---+
        |   |V0 |   |
        +---+---+---+

        The interpolation is done in the following by
        interpolating linearly U depending on the longitude coordinate and
        interpolating linearly V depending on the latitude coordinate.
        Curvilinear grids are treated properly, since the element is projected to a rectilinear parent element.
        """
        if self.U.grid.gtype in [GridCode.RectilinearSGrid, GridCode.CurvilinearSGrid]:
            (u, v, w) = self.spatial_c_grid_interpolation3D_full(ti, z, y, x, time, particle=particle)
        else:
            (u, v) = self.spatial_c_grid_interpolation2D(ti, z, y, x, time, particle=particle)
            w = self.W.eval(time, z, y, x, particle=particle, applyConversion=False)
            w = self.W.units.to_target(w, x, y, z)
        return (u, v, w)

    def _is_land2D(self, di, yi, xi):
        if self.U.data.ndim == 3:
            if di < np.shape(self.U.data)[0]:
                return np.isclose(self.U.data[di, yi, xi], 0.) and np.isclose(self.V.data[di, yi, xi], 0.)
            else:
                return True
        else:
            if di < self.U.grid.zdim and yi < np.shape(self.U.data)[-2] and xi < np.shape(self.U.data)[-1]:
                return np.isclose(self.U.data[0, di, yi, xi], 0.) and np.isclose(self.V.data[0, di, yi, xi], 0.)
            else:
                return True

    def spatial_slip_interpolation(self, ti, z, y, x, time, particle=None):
        (xsi, eta, zeta, xi, yi, zi) = self.U.search_indices(x, y, z, ti, time, particle=particle)
        di = ti if self.U.grid.zdim == 1 else zi  # general third dimension

        f_u, f_v, f_w = 1, 1, 1
        if self._is_land2D(di, yi, xi) and self._is_land2D(di, yi, xi+1) and self._is_land2D(di+1, yi, xi) \
                and self._is_land2D(di+1, yi, xi+1) and eta > 0:
            if self.U.interp_method == 'partialslip':
                f_u = f_u * (.5 + .5 * eta) / eta
                if self.vector_type == '3D':
                    f_w = f_w * (.5 + .5 * eta) / eta
            elif self.U.interp_method == 'freeslip':
                f_u = f_u / eta
                if self.vector_type == '3D':
                    f_w = f_w / eta
        if self._is_land2D(di, yi+1, xi) and self._is_land2D(di, yi+1, xi+1) and self._is_land2D(di+1, yi+1, xi) \
                and self._is_land2D(di+1, yi+1, xi+1) and eta < 1:
            if self.U.interp_method == 'partialslip':
                f_u = f_u * (1 - .5 * eta) / (1 - eta)
                if self.vector_type == '3D':
                    f_w = f_w * (1 - .5 * eta) / (1 - eta)
            elif self.U.interp_method == 'freeslip':
                f_u = f_u / (1 - eta)
                if self.vector_type == '3D':
                    f_w = f_w / (1 - eta)
        if self._is_land2D(di, yi, xi) and self._is_land2D(di, yi+1, xi) and self._is_land2D(di+1, yi, xi) \
                and self._is_land2D(di+1, yi+1, xi) and xsi > 0:
            if self.U.interp_method == 'partialslip':
                f_v = f_v * (.5 + .5 * xsi) / xsi
                if self.vector_type == '3D':
                    f_w = f_w * (.5 + .5 * xsi) / xsi
            elif self.U.interp_method == 'freeslip':
                f_v = f_v / xsi
                if self.vector_type == '3D':
                    f_w = f_w / xsi
        if self._is_land2D(di, yi, xi+1) and self._is_land2D(di, yi+1, xi+1) and self._is_land2D(di+1, yi, xi+1) \
                and self._is_land2D(di+1, yi+1, xi+1) and xsi < 1:
            if self.U.interp_method == 'partialslip':
                f_v = f_v * (1 - .5 * xsi) / (1 - xsi)
                if self.vector_type == '3D':
                    f_w = f_w * (1 - .5 * xsi) / (1 - xsi)
            elif self.U.interp_method == 'freeslip':
                f_v = f_v / (1 - xsi)
                if self.vector_type == '3D':
                    f_w = f_w / (1 - xsi)
        if self.U.grid.zdim > 1:
            if self._is_land2D(di, yi, xi) and self._is_land2D(di, yi, xi+1) and self._is_land2D(di, yi+1, xi) \
                    and self._is_land2D(di, yi+1, xi+1) and zeta > 0:
                if self.U.interp_method == 'partialslip':
                    f_u = f_u * (.5 + .5 * zeta) / zeta
                    f_v = f_v * (.5 + .5 * zeta) / zeta
                elif self.U.interp_method == 'freeslip':
                    f_u = f_u / zeta
                    f_v = f_v / zeta
            if self._is_land2D(di+1, yi, xi) and self._is_land2D(di+1, yi, xi+1) and self._is_land2D(di+1, yi+1, xi) \
                    and self._is_land2D(di+1, yi+1, xi+1) and zeta < 1:
                if self.U.interp_method == 'partialslip':
                    f_u = f_u * (1 - .5 * zeta) / (1 - zeta)
                    f_v = f_v * (1 - .5 * zeta) / (1 - zeta)
                elif self.U.interp_method == 'freeslip':
                    f_u = f_u / (1 - zeta)
                    f_v = f_v / (1 - zeta)

        u = f_u * self.U.eval(time, z, y, x, particle)
        v = f_v * self.V.eval(time, z, y, x, particle)
        if self.vector_type == '3D':
            w = f_w * self.W.eval(time, z, y, x, particle)
            return u, v, w
        else:
            return u, v

    def eval(self, time, z, y, x, particle=None):
        if self.U.interp_method not in ['cgrid_velocity', 'partialslip', 'freeslip']:
            u = self.U.eval(time, z, y, x, particle=particle, applyConversion=False)
            v = self.V.eval(time, z, y, x, particle=particle, applyConversion=False)
            u = self.U.units.to_target(u, x, y, z)
            v = self.V.units.to_target(v, x, y, z)
            if self.vector_type == '3D':
                w = self.W.eval(time, z, y, x, particle=particle, applyConversion=False)
                w = self.W.units.to_target(w, x, y, z)
                return (u, v, w)
            else:
                return (u, v)
        else:
            interp = {'cgrid_velocity': {'2D': self.spatial_c_grid_interpolation2D, '3D': self.spatial_c_grid_interpolation3D},
                      'partialslip': {'2D': self.spatial_slip_interpolation, '3D': self.spatial_slip_interpolation},
                      'freeslip': {'2D': self.spatial_slip_interpolation, '3D': self.spatial_slip_interpolation}}
            grid = self.U.grid
            (ti, periods) = self.U.time_index(time)
            time -= periods*(grid.time_full[-1]-grid.time_full[0])
            if ti < grid.tdim-1 and time > grid.time[ti]:
                t0 = grid.time[ti]
                t1 = grid.time[ti + 1]
                if self.vector_type == '3D':
                    (u0, v0, w0) = interp[self.U.interp_method]['3D'](ti, z, y, x, time, particle=particle)
                    (u1, v1, w1) = interp[self.U.interp_method]['3D'](ti + 1, z, y, x, time, particle=particle)
                    w = w0 + (w1 - w0) * ((time - t0) / (t1 - t0))
                else:
                    (u0, v0) = interp[self.U.interp_method]['2D'](ti, z, y, x, time, particle=particle)
                    (u1, v1) = interp[self.U.interp_method]['2D'](ti + 1, z, y, x, time, particle=particle)
                u = u0 + (u1 - u0) * ((time - t0) / (t1 - t0))
                v = v0 + (v1 - v0) * ((time - t0) / (t1 - t0))
                if self.vector_type == '3D':
                    return (u, v, w)
                else:
                    return (u, v)
            else:
                # Skip temporal interpolation if time is outside
                # of the defined time range or if we have hit an
                # exact value in the time array.
                if self.vector_type == '3D':
                    return interp[self.U.interp_method]['3D'](ti, z, y, x, grid.time[ti], particle=particle)
                else:
                    return interp[self.U.interp_method]['2D'](ti, z, y, x, grid.time[ti], particle=particle)

#     def __getitem__(self, key):
#         if _isParticle(key):
#             return self.eval(key.time, key.depth, key.lat, key.lon, key)
#         else:
#             return self.eval(*key)


class DeferredArray():
    """Class used for throwing error when Field.data is not read in deferred loading mode"""
    data_shape = ()

    def __init__(self):
        self.data_shape = (1,)

    def compute_shape(self, xdim, ydim, zdim, tdim, tslices):
        if zdim == 1 and tdim == 1:
            self.data_shape = (tslices, 1, ydim, xdim)
        elif zdim > 1 or tdim > 1:
            if zdim > 1:
                self.data_shape = (1, zdim, ydim, xdim)
            else:
                self.data_shape = (max(tdim, tslices), 1, ydim, xdim)
        else:
            self.data_shape = (tdim, zdim, ydim, xdim)
        return self.data_shape

    def __getitem__(self, key):
        raise RuntimeError("Field is in deferred_load mode, so can't be accessed. Use .computeTimeChunk() method to force loading of data")


class SummedField(list):
    """Class SummedField is a list of Fields over which Field interpolation
    is summed. This can e.g. be used when combining multiple flow fields,
    where the total flow is the sum of all the individual flows.
    Note that the individual Fields can be on different Grids.
    Also note that, since SummedFields are lists, the individual Fields can
    still be queried through their list index (e.g. SummedField[1]).
    SummedField is composed of either Fields or VectorFields.

    See `here <https://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_SummedFields.ipynb>`_
    for a detailed tutorial

    :param name: Name of the SummedField
    :param F: List of fields. F can be a scalar Field, a VectorField, or the zonal component (U) of the VectorField
    :param V: List of fields defining the meridional component of a VectorField, if F is the zonal component. (default: None)
    :param W: List of fields defining the vertical component of a VectorField, if F and V are the zonal and meridional components (default: None)
    """

    def __init__(self, name, F, V=None, W=None):
        if V is None:
            if isinstance(F[0], VectorField):
                vector_type = F[0].vector_type
            for Fi in F:
                assert isinstance(Fi, Field) or (isinstance(Fi, VectorField) and Fi.vector_type == vector_type), 'Components of a SummedField must be Field or VectorField'
                self.append(Fi)
        elif W is None:
            for (i, Fi, Vi) in zip(range(len(F)), F, V):
                assert isinstance(Fi, Field) and isinstance(Vi, Field), \
                    'F, and V components of a SummedField must be Field'
                self.append(VectorField(name+'_%d' % i, Fi, Vi))
        else:
            for (i, Fi, Vi, Wi) in zip(range(len(F)), F, V, W):
                assert isinstance(Fi, Field) and isinstance(Vi, Field) and isinstance(Wi, Field), \
                    'F, V and W components of a SummedField must be Field'
                self.append(VectorField(name+'_%d' % i, Fi, Vi, Wi))
        self.name = name

    def __getitem__(self, key):
        if isinstance(key, int):
            return list.__getitem__(self, key)
        else:
            vals = []
            val = None
            for iField in range(len(self)):
                if _isParticle(key):
                    val = list.__getitem__(self, iField).eval(key.time, key.depth, key.lat, key.lon, particle=None)
                else:
                    val = list.__getitem__(self, iField).eval(*key)
                vals.append(val)
            return tuple(np.sum(vals, 0)) if isinstance(val, tuple) else np.sum(vals)

    def __add__(self, field):
        if isinstance(field, Field):
            assert isinstance(self[0], type(field)), 'Fields in a SummedField should be either all scalars or all vectors'
            self.append(field)
        elif isinstance(field, SummedField):
            assert isinstance(self[0], type(field[0])), 'Fields in a SummedField should be either all scalars or all vectors'
            for fld in field:
                self.append(fld)
        return self


class NestedField(list):
    """Class NestedField is a list of Fields from which the first one to be not declared out-of-boundaries
    at particle position is interpolated. This induces that the order of the fields in the list matters.
    Each one it its turn, a field is interpolated: if the interpolation succeeds or if an error other
    than `ErrorOutOfBounds` is thrown, the function is stopped. Otherwise, next field is interpolated.
    NestedField returns an `ErrorOutOfBounds` only if last field is as well out of boundaries.
    NestedField is composed of either Fields or VectorFields.

    See `here <https://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_NestedFields.ipynb>`_
    for a detailed tutorial

    :param name: Name of the NestedField
    :param F: List of fields (order matters). F can be a scalar Field, a VectorField, or the zonal component (U) of the VectorField
    :param V: List of fields defining the meridional component of a VectorField, if F is the zonal component. (default: None)
    :param W: List of fields defining the vertical component of a VectorField, if F and V are the zonal and meridional components (default: None)
    """

    def __init__(self, name, F, V=None, W=None):
        if V is None:
            if isinstance(F[0], VectorField):
                vector_type = F[0].vector_type
            for Fi in F:
                assert isinstance(Fi, Field) or (isinstance(Fi, VectorField) and Fi.vector_type == vector_type), 'Components of a NestedField must be Field or VectorField'
                self.append(Fi)
        elif W is None:
            for (i, Fi, Vi) in zip(range(len(F)), F, V):
                assert isinstance(Fi, Field) and isinstance(Vi, Field), \
                    'F, and V components of a NestedField must be Field'
                self.append(VectorField(name+'_%d' % i, Fi, Vi))
        else:
            for (i, Fi, Vi, Wi) in zip(range(len(F)), F, V, W):
                assert isinstance(Fi, Field) and isinstance(Vi, Field) and isinstance(Wi, Field), \
                    'F, V and W components of a NestedField must be Field'
                self.append(VectorField(name+'_%d' % i, Fi, Vi, Wi))
        self.name = name

    def __getitem__(self, key):
        if isinstance(key, int):
            return list.__getitem__(self, key)
        else:
            for iField in range(len(self)):
                try:
                    if _isParticle(key):
                        val = list.__getitem__(self, iField).eval(key.time, key.depth, key.lat, key.lon, particle=None)
                    else:
                        val = list.__getitem__(self, iField).eval(*key)
                    break
                except (FieldOutOfBoundError, FieldSamplingError):
                    if iField == len(self)-1:
                        raise
                    else:
                        pass
            return val

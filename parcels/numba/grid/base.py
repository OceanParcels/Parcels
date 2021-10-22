from enum import IntEnum

import numpy as np

from parcels.tools.converters import TimeConverter
from numba.experimental import jitclass
import numba as nb
from copy import deepcopy
from numba.core.typing.asnumbatype import as_numba_type
from numba import njit
# from parcels.tools.statuscodes import FieldOutOfBoundError,\
    # FieldOutOfBoundSurfaceError, FieldSamplingError


class FieldOutOfBoundError(Exception):
    pass


class FieldOutOfBoundSurfaceError(Exception):
    pass


class FieldSamplingError(Exception):
    pass


class GridCode(IntEnum):
    RectilinearZGrid = 0
    RectilinearSGrid = 1
    CurvilinearZGrid = 2
    CurvilinearSGrid = 3


class GridStatus(IntEnum):
    Updated = 0
    FirstUpdated = 1
    NeedsUpdate = 2


@jitclass(spec=[
    ("x", nb.float32),
    ("y", nb.float32),
    ("z", nb.float32)])
class FOBErrorData():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


_base_spec = [
    ("xi", nb.int32[:]),
    ("yi", nb.int32[:]),
    ("zi", nb.int32[:]),
    ("ti", nb.int32),
    ("time", nb.float64[:]),
    ("time_full", nb.float64[:]),
    ("time_origin", nb.float64),
    ("mesh", nb.types.string),
    ("zonal_periodic", nb.bool_),
    ("zonal_halo", nb.int32),
    ("meridional_halo", nb.int32),
    ("lat_flipped", nb.bool_),
    ("defer_load", nb.bool_),
    ("lonlat_minmax", nb.float32[:]),
    ("periods", nb.int32),
    ("load_chunk", nb.types.ListType(nb.int32)),
    ("chunksize", nb.int32),
    ("_add_last_periodic_data_timestep", nb.bool_),
    ("chunk_info", nb.int32),
    ("xdim", nb.int64),
    ("ydim", nb.int64),
    ("zdim", nb.int64),
    ("tdim", nb.int64),
    ("gtype", nb.types.IntEnumMember(GridCode, nb.int64)),
    ("update_status", nb.types.IntEnumMember(GridStatus, nb.int64)),
    ("fob_data", as_numba_type(FOBErrorData))
]


# @jitclass(spec=_base_spec)
class BaseGrid(object):
    """Grid class that defines a (spatial and temporal) grid on which Fields are defined

    """
    def __init__(self, lon, lat, time, time_origin, mesh):
        self.xi = np.empty(0, dtype=nb.int32)
        self.yi = np.empty(0, dtype=nb.int32)
        self.zi = np.empty(0, dtype=nb.int32)
        self.ti = -1
        self.lon = lon
        self.lat = lat
#         self.time = np.zeros(1, dtype=np.float64) if time is None else time
        self.time = time
        self.time_origin = time_origin
        self.time_full = self.time  # needed for deferred_loaded Fields
#         self.time_origin = TimeConverter() if time_origin is None else time_origin
#         assert isinstance(self.time_origin, TimeConverter), 'time_origin needs to be a TimeConverter object'
        self.mesh = mesh
#         self.cell_edge_sizes = {}
        self.zonal_periodic = False
        self.zonal_halo = 0
        self.meridional_halo = 0
        self.lat_flipped = False
        self.defer_load = False
        self.lonlat_minmax = np.array([np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)], dtype=np.float32)
        self.periods = 0
        self.load_chunk = nb.typed.List([nb.int32(-1)])
        self.chunk_info = -1
        self.chunksize = -1
        self._add_last_periodic_data_timestep = False
#         self.depth_field = None

    def lon_grid_to_target(self):
        if self.lon_remapping:
            self.lon = self.lon_remapping.to_target(self.lon)

    def lon_grid_to_source(self):
        if self.lon_remapping:
            self.lon = self.lon_remapping.to_source(self.lon)

    def lon_particle_to_target(self, lon):
        if self.lon_remapping:
            return self.lon_remapping.particle_to_target(lon)
        return lon

    def advancetime(self, grid_new):
#         assert isinstance(grid_new.time_origin, type(self.time_origin)), 'time_origin of new and old grids must be either both None or both a date'
        if self.time_origin:
            grid_new.time = grid_new.time + self.time_origin.reltime(grid_new.time_origin)
        if len(grid_new.time) != 1:
            raise RuntimeError('New FieldSet needs to have only one snapshot')
        if grid_new.time > self.time[-1]:  # forward in time, so appending at end
            self.time = np.concatenate((self.time[1:], grid_new.time))
            return 1
        elif grid_new.time < self.time[0]:  # backward in time, so prepending at start
            self.time = np.concatenate((grid_new.time, self.time[:-1]))
            return -1
        else:
            raise RuntimeError("Time of field_new in Field.advancetime() overlaps with times in old Field")

    def check_zonal_periodic(self):
        if self.zonal_periodic or self.mesh == 'flat' or self.lon.size == 1:
            return
#         dx = (self.lon[1:] - self.lon[:-1]) if len(self.lon.shape) == 1 else self.lon[0, 1:] - self.lon[0, :-1]
        dx = self.get_dlon()
        dx = np.where(dx < -180, dx+360, dx)
        dx = np.where(dx > 180, dx-360, dx)
        self.zonal_periodic = np.sum(dx) > 359.9

    def add_Sdepth_periodic_halo(self, zonal, meridional, halosize):
        if zonal:
#             if len(self.depth.shape) == 3:
#                 self.depth = np.concatenate((self.depth[:, :, -halosize:], self.depth,
#                                              self.depth[:, :, 0:halosize]), axis=len(self.depth.shape) - 1)
#                 assert self.depth.shape[2] == self.xdim, "Third dim must be x."
#             else:
            self.depth = np.concatenate((self.depth[:, :, :, -halosize:], self.depth,
                                         self.depth[:, :, :, 0:halosize]), axis=len(self.depth.shape) - 1)
            assert self.depth.shape[3] == self.xdim, "Fourth dim must be x."
        if meridional:
#             if len(self.depth.shape) == 3:
#                 self.depth = np.concatenate((self.depth[:, -halosize:, :], self.depth,
#                                              self.depth[:, 0:halosize, :]), axis=len(self.depth.shape) - 2)
#                 assert self.depth.shape[1] == self.ydim, "Second dim must be y."
#             else:
            self.depth = np.concatenate((self.depth[:, :, -halosize:, :], self.depth,
                                         self.depth[:, :, 0:halosize, :]), axis=len(self.depth.shape) - 2)
            assert self.depth.shape[2] == self.ydim, "Third dim must be y."

    def computeTimeChunk(self, f, time, signdt):
        nextTime_loc = np.infty if signdt >= 0 else -np.infty
#         periods = self.periods.value if isinstance(self.periods, c_int) else self.periods
        periods = self.periods
        prev_time_indices = self.time
        if self.update_status == GridStatus.NeedsUpdate:
            if self.ti >= 0:
                if time - periods*(self.time_full[-1]-self.time_full[0]) < self.time[0] or time - periods*(self.time_full[-1]-self.time_full[0]) > self.time[1]:
                    self.ti = -1  # reset
                elif signdt >= 0 and (time - periods*(self.time_full[-1]-self.time_full[0]) < self.time_full[0] or time - periods*(self.time_full[-1]-self.time_full[0]) >= self.time_full[-1]):
                    self.ti = -1  # reset
                elif signdt < 0 and (time - periods*(self.time_full[-1]-self.time_full[0]) <= self.time_full[0] or time - periods*(self.time_full[-1]-self.time_full[0]) > self.time_full[-1]):
                    self.ti = -1  # reset
                elif signdt >= 0 and time - periods*(self.time_full[-1]-self.time_full[0]) >= self.time[1] and self.ti < len(self.time_full)-2:
                    self.ti += 1
                    self.time = self.time_full[self.ti:self.ti+2]
                    self.update_status = GridStatus.Updated
                elif signdt < 0 and time - periods*(self.time_full[-1]-self.time_full[0]) <= self.time[0] and self.ti > 0:
                    self.ti -= 1
                    self.time = self.time_full[self.ti:self.ti+2]
                    self.update_status = GridStatus.Updated
            if self.ti == -1:
                self.time = self.time_full
                self.ti, _ = f.time_index(time)
#                 periods = self.periods.value if isinstance(self.periods, c_int) else self.periods
                periods = self.periods
                if signdt == -1 and self.ti == 0 and (time - periods*(self.time_full[-1]-self.time_full[0])) == self.time[0] and f.time_periodic:
                    self.ti = len(self.time)-1
                    periods -= 1
                if signdt == -1 and self.ti > 0 and self.time_full[self.ti] == time:
                    self.ti -= 1
                if self.ti >= len(self.time_full) - 1:
                    self.ti = len(self.time_full) - 2

                self.time = self.time_full[self.ti:self.ti+2]
                self.tdim = 2
                if prev_time_indices is None or len(prev_time_indices) != 2 or len(prev_time_indices) != len(self.time):
                    self.update_status = GridStatus.FirstUpdated
                elif functools.reduce(lambda i, j: i and j, map(lambda m, k: m == k, self.time, prev_time_indices), True) and len(prev_time_indices) == len(self.time):
                    self.update_status = GridStatus.NeedsUpdate
                elif functools.reduce(lambda i, j: i and j, map(lambda m, k: m == k, self.time[:1], prev_time_indices[:1]), True) and len(prev_time_indices) == len(self.time):
                    self.update_status = GridStatus.Updated
                else:
                    self.update_status = GridStatus.FirstUpdated
            if signdt >= 0 and (self.ti < len(self.time_full)-2 or not f.allow_time_extrapolation):
                nextTime_loc = self.time[1] + periods*(self.time_full[-1]-self.time_full[0])
            elif signdt < 0 and (self.ti > 0 or not f.allow_time_extrapolation):
                nextTime_loc = self.time[0] + periods*(self.time_full[-1]-self.time_full[0])
        return nextTime_loc

    def FieldOutOfBoundError(self, x, y, z):
        self.fob_data = FOBErrorData(x, y, z)
        raise FieldOutOfBoundError()
#     FieldOutOfBoundError(x, y, z)

    def FieldOutOfBoundSurfaceError(self, x, y, z):
        self.fob_data = FOBErrorData(x, y, z)
        raise FieldOutOfBoundSurfaceError()

    def FieldSamplingError(self, x, y, z):
        self.fob_data = FOBErrorData(x, y, z)
        raise FieldSamplingError()

    @property
    def chunk_not_loaded(self):
        return 0

    @property
    def chunk_loading_requested(self):
        return 1

    @property
    def chunk_loaded_touched(self):
        return 2

    @property
    def chunk_deprecated(self):
        return 3

    @property
    def chunk_loaded(self):
        return [2, 3]

    # def search_indices(self, x, y, z, ti=-1, time=-1, search2D=False):
    #     if self.gtype in [GridCode.RectilinearSGrid, GridCode.RectilinearZGrid]:
    #         return self.search_indices_rectilinear(x, y, z, ti, time, search2D=search2D)
    #     else:
    #         return self.search_indices_curvilinear(x, y, z, ti, time, search2D=search2D)


# @jitclass(spec=_base_spec)

import numpy as np

from numba.experimental import jitclass
import numba as nb
from numba.core.typing.asnumbatype import as_numba_type

from .statuscodes import FieldOutOfBoundError, FieldOutOfBoundSurfaceError
from .statuscodes import FieldSamplingError
from .statuscodes import GridCode, GridStatus


@jitclass(spec=[
    ("x", nb.float32),
    ("y", nb.float32),
    ("z", nb.float32)])
class FOBErrorData():
    """"Object for storing data related to Field errors."""
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def _base_grid_spec():
    """Grid specifications that are shared between different types

    lat, lon, depth need to be added by the specific implementations.
    """
    return [
        ("xi", nb.types.DictType(nb.int64, nb.int64)),
        ("yi", nb.types.DictType(nb.int64, nb.int64)),
        ("zi", nb.types.DictType(nb.int64, nb.int64)),
        ("ti", nb.int32),
        ("time", nb.float64[:]),
        ("time_full", nb.float64[:]),
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
        ("z4d", nb.bool_),
        ("gtype", nb.types.IntEnumMember(GridCode, nb.int64)),
        ("update_status", nb.types.IntEnumMember(GridStatus, nb.int64)),
        ("fob_data", as_numba_type(FOBErrorData)),
        ("depth_field", nb.float32[:]),
    ]


class BaseGrid(object):
    """Grid class that defines a (spatial and temporal) grid on which Fields are defined

    Base class derived by RectilinearGrid, etc. Only then the class should be
    compiled/jitclass(ed).
    """
    def __init__(self, lon, lat, time, mesh):
        self.xi = nb.typed.Dict.empty(nb.int64, nb.int64)
        self.yi = nb.typed.Dict.empty(nb.int64, nb.int64)
        self.zi = nb.typed.Dict.empty(nb.int64, nb.int64)
        self.ti = -1
        self.lon = lon.astype(nb.float32)
        self.lat = lat.astype(nb.float32)
        self.time = np.zeros(1, dtype=nb.float64) if time is None else time.astype(nb.float64)
        self.time_full = self.time  # needed for deferred_loaded Fields
        self.mesh = mesh
        self.zonal_periodic = False
        self.zonal_halo = 0
        self.meridional_halo = 0
        self.lat_flipped = False
        self.defer_load = False
        self.lonlat_minmax = np.array(
            [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)],
            dtype=np.float32)
        self.periods = 0
        self.load_chunk = nb.typed.List([nb.int32(-1)])
        self.chunk_info = -1
        self.chunksize = -1
        self._add_last_periodic_data_timestep = False
        self.depth_field = np.empty(0, dtype=nb.float32)

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
        dx = self.get_dlon()
        dx = np.where(dx < -180, dx+360, dx)
        dx = np.where(dx > 180, dx-360, dx)
        self.zonal_periodic = np.sum(dx) > 359.9

    def add_Sdepth_periodic_halo(self, zonal, meridional, halosize):
        if zonal:
            self.depth = np.concatenate(
                (self.depth[:, :, :, -halosize:], self.depth,
                 self.depth[:, :, :, 0:halosize]), axis=len(self.depth.shape)-1)
            assert self.depth.shape[3] == self.xdim, "Fourth dim must be x."
        if meridional:
            self.depth = np.concatenate(
                (self.depth[:, :, -halosize:, :], self.depth,
                 self.depth[:, :, 0:halosize, :]), axis=len(self.depth.shape)-2)
            assert self.depth.shape[2] == self.ydim, "Third dim must be y."

    def computeTimeChunk(self, f, time, signdt):
        raise NotImplementedError

    def FieldOutOfBoundError(self, x, y, z):
        self.fob_data = FOBErrorData(x, y, z)
        raise FieldOutOfBoundError()

    def FieldOutOfBoundSurfaceError(self, x, y, z):
        self.fob_data = FOBErrorData(x, y, z)
        raise FieldOutOfBoundSurfaceError()

    def FieldSamplingError(self, x, y, z):
        self.fob_data = FOBErrorData(x, y, z)
        raise FieldSamplingError()

    # I'm not 100% sure if this part works and to what extent,
    # but it relates to having chunked memory, and loading small chunks
    # instead of everything at once.
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

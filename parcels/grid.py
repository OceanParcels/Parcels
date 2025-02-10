import functools
import warnings
from ctypes import POINTER, Structure, c_double, c_float, c_int, c_void_p, cast, pointer
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from parcels._typing import Mesh, UpdateStatus, assert_valid_mesh
from parcels.tools._helpers import deprecated_made_private
from parcels.tools.converters import Geographic, GeographicPolar, TimeConverter, UnitConverter
from parcels.tools.warnings import FieldSetWarning

__all__ = [
    "CGrid",
    "CurvilinearSGrid",
    "CurvilinearZGrid",
    "Grid",
    "GridCode",
    "GridType",
    "RectilinearSGrid",
    "RectilinearZGrid",
]


class GridType(IntEnum):
    RectilinearZGrid = 0
    RectilinearSGrid = 1
    CurvilinearZGrid = 2
    CurvilinearSGrid = 3


# GridCode has been renamed to GridType for consistency.
# TODO: Remove alias in Parcels v4
GridCode = GridType


class CGrid(Structure):
    _fields_ = [("gtype", c_int), ("grid", c_void_p)]


class Grid:
    """Grid class that defines a (spatial and temporal) grid on which Fields are defined."""

    def __init__(
        self,
        lon: npt.NDArray,
        lat: npt.NDArray,
        time: npt.NDArray | None,
        time_origin: TimeConverter | None,
        mesh: Mesh,
    ):
        self._ti = -1
        self._update_status: UpdateStatus | None = None
        if not lon.flags["C_CONTIGUOUS"]:
            lon = np.array(lon, order="C")
        if not lat.flags["C_CONTIGUOUS"]:
            lat = np.array(lat, order="C")
        time = np.zeros(1, dtype=np.float64) if time is None else time
        if not time.flags["C_CONTIGUOUS"]:
            time = np.array(time, order="C")
        if not lon.dtype == np.float32:
            lon = lon.astype(np.float32)
        if not lat.dtype == np.float32:
            lat = lat.astype(np.float32)
        if not time.dtype == np.float64:
            assert isinstance(
                time[0], (np.integer, np.floating, float, int)
            ), "Time vector must be an array of int or floats"
            time = time.astype(np.float64)

        self._lon = lon
        self._lat = lat
        self.time = time
        self.time_full = self.time  # needed for deferred_loaded Fields
        self._time_origin = TimeConverter() if time_origin is None else time_origin
        assert isinstance(self.time_origin, TimeConverter), "time_origin needs to be a TimeConverter object"
        assert_valid_mesh(mesh)
        self._mesh = mesh
        self._cstruct = None
        self._cell_edge_sizes: dict[str, npt.NDArray] = {}
        self._zonal_periodic = False
        self._zonal_halo = 0
        self._meridional_halo = 0
        self._lat_flipped = False
        self._defer_load = False
        self._lonlat_minmax = np.array(
            [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)], dtype=np.float32
        )
        self.periods = 0
        self._load_chunk: npt.NDArray = np.array([])
        self.chunk_info = None
        self.chunksize = None
        self._add_last_periodic_data_timestep = False
        self.depth_field = None

    def __repr__(self):
        with np.printoptions(threshold=5, suppress=True, linewidth=120, formatter={"float": "{: 0.2f}".format}):
            return (
                f"{type(self).__name__}("
                f"lon={self.lon!r}, lat={self.lat!r}, time={self.time!r}, "
                f"time_origin={self.time_origin!r}, mesh={self.mesh!r})"
            )

    @property
    def lon(self):
        return self._lon

    @property
    def lat(self):
        return self._lat

    @property
    def depth(self):
        return self._depth

    def negate_depth(self):
        """Method to flip the sign of the depth dimension of a Grid.
        Note that this method does _not_ change the direction of the vertical velocity;
        for that users need to add a fieldset.W.set_scaling_factor(-1.0)
        """
        self._depth = -self._depth

    @property
    def mesh(self):
        return self._mesh

    @property
    def meridional_halo(self):
        return self._meridional_halo

    @property
    def lonlat_minmax(self):
        return self._lonlat_minmax

    @property
    def time_origin(self):
        return self._time_origin

    @property
    def zonal_periodic(self):
        return self._zonal_periodic

    @property
    def zonal_halo(self):
        return self._zonal_halo

    @property
    def defer_load(self):
        return self._defer_load

    @property
    def cell_edge_sizes(self):
        return self._cell_edge_sizes

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def ti(self):
        return self._ti

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def cstruct(self):
        return self._cstruct

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def lat_flipped(self):
        return self._lat_flipped

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def cgrid(self):
        return self._cgrid

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def gtype(self):
        return self._gtype

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def z4d(self):
        return self._z4d

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def update_status(self):
        return self._update_status

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def load_chunk(self):
        return self._load_chunk

    @staticmethod
    def create_grid(
        lon: npt.ArrayLike,
        lat: npt.ArrayLike,
        depth,
        time,
        time_origin,
        mesh: Mesh,
        **kwargs,
    ):
        lon = np.array(lon)
        lat = np.array(lat)

        if depth is not None:
            depth = np.array(depth)

        if len(lon.shape) <= 1:
            if depth is None or len(depth.shape) <= 1:
                return RectilinearZGrid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh, **kwargs)
            else:
                return RectilinearSGrid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh, **kwargs)
        else:
            if depth is None or len(depth.shape) <= 1:
                return CurvilinearZGrid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh, **kwargs)
            else:
                return CurvilinearSGrid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh, **kwargs)

    @property
    def ctypes_struct(self):
        # This is unnecessary for the moment, but it could be useful when going will fully unstructured grids
        self._cgrid = cast(pointer(self._child_ctypes_struct), c_void_p)
        cstruct = CGrid(self._gtype, self._cgrid.value)
        return cstruct

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def child_ctypes_struct(self):
        return self._child_ctypes_struct

    @property
    def _child_ctypes_struct(self):
        """Returns a ctypes struct object containing all relevant
        pointers and sizes for this grid.
        """

        class CStructuredGrid(Structure):
            # z4d is only to have same cstruct as RectilinearSGrid
            _fields_ = [
                ("xdim", c_int),
                ("ydim", c_int),
                ("zdim", c_int),
                ("tdim", c_int),
                ("z4d", c_int),
                ("mesh_spherical", c_int),
                ("zonal_periodic", c_int),
                ("chunk_info", POINTER(c_int)),
                ("load_chunk", POINTER(c_int)),
                ("tfull_min", c_double),
                ("tfull_max", c_double),
                ("periods", POINTER(c_int)),
                ("lonlat_minmax", POINTER(c_float)),
                ("lon", POINTER(c_float)),
                ("lat", POINTER(c_float)),
                ("depth", POINTER(c_float)),
                ("time", POINTER(c_double)),
            ]

        # Create and populate the c-struct object
        if not self._cstruct:  # Not to point to the same grid various times if grid in various fields
            if not isinstance(self.periods, c_int):
                self.periods = c_int()
                self.periods.value = 0
            self._cstruct = CStructuredGrid(
                self.xdim,
                self.ydim,
                self.zdim,
                self.tdim,
                self._z4d,
                int(self.mesh == "spherical"),
                int(self.zonal_periodic),
                (c_int * len(self.chunk_info))(*self.chunk_info),
                self._load_chunk.ctypes.data_as(POINTER(c_int)),
                self.time_full[0],
                self.time_full[-1],
                pointer(self.periods),
                self.lonlat_minmax.ctypes.data_as(POINTER(c_float)),
                self.lon.ctypes.data_as(POINTER(c_float)),
                self.lat.ctypes.data_as(POINTER(c_float)),
                self.depth.ctypes.data_as(POINTER(c_float)),
                self.time.ctypes.data_as(POINTER(c_double)),
            )
        return self._cstruct

    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def check_zonal_periodic(self, *args, **kwargs):
        return self._check_zonal_periodic(*args, **kwargs)

    def _check_zonal_periodic(self):
        if self.zonal_periodic or self.mesh == "flat" or self.lon.size == 1:
            return
        dx = (self.lon[1:] - self.lon[:-1]) if len(self.lon.shape) == 1 else self.lon[0, 1:] - self.lon[0, :-1]
        dx = np.where(dx < -180, dx + 360, dx)
        dx = np.where(dx > 180, dx - 360, dx)
        self._zonal_periodic = sum(dx) > 359.9

    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def add_Sdepth_periodic_halo(self, *args, **kwargs):
        return self._add_Sdepth_periodic_halo(*args, **kwargs)

    def _add_Sdepth_periodic_halo(self, zonal, meridional, halosize):
        if zonal:
            if len(self.depth.shape) == 3:
                self._depth = np.concatenate(
                    (self.depth[:, :, -halosize:], self.depth, self.depth[:, :, 0:halosize]),
                    axis=len(self.depth.shape) - 1,
                )
                assert self.depth.shape[2] == self.xdim, "Third dim must be x."
            else:
                self._depth = np.concatenate(
                    (self.depth[:, :, :, -halosize:], self.depth, self.depth[:, :, :, 0:halosize]),
                    axis=len(self.depth.shape) - 1,
                )
                assert self.depth.shape[3] == self.xdim, "Fourth dim must be x."
        if meridional:
            if len(self.depth.shape) == 3:
                self._depth = np.concatenate(
                    (self.depth[:, -halosize:, :], self.depth, self.depth[:, 0:halosize, :]),
                    axis=len(self.depth.shape) - 2,
                )
                assert self.depth.shape[1] == self.ydim, "Second dim must be y."
            else:
                self._depth = np.concatenate(
                    (self.depth[:, :, -halosize:, :], self.depth, self.depth[:, :, 0:halosize, :]),
                    axis=len(self.depth.shape) - 2,
                )
                assert self.depth.shape[2] == self.ydim, "Third dim must be y."

    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def computeTimeChunk(self, *args, **kwargs):
        return self._computeTimeChunk(*args, **kwargs)

    def _computeTimeChunk(self, f, time, signdt):
        nextTime_loc = np.inf if signdt >= 0 else -np.inf
        periods = self.periods.value if isinstance(self.periods, c_int) else self.periods
        prev_time_indices = self.time
        if self._update_status == "not_updated":
            if self._ti >= 0:
                if (
                    time - periods * (self.time_full[-1] - self.time_full[0]) < self.time[0]
                    or time - periods * (self.time_full[-1] - self.time_full[0]) > self.time[1]
                ):
                    self._ti = -1  # reset
                elif signdt >= 0 and (
                    time - periods * (self.time_full[-1] - self.time_full[0]) < self.time_full[0]
                    or time - periods * (self.time_full[-1] - self.time_full[0]) >= self.time_full[-1]
                ):
                    self._ti = -1  # reset
                elif signdt < 0 and (
                    time - periods * (self.time_full[-1] - self.time_full[0]) <= self.time_full[0]
                    or time - periods * (self.time_full[-1] - self.time_full[0]) > self.time_full[-1]
                ):
                    self._ti = -1  # reset
                elif (
                    signdt >= 0
                    and time - periods * (self.time_full[-1] - self.time_full[0]) >= self.time[1]
                    and self._ti < len(self.time_full) - 2
                ):
                    self._ti += 1
                    self.time = self.time_full[self._ti : self._ti + 2]
                    self._update_status = "updated"
                elif (
                    signdt < 0
                    and time - periods * (self.time_full[-1] - self.time_full[0]) <= self.time[0]
                    and self._ti > 0
                ):
                    self._ti -= 1
                    self.time = self.time_full[self._ti : self._ti + 2]
                    self._update_status = "updated"
            if self._ti == -1:
                self.time = self.time_full
                self._ti, _ = f._time_index(time)
                periods = self.periods.value if isinstance(self.periods, c_int) else self.periods
                if (
                    signdt == -1
                    and self._ti == 0
                    and (time - periods * (self.time_full[-1] - self.time_full[0])) == self.time[0]
                    and f.time_periodic
                ):
                    self._ti = len(self.time) - 1
                    periods -= 1
                if signdt == -1 and self._ti > 0 and self.time_full[self._ti] == time:
                    self._ti -= 1
                if self._ti >= len(self.time_full) - 1:
                    self._ti = len(self.time_full) - 2

                self.time = self.time_full[self._ti : self._ti + 2]
                self.tdim = 2
                if prev_time_indices is None or len(prev_time_indices) != 2 or len(prev_time_indices) != len(self.time):
                    self._update_status = "first_updated"
                elif functools.reduce(
                    lambda i, j: i and j, map(lambda m, k: m == k, self.time, prev_time_indices), True
                ) and len(prev_time_indices) == len(self.time):
                    self._update_status = "not_updated"
                elif functools.reduce(
                    lambda i, j: i and j, map(lambda m, k: m == k, self.time[:1], prev_time_indices[:1]), True
                ) and len(prev_time_indices) == len(self.time):
                    self._update_status = "updated"
                else:
                    self._update_status = "first_updated"
            if signdt >= 0 and (self._ti < len(self.time_full) - 2 or not f.allow_time_extrapolation):
                nextTime_loc = self.time[1] + periods * (self.time_full[-1] - self.time_full[0])
            elif signdt < 0 and (self._ti > 0 or not f.allow_time_extrapolation):
                nextTime_loc = self.time[0] + periods * (self.time_full[-1] - self.time_full[0])
        return nextTime_loc

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def chunk_not_loaded(self):
        return self._chunk_not_loaded

    @property
    def _chunk_not_loaded(self):
        return 0

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def chunk_loading_requested(self):
        return self._chunk_loading_requested

    @property
    def _chunk_loading_requested(self):
        return 1

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def chunk_loaded_touched(self):
        return self._chunk_loaded_touched

    @property
    def _chunk_loaded_touched(self):
        return 2

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def chunk_deprecated(self):
        return self._chunk_deprecated

    @property
    def _chunk_deprecated(self):
        return 3

    @property
    @deprecated_made_private  # TODO: Remove 6 months after v3.1.0
    def chunk_loaded(self):
        return self._chunk_loaded

    @property
    def _chunk_loaded(self):
        return [2, 3]


class RectilinearGrid(Grid):
    """Rectilinear Grid class

    Private base class for RectilinearZGrid and RectilinearSGrid

    """

    def __init__(self, lon, lat, time, time_origin, mesh: Mesh):
        assert isinstance(lon, np.ndarray) and len(lon.shape) <= 1, "lon is not a numpy vector"
        assert isinstance(lat, np.ndarray) and len(lat.shape) <= 1, "lat is not a numpy vector"
        assert isinstance(time, np.ndarray) or not time, "time is not a numpy array"
        if isinstance(time, np.ndarray):
            assert len(time.shape) == 1, "time is not a vector"

        super().__init__(lon, lat, time, time_origin, mesh)
        self.tdim = self.time.size

        if self.ydim > 1 and self.lat[-1] < self.lat[0]:
            self._lat = np.flip(self.lat, axis=0)
            self._lat_flipped = True
            warnings.warn(
                "Flipping lat data from North-South to South-North. "
                "Note that this may lead to wrong sign for meridional velocity, so tread very carefully",
                FieldSetWarning,
                stacklevel=2,
            )

    @property
    def xdim(self):
        return self.lon.size

    @property
    def ydim(self):
        return self.lat.size

    def add_periodic_halo(self, zonal: bool, meridional: bool, halosize: int = 5):
        """Add a 'halo' to the Grid, through extending the Grid (and lon/lat)
        similarly to the halo created for the Fields

        Parameters
        ----------
        zonal : bool
            Create a halo in zonal direction
        meridional : bool
            Create a halo in meridional direction
        halosize : int
            size of the halo (in grid points). Default is 5 grid points
        """
        if zonal:
            lonshift = self.lon[-1] - 2 * self.lon[0] + self.lon[1]
            if not np.allclose(self.lon[1] - self.lon[0], self.lon[-1] - self.lon[-2]):
                warnings.warn(
                    "The zonal halo is located at the east and west of current grid, "
                    "with a dx = lon[1]-lon[0] between the last nodes of the original grid and the first ones of the halo. "
                    "In your grid, lon[1]-lon[0] != lon[-1]-lon[-2]. Is the halo computed as you expect?",
                    FieldSetWarning,
                    stacklevel=2,
                )
            self._lon = np.concatenate((self.lon[-halosize:] - lonshift, self.lon, self.lon[0:halosize] + lonshift))
            self._zonal_periodic = True
            self._zonal_halo = halosize
        if meridional:
            if not np.allclose(self.lat[1] - self.lat[0], self.lat[-1] - self.lat[-2]):
                warnings.warn(
                    "The meridional halo is located at the north and south of current grid, "
                    "with a dy = lat[1]-lat[0] between the last nodes of the original grid and the first ones of the halo. "
                    "In your grid, lat[1]-lat[0] != lat[-1]-lat[-2]. Is the halo computed as you expect?",
                    FieldSetWarning,
                    stacklevel=2,
                )
            latshift = self.lat[-1] - 2 * self.lat[0] + self.lat[1]
            self._lat = np.concatenate((self.lat[-halosize:] - latshift, self.lat, self.lat[0:halosize] + latshift))
            self._meridional_halo = halosize
        self._lonlat_minmax = np.array(
            [np.nanmin(self.lon), np.nanmax(self.lon), np.nanmin(self.lat), np.nanmax(self.lat)], dtype=np.float32
        )
        if isinstance(self, RectilinearSGrid):
            self._add_Sdepth_periodic_halo(zonal, meridional, halosize)


class RectilinearZGrid(RectilinearGrid):
    """Rectilinear Z Grid.

    Parameters
    ----------
    lon :
        Vector containing the longitude coordinates of the grid
    lat :
        Vector containing the latitude coordinates of the grid
    depth :
        Vector containing the vertical coordinates of the grid, which are z-coordinates.
        The depth of the different layers is thus constant.
    time :
        Vector containing the time coordinates of the grid
    time_origin : parcels.tools.converters.TimeConverter
        Time origin of the time axis
    mesh : str
        String indicating the type of mesh coordinates and
        units used during velocity interpolation:

        1. spherical (default): Lat and lon in degree, with a
           correction for zonal velocity U near the poles.
        2. flat: No conversion, lat/lon are assumed to be in m.
    """

    def __init__(self, lon, lat, depth=None, time=None, time_origin=None, mesh: Mesh = "flat"):
        super().__init__(lon, lat, time, time_origin, mesh)
        if isinstance(depth, np.ndarray):
            assert len(depth.shape) <= 1, "depth is not a vector"

        self._gtype = GridType.RectilinearZGrid
        self._depth = np.zeros(1, dtype=np.float32) if depth is None else depth
        if not self.depth.flags["C_CONTIGUOUS"]:
            self._depth = np.array(self.depth, order="C")
        self._z4d = -1  # only used in RectilinearSGrid
        if not self.depth.dtype == np.float32:
            self._depth = self.depth.astype(np.float32)

    @property
    def zdim(self):
        return self.depth.size


class RectilinearSGrid(RectilinearGrid):
    """Rectilinear S Grid. Same horizontal discretisation as a rectilinear z grid,
       but with s vertical coordinates

    Parameters
    ----------
    lon :
        Vector containing the longitude coordinates of the grid
    lat :
        Vector containing the latitude coordinates of the grid
    depth :
        4D (time-evolving) or 3D (time-independent) array containing the vertical coordinates of the grid,
        which are s-coordinates.
        s-coordinates can be terrain-following (sigma) or iso-density (rho) layers,
        or any generalised vertical discretisation.
        The depth of each node depends then on the horizontal position (lon, lat),
        the number of the layer and the time is depth is a 4D array.
        depth array is either a 4D array[xdim][ydim][zdim][tdim] or a 3D array[xdim][ydim[zdim].
    time :
        Vector containing the time coordinates of the grid
    time_origin : parcels.tools.converters.TimeConverter
        Time origin of the time axis
    mesh : str
        String indicating the type of mesh coordinates and
        units used during velocity interpolation:

        1. spherical (default): Lat and lon in degree, with a
           correction for zonal velocity U near the poles.
        2. flat: No conversion, lat/lon are assumed to be in m.
    """

    def __init__(
        self,
        lon: npt.NDArray,
        lat: npt.NDArray,
        depth: npt.NDArray,
        time: npt.NDArray | None = None,
        time_origin: TimeConverter | None = None,
        mesh: Mesh = "flat",
    ):
        super().__init__(lon, lat, time, time_origin, mesh)
        assert isinstance(depth, np.ndarray) and len(depth.shape) in [3, 4], "depth is not a 3D or 4D numpy array"

        self._gtype = GridType.RectilinearSGrid
        self._depth = depth
        if not self.depth.flags["C_CONTIGUOUS"]:
            self._depth = np.array(self.depth, order="C")
        self._z4d = 1 if len(self.depth.shape) == 4 else 0
        if self._z4d:
            # self.depth.shape[0] is 0 for S grids loaded from netcdf file
            assert (
                self.tdim == self.depth.shape[0] or self.depth.shape[0] == 0
            ), "depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]"
            assert (
                self.xdim == self.depth.shape[-1] or self.depth.shape[-1] == 0
            ), "depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]"
            assert (
                self.ydim == self.depth.shape[-2] or self.depth.shape[-2] == 0
            ), "depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]"
        else:
            assert (
                self.xdim == self.depth.shape[-1]
            ), "depth dimension has the wrong format. It should be [zdim, ydim, xdim]"
            assert (
                self.ydim == self.depth.shape[-2]
            ), "depth dimension has the wrong format. It should be [zdim, ydim, xdim]"
        if not self.depth.dtype == np.float32:
            self._depth = self.depth.astype(np.float32)
        if self._lat_flipped:
            self._depth = np.flip(self.depth, axis=-2)

    @property
    def zdim(self):
        return self.depth.shape[-3]


class CurvilinearGrid(Grid):
    def __init__(
        self,
        lon: npt.NDArray,
        lat: npt.NDArray,
        time: npt.NDArray | None = None,
        time_origin: TimeConverter | None = None,
        mesh: Mesh = "flat",
    ):
        assert isinstance(lon, np.ndarray) and len(lon.squeeze().shape) == 2, "lon is not a 2D numpy array"
        assert isinstance(lat, np.ndarray) and len(lat.squeeze().shape) == 2, "lat is not a 2D numpy array"
        assert isinstance(time, np.ndarray) or not time, "time is not a numpy array"
        if isinstance(time, np.ndarray):
            assert len(time.shape) == 1, "time is not a vector"

        lon = lon.squeeze()
        lat = lat.squeeze()
        super().__init__(lon, lat, time, time_origin, mesh)
        self.tdim = self.time.size

    @property
    def xdim(self):
        return self.lon.shape[1]

    @property
    def ydim(self):
        return self.lon.shape[0]

    def add_periodic_halo(self, zonal, meridional, halosize=5):
        """Add a 'halo' to the Grid, through extending the Grid (and lon/lat)
        similarly to the halo created for the Fields

        Parameters
        ----------
        zonal : bool
            Create a halo in zonal direction
        meridional : bool
            Create a halo in meridional direction
        halosize : int
            size of the halo (in grid points). Default is 5 grid points
        """
        raise NotImplementedError(
            "CurvilinearGrid does not support add_periodic_halo. See https://github.com/OceanParcels/Parcels/pull/1811"
        )


class CurvilinearZGrid(CurvilinearGrid):
    """Curvilinear Z Grid.

    Parameters
    ----------
    lon :
        2D array containing the longitude coordinates of the grid
    lat :
        2D array containing the latitude coordinates of the grid
    depth :
        Vector containing the vertical coordinates of the grid, which are z-coordinates.
        The depth of the different layers is thus constant.
    time :
        Vector containing the time coordinates of the grid
    time_origin : parcels.tools.converters.TimeConverter
        Time origin of the time axis
    mesh : str
        String indicating the type of mesh coordinates and
        units used during velocity interpolation:

        1. spherical (default): Lat and lon in degree, with a
           correction for zonal velocity U near the poles.
        2. flat: No conversion, lat/lon are assumed to be in m.
    """

    def __init__(
        self,
        lon: npt.NDArray,
        lat: npt.NDArray,
        depth: npt.NDArray | None = None,
        time: npt.NDArray | None = None,
        time_origin: TimeConverter | None = None,
        mesh: Mesh = "flat",
    ):
        super().__init__(lon, lat, time, time_origin, mesh)
        if isinstance(depth, np.ndarray):
            assert len(depth.shape) == 1, "depth is not a vector"

        self._gtype = GridType.CurvilinearZGrid
        self._depth = np.zeros(1, dtype=np.float32) if depth is None else depth
        if not self.depth.flags["C_CONTIGUOUS"]:
            self._depth = np.array(self.depth, order="C")
        self._z4d = -1  # only for SGrid
        if not self.depth.dtype == np.float32:
            self._depth = self.depth.astype(np.float32)

    @property
    def zdim(self):
        return self.depth.size


class CurvilinearSGrid(CurvilinearGrid):
    """Curvilinear S Grid.

    Parameters
    ----------
    lon :
        2D array containing the longitude coordinates of the grid
    lat :
        2D array containing the latitude coordinates of the grid
    depth :
        4D (time-evolving) or 3D (time-independent) array containing the vertical coordinates of the grid,
        which are s-coordinates.
        s-coordinates can be terrain-following (sigma) or iso-density (rho) layers,
        or any generalised vertical discretisation.
        The depth of each node depends then on the horizontal position (lon, lat),
        the number of the layer and the time is depth is a 4D array.
        depth array is either a 4D array[xdim][ydim][zdim][tdim] or a 3D array[xdim][ydim[zdim].
    time :
        Vector containing the time coordinates of the grid
    time_origin : parcels.tools.converters.TimeConverter
        Time origin of the time axis
    mesh : str
        String indicating the type of mesh coordinates and
        units used during velocity interpolation:

        1. spherical (default): Lat and lon in degree, with a
           correction for zonal velocity U near the poles.
        2. flat: No conversion, lat/lon are assumed to be in m.
    """

    def __init__(
        self,
        lon: npt.NDArray,
        lat: npt.NDArray,
        depth: npt.NDArray,
        time: npt.NDArray | None = None,
        time_origin: TimeConverter | None = None,
        mesh: Mesh = "flat",
    ):
        super().__init__(lon, lat, time, time_origin, mesh)
        assert isinstance(depth, np.ndarray) and len(depth.shape) in [3, 4], "depth is not a 4D numpy array"

        self._gtype = GridType.CurvilinearSGrid
        self._depth = depth  # should be a C-contiguous array of floats
        if not self.depth.flags["C_CONTIGUOUS"]:
            self._depth = np.array(self.depth, order="C")
        self._z4d = 1 if len(self.depth.shape) == 4 else 0
        if self._z4d:
            # self.depth.shape[0] is 0 for S grids loaded from netcdf file
            assert (
                self.tdim == self.depth.shape[0] or self.depth.shape[0] == 0
            ), "depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]"
            assert (
                self.xdim == self.depth.shape[-1] or self.depth.shape[-1] == 0
            ), "depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]"
            assert (
                self.ydim == self.depth.shape[-2] or self.depth.shape[-2] == 0
            ), "depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]"
        else:
            assert (
                self.xdim == self.depth.shape[-1]
            ), "depth dimension has the wrong format. It should be [zdim, ydim, xdim]"
            assert (
                self.ydim == self.depth.shape[-2]
            ), "depth dimension has the wrong format. It should be [zdim, ydim, xdim]"
        if not self.depth.dtype == np.float32:
            self._depth = self.depth.astype(np.float32)

    @property
    def zdim(self):
        return self.depth.shape[-3]


def _calc_cell_edge_sizes(grid: RectilinearGrid) -> None:
    """Method to calculate cell sizes based on numpy.gradient method.

    Currently only works for Rectilinear Grids. Operates in place adding a `cell_edge_sizes`
    attribute to the grid.
    """
    if not grid.cell_edge_sizes:
        if grid._gtype in (GridType.RectilinearZGrid, GridType.RectilinearSGrid):  # type: ignore[attr-defined]
            grid.cell_edge_sizes["x"] = np.zeros((grid.ydim, grid.xdim), dtype=np.float32)
            grid.cell_edge_sizes["y"] = np.zeros((grid.ydim, grid.xdim), dtype=np.float32)

            x_conv = GeographicPolar() if grid.mesh == "spherical" else UnitConverter()
            y_conv = Geographic() if grid.mesh == "spherical" else UnitConverter()
            for y, (lat, dy) in enumerate(zip(grid.lat, np.gradient(grid.lat), strict=False)):
                for x, (lon, dx) in enumerate(zip(grid.lon, np.gradient(grid.lon), strict=False)):
                    grid.cell_edge_sizes["x"][y, x] = x_conv.to_source(dx, grid.depth[0], lat, lon)
                    grid.cell_edge_sizes["y"][y, x] = y_conv.to_source(dy, grid.depth[0], lat, lon)
        else:
            raise ValueError(
                f"_cell_edge_sizes() not implemented for {grid._gtype} grids. "  # type: ignore[attr-defined]
                "You can provide Field.grid.cell_edge_sizes yourself by in, e.g., "
                "NEMO using the e1u fields etc from the mesh_mask.nc file."
            )


def _calc_cell_areas(grid: RectilinearGrid) -> np.ndarray:
    if not grid.cell_edge_sizes:
        _calc_cell_edge_sizes(grid)
    return grid.cell_edge_sizes["x"] * grid.cell_edge_sizes["y"]

import functools
import warnings
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from parcels._typing import Mesh, UpdateStatus, assert_valid_mesh
from parcels.tools.converters import TimeConverter
from parcels.tools.warnings import FieldSetWarning

__all__ = [
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
        lon = np.array(lon)
        lat = np.array(lat)
        time = np.zeros(1, dtype=np.float64) if time is None else time
        time = np.array(time)
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
        self._zonal_periodic = False
        self._zonal_halo = 0
        self._meridional_halo = 0
        self._lat_flipped = False
        self._defer_load = False
        self._lonlat_minmax = np.array(
            [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)], dtype=np.float32
        )
        self._load_chunk: npt.NDArray = np.array([])
        self.chunk_info = None
        self.chunksize = None
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

    def _check_zonal_periodic(self):
        if self.zonal_periodic or self.mesh == "flat" or self.lon.size == 1:
            return
        dx = (self.lon[1:] - self.lon[:-1]) if len(self.lon.shape) == 1 else self.lon[0, 1:] - self.lon[0, :-1]
        dx = np.where(dx < -180, dx + 360, dx)
        dx = np.where(dx > 180, dx - 360, dx)
        self._zonal_periodic = sum(dx) > 359.9

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

    def _computeTimeChunk(self, f, time, signdt):
        nextTime_loc = np.inf if signdt >= 0 else -np.inf
        prev_time_indices = self.time
        if self._update_status == "not_updated":
            if self._ti >= 0:
                if time < self.time[0] or time > self.time[1]:
                    self._ti = -1  # reset
                elif signdt >= 0 and (time < self.time_full[0] or time >= self.time_full[-1]):
                    self._ti = -1  # reset
                elif signdt < 0 and (time <= self.time_full[0] or time > self.time_full[-1]):
                    self._ti = -1  # reset
                elif signdt >= 0 and time >= self.time[1] and self._ti < len(self.time_full) - 2:
                    self._ti += 1
                    self.time = self.time_full[self._ti : self._ti + 2]
                    self._update_status = "updated"
                elif signdt < 0 and time <= self.time[0] and self._ti > 0:
                    self._ti -= 1
                    self.time = self.time_full[self._ti : self._ti + 2]
                    self._update_status = "updated"
            if self._ti == -1:
                self.time = self.time_full
                self._ti = f._time_index(time)

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
                nextTime_loc = self.time[1]
            elif signdt < 0 and (self._ti > 0 or not f.allow_time_extrapolation):
                nextTime_loc = self.time[0]
        return nextTime_loc

    @property
    def _chunk_not_loaded(self):
        return 0

    @property
    def _chunk_loading_requested(self):
        return 1

    @property
    def _chunk_loaded_touched(self):
        return 2

    @property
    def _chunk_deprecated(self):
        return 3

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
        self._depth = np.array(self.depth)
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
        self._depth = np.array(self.depth)
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
        self._depth = depth
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

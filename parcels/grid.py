from parcels.loggers import logger
import numpy as np
from ctypes import Structure, c_int, c_float, c_double, POINTER, cast, c_void_p, pointer
from enum import IntEnum

__all__ = ['GridCode', 'RectilinearZGrid', 'RectilinearSGrid', 'CurvilinearZGrid', 'CurvilinearSGrid', 'GridIndex', 'CGrid']


class GridCode(IntEnum):
    RectilinearZGrid = 0
    RectilinearSGrid = 1
    CurvilinearZGrid = 2
    CurvilinearSGrid = 3


class CGrid(Structure):
    _fields_ = [('gtype', c_int),
                ('grid', c_void_p)]


class Grid(object):
    """Grid class that defines a (spatial and temporal) grid on which Fields are defined

    """

    @property
    def ctypes_struct(self):
        self.cgrid = cast(pointer(self.child_ctypes_struct), c_void_p)
        cstruct = CGrid(self.gtype, self.cgrid.value)
        return cstruct

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


class RectilinearGrid(Grid):
    """Rectilinear Grid
       Private base class for RectilinearZGrid and RectilinearSGrid

    """

    def __init__(self, lon, lat, time, time_origin, mesh):
        assert(isinstance(lon, np.ndarray) and len(lon.shape) == 1), 'lon is not a numpy vector'
        assert(isinstance(lat, np.ndarray) and len(lat.shape) == 1), 'lat is not a numpy vector'
        assert (isinstance(time, np.ndarray) or not time), 'time is not a numpy array'
        if isinstance(time, np.ndarray):
            assert(len(time.shape) == 1), 'time is not a vector'

        self.lon = lon
        self.lat = lat
        self.time = np.zeros(1, dtype=np.float64) if time is None else time
        if not self.lon.dtype == np.float32:
            logger.warning_once("Casting lon data to np.float32")
            self.lon = self.lon.astype(np.float32)
        if not self.lat.dtype == np.float32:
            logger.warning_once("Casting lat data to np.float32")
            self.lat = self.lat.astype(np.float32)
        if not self.time.dtype == np.float64:
            logger.warning_once("Casting time data to np.float64")
            self.time = self.time.astype(np.float64)
        self.xdim = self.lon.size
        self.ydim = self.lat.size
        self.tdim = self.time.size
        self.time_origin = time_origin
        self.mesh = mesh
        self.cstruct = None
        self.cell_edge_sizes = {}

    def add_periodic_halo(self, zonal, meridional, halosize=5):
        """Add a 'halo' to the Grid, through extending the Grid (and lon/lat)
        similarly to the halo created for the Fields

        :param zonal: Create a halo in zonal direction (boolean)
        :param meridional: Create a halo in meridional direction (boolean)
        :param halosize: size of the halo (in grid points). Default is 5 grid points
        """
        if zonal:
            lonshift = (self.lon[-1] - 2 * self.lon[0] + self.lon[1])
            self.lon = np.concatenate((self.lon[-halosize:] - lonshift,
                                      self.lon, self.lon[0:halosize] + lonshift))
            self.xdim = self.lon.size
        if meridional:
            latshift = (self.lat[-1] - 2 * self.lat[0] + self.lat[1])
            self.lat = np.concatenate((self.lat[-halosize:] - latshift,
                                      self.lat, self.lat[0:halosize] + latshift))
            self.ydim = self.lat.size

    def advancetime(self, grid_new):
        if len(grid_new.time) is not 1:
            raise RuntimeError('New FieldSet needs to have only one snapshot')
        if grid_new.time > self.time[-1]:  # forward in time, so appending at end
            self.time = np.concatenate((self.time[1:], grid_new.time))
            return 1
        elif grid_new.time < self.time[0]:  # backward in time, so prepending at start
            self.time = np.concatenate((grid_new.time, self.time[:-1]))
            return -1
        else:
            raise RuntimeError("Time of field_new in Field.advancetime() overlaps with times in old Field")

    @property
    def child_ctypes_struct(self):
        """Returns a ctypes struct object containing all relevant
        pointers and sizes for this grid."""

        class CStructuredGrid(Structure):
            # z4d is only to have same cstruct as RectilinearSGrid
            _fields_ = [('xdim', c_int), ('ydim', c_int), ('zdim', c_int),
                        ('tdim', c_int), ('z4d', c_int), ('min_lon_source', c_int),
                        ('lon', POINTER(c_float)), ('lat', POINTER(c_float)),
                        ('depth', POINTER(c_float)), ('time', POINTER(c_double))
                        ]

        # Create and populate the c-struct object
        if not self.cstruct:  # Not to point to the same grid various times if grid in various fields
            self.cstruct = CStructuredGrid(self.xdim, self.ydim, self.zdim,
                                           self.tdim, self.z4d, self.mesh == 'spherical',
                                           self.lon.ctypes.data_as(POINTER(c_float)),
                                           self.lat.ctypes.data_as(POINTER(c_float)),
                                           self.depth.ctypes.data_as(POINTER(c_float)),
                                           self.time.ctypes.data_as(POINTER(c_double)))
        return self.cstruct


class RectilinearZGrid(RectilinearGrid):
    """Rectilinear Z Grid

    :param lon: Vector containing the longitude coordinates of the grid
    :param lat: Vector containing the latitude coordinates of the grid
    :param depth: Vector containing the vertical coordinates of the grid, which are z-coordinates.
           The depth of the different layers is thus constant.
    :param time: Vector containing the time coordinates of the grid
    :param time_origin: Time origin of the time axis
    :param mesh: String indicating the type of mesh coordinates and
           units used during velocity interpolation:

           1. spherical (default): Lat and lon in degree, with a
              correction for zonal velocity U near the poles.
           2. flat: No conversion, lat/lon are assumed to be in m.
    """

    def __init__(self, lon, lat, depth=None, time=None, time_origin=0, mesh='flat'):
        RectilinearGrid.__init__(self, lon, lat, time, time_origin, mesh)
        if isinstance(depth, np.ndarray):
            assert(len(depth.shape) == 1), 'depth is not a vector'

        self.gtype = GridCode.RectilinearZGrid
        self.depth = np.zeros(1, dtype=np.float32) if depth is None else depth
        self.zdim = self.depth.size
        self.z4d = -1  # only used in RectilinearSGrid
        if not self.depth.dtype == np.float32:
            logger.warning_once("Casting depth data to np.float32")
            self.depth = self.depth.astype(np.float32)


class RectilinearSGrid(RectilinearGrid):
    """Rectilinear S Grid. Same horizontal discretisation as a rectilinear z grid,
       but with s vertical coordinates

    :param lon: Vector containing the longitude coordinates of the grid
    :param lat: Vector containing the latitude coordinates of the grid
    :param depth: 4D (time-evolving) or 3D (time-independent) array containing the vertical coordinates of the grid,
           which are s-coordinates.
           s-coordinates can be terrain-following (sigma) or iso-density (rho) layers,
           or any generalised vertical discretisation.
           The depth of each node depends then on the horizontal position (lon, lat),
           the number of the layer and the time is depth is a 4D array.
           depth array is either a 4D array[xdim][ydim][zdim][tdim] or a 3D array[xdim][ydim[zdim].
    :param time: Vector containing the time coordinates of the grid
    :param time_origin: Time origin of the time axis
    :param mesh: String indicating the type of mesh coordinates and
           units used during velocity interpolation:

           1. spherical (default): Lat and lon in degree, with a
              correction for zonal velocity U near the poles.
           2. flat: No conversion, lat/lon are assumed to be in m.
    """

    def __init__(self, lon, lat, depth, time=None, time_origin=0, mesh='flat'):
        RectilinearGrid.__init__(self, lon, lat, time, time_origin, mesh)
        assert(isinstance(depth, np.ndarray) and len(depth.shape) in [3, 4]), 'depth is not a 4D numpy array'

        self.gtype = GridCode.RectilinearSGrid
        self.depth = depth
        self.zdim = self.depth.shape[2]
        self.z4d = len(self.depth.shape) == 4
        if not self.depth.dtype == np.float32:
            logger.warning_once("Casting depth data to np.float32")
            self.depth = self.depth.astype(np.float32)


class CurvilinearGrid(Grid):

    def __init__(self, lon, lat, time=None, time_origin=0, mesh='flat'):
        assert(isinstance(lon, np.ndarray) and len(lon.squeeze().shape) == 2), 'lon is not a 2D numpy array'
        assert(isinstance(lat, np.ndarray) and len(lat.squeeze().shape) == 2), 'lat is not a 2D numpy array'
        assert (isinstance(time, np.ndarray) or not time), 'time is not a numpy array'
        if isinstance(time, np.ndarray):
            assert(len(time.shape) == 1), 'time is not a vector'

        self.lon = lon.squeeze()
        self.lat = lat.squeeze()
        self.time = np.zeros(1, dtype=np.float64) if time is None else time
        if not self.lon.dtype == np.float32:
            logger.warning_once("Casting lon data to np.float32")
            self.lon = self.lon.astype(np.float32)
        if not self.lat.dtype == np.float32:
            logger.warning_once("Casting lat data to np.float32")
            self.lat = self.lat.astype(np.float32)
        if not self.time.dtype == np.float64:
            logger.warning_once("Casting time data to np.float64")
            self.time = self.time.astype(np.float64)
        self.time_origin = time_origin
        self.mesh = mesh
        self.cstruct = None
        self.xdim = self.lon.shape[1]
        self.ydim = self.lon.shape[0]
        self.tdim = self.time.size
        self.cell_edge_sizes = {}

    def add_periodic_halo(self, zonal, meridional, halosize=5):
        """Add a 'halo' to the Grid, through extending the Grid (and lon/lat)
        similarly to the halo created for the Fields

        :param zonal: Create a halo in zonal direction (boolean)
        :param meridional: Create a halo in meridional direction (boolean)
        :param halosize: size of the halo (in grid points). Default is 5 grid points
        """
        if zonal:
            lonshift = self.lon[:, -1] - 2 * self.lon[:, 0] + self.lon[:, 1]
            self.lon = np.concatenate((self.lon[:, -halosize:] - lonshift[:, np.newaxis],
                                       self.lon, self.lon[:, 0:halosize] + lonshift[:, np.newaxis]),
                                      axis=len(self.lon.shape)-1)
            self.lat = np.concatenate((self.lat[:, -halosize:],
                                       self.lat, self.lat[:, 0:halosize]),
                                      axis=len(self.lat.shape)-1)
            self.xdim = self.lon.shape[1]
            self.ydim = self.lat.shape[0]
        if meridional:
            latshift = self.lat[-1, :] - 2 * self.lat[0, :] + self.lat[1, :]
            self.lat = np.concatenate((self.lat[-halosize:, :] - latshift[np.newaxis, :],
                                       self.lat, self.lat[0:halosize, :] + latshift[np.newaxis, :]),
                                      axis=len(self.lat.shape)-2)
            self.lon = np.concatenate((self.lon[-halosize:, :],
                                       self.lon, self.lon[0:halosize, :]),
                                      axis=len(self.lon.shape)-2)
            self.xdim = self.lon.shape[1]
            self.ydim = self.lat.shape[0]

    def advancetime(self, grid_new):
        if len(grid_new.time) is not 1:
            raise RuntimeError('New FieldSet needs to have only one snapshot')
        if grid_new.time > self.time[-1]:  # forward in time, so appending at end
            self.time = np.concatenate((self.time[1:], grid_new.time))
            return 1
        elif grid_new.time < self.time[0]:  # backward in time, so prepending at start
            self.time = np.concatenate((grid_new.time, self.time[:-1]))
            return -1
        else:
            raise RuntimeError("Time of field_new in Field.advancetime() overlaps with times in old Field")

    @property
    def child_ctypes_struct(self):
        """Returns a ctypes struct object containing all relevant
        pointers and sizes for this grid."""

        class CStructuredGrid(Structure):
            _fields_ = [('xdim', c_int), ('ydim', c_int), ('zdim', c_int),
                        ('tdim', c_int), ('z4d', c_int), ('min_lon_source', c_int),
                        ('lon', POINTER(c_float)), ('lat', POINTER(c_float)),
                        ('depth', POINTER(c_float)), ('time', POINTER(c_double))
                        ]

        # Create and populate the c-struct object
        if not self.cstruct:  # Not to point to the same grid various times if grid in various fields
            self.cstruct = CStructuredGrid(self.xdim, self.ydim, self.zdim,
                                           self.tdim, self.z4d, self.mesh == 'spherical',
                                           self.lon.ctypes.data_as(POINTER(c_float)),
                                           self.lat.ctypes.data_as(POINTER(c_float)),
                                           self.depth.ctypes.data_as(POINTER(c_float)),
                                           self.time.ctypes.data_as(POINTER(c_double)))
        return self.cstruct


class CurvilinearZGrid(CurvilinearGrid):
    """Curvilinear Z Grid.

    :param lon: 2D array containing the longitude coordinates of the grid
    :param lat: 2D array containing the latitude coordinates of the grid
    :param depth: Vector containing the vertical coordinates of the grid, which are z-coordinates.
           The depth of the different layers is thus constant.
    :param time: Vector containing the time coordinates of the grid
    :param time_origin: Time origin of the time axis
    :param mesh: String indicating the type of mesh coordinates and
           units used during velocity interpolation:

           1. spherical (default): Lat and lon in degree, with a
              correction for zonal velocity U near the poles.
           2. flat: No conversion, lat/lon are assumed to be in m.
    """

    def __init__(self, lon, lat, depth=None, time=None, time_origin=0, mesh='flat'):
        CurvilinearGrid.__init__(self, lon, lat, time, time_origin, mesh)
        if isinstance(depth, np.ndarray):
            assert(len(depth.shape) == 1), 'depth is not a vector'

        self.gtype = GridCode.CurvilinearZGrid
        self.depth = np.zeros(1, dtype=np.float32) if depth is None else depth
        self.zdim = self.depth.size
        self.z4d = -1  # only for SGrid
        if not self.depth.dtype == np.float32:
            logger.warning_once("Casting depth data to np.float32")
            self.depth = self.depth.astype(np.float32)


class CurvilinearSGrid(CurvilinearGrid):
    """Curvilinear S Grid.

    :param lon: 2D array containing the longitude coordinates of the grid
    :param lat: 2D array containing the latitude coordinates of the grid
    :param depth: 4D (time-evolving) or 3D (time-independent) array containing the vertical coordinates of the grid,
           which are s-coordinates.
           s-coordinates can be terrain-following (sigma) or iso-density (rho) layers,
           or any generalised vertical discretisation.
           The depth of each node depends then on the horizontal position (lon, lat),
           the number of the layer and the time is depth is a 4D array.
           depth array is either a 4D array[xdim][ydim][zdim][tdim] or a 3D array[xdim][ydim[zdim].
    :param time: Vector containing the time coordinates of the grid
    :param time_origin: Time origin of the time axis
    :param mesh: String indicating the type of mesh coordinates and
           units used during velocity interpolation:

           1. spherical (default): Lat and lon in degree, with a
              correction for zonal velocity U near the poles.
           2. flat: No conversion, lat/lon are assumed to be in m.
    """

    def __init__(self, lon, lat, depth, time=None, time_origin=0, mesh='flat'):
        CurvilinearGrid.__init__(self, lon, lat, time, time_origin, mesh)
        assert(isinstance(depth, np.ndarray) and len(depth.shape) in [3, 4]), 'depth is not a 4D numpy array'

        self.gtype = GridCode.CurvilinearSGrid
        self.depth = depth
        self.zdim = self.depth.shape[2]
        self.z4d = len(self.depth.shape) == 4
        if not self.depth.dtype == np.float32:
            logger.warning_once("Casting depth data to np.float32")
            self.depth = self.depth.astype(np.float32)


class GVariable(object):
    def __init__(self, name):
        self.name = name

    def __get__(self, instance, cls):
        return instance._cptr.__getitem__(self.name)

    def __set__(self, instance, value):
        instance._cptr.__setitem__(self.name, value)


class GridIndex(object):
    """GridIndex class that defines the indices of the particle in the grid

    :param grid: grid related to this grid index

    """
    xi = GVariable('xi')
    yi = GVariable('yi')
    zi = GVariable('zi')
    ti = GVariable('ti')

    def __init__(self, grid, *args, **kwargs):
        self._cptr = kwargs.pop('cptr', None)
        self.xi = 0
        self.yi = 0
        self.zi = 0
        self.ti = -1   # -1 means that ti has not been computed yet

    @classmethod
    def dtype(cls):
        type_list = [('xi', np.int32), ('yi', np.int32), ('zi', np.int32), ('ti', np.int32)]
        return np.dtype(type_list)

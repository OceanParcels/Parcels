from parcels.loggers import logger
import numpy as np
from ctypes import Structure, c_int, c_float, c_double, POINTER, cast, c_void_p, pointer
from enum import IntEnum

__all__ = ['GridCode', 'RectilinearZGrid', 'RectilinearSGrid', 'GridIndex', 'CGrid']


class GridCode(IntEnum):
    RectilinearZGrid = 0
    RectilinearSGrid = 1
    CurvilinearGrid = 2


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


class RectilinearZGrid(Grid):
    """Rectilinear Z Grid

    :param name: Name of the grid
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

    def __init__(self, name, lon, lat, depth=None, time=None, time_origin=0, mesh='flat'):
        assert isinstance(lon, np.ndarray), 'lon is not a numpy array'
        sh = lon.shape
        assert(len(sh) == 1 or len(sh) == 2 and min(sh) == 2), 'lon is not a vector'
        assert isinstance(lat, np.ndarray), 'lat is not a numpy array'
        sh = lat.shape
        assert(len(sh) == 1 or len(sh) == 2 and min(sh) == 2), 'lat is not a vector'
        assert (isinstance(depth, np.ndarray) or not depth), 'depth is not a numpy array'
        if isinstance(depth, np.ndarray):
            sh = depth.shape
            assert(len(sh) == 1 or len(sh) == 2 and min(sh) == 2), 'depth is not a vector'
        assert (isinstance(time, np.ndarray) or not time), 'time is not a numpy array'
        if isinstance(time, np.ndarray):
            sh = time.shape
            assert(len(sh) == 1 or len(sh) == 2 and min(sh) == 2), 'time is not a vector'

        self.name = name
        self.gtype = GridCode.RectilinearZGrid
        self.lon = lon
        self.lat = lat
        self.depth = np.zeros(1, dtype=np.float32) if depth is None else depth
        self.time = np.zeros(1, dtype=np.float64) if time is None else time
        if not self.lon.dtype == np.float32:
            logger.warning_once("Casting lon data to np.float32")
            self.lon = self.lon.astype(np.float32)
        if not self.lat.dtype == np.float32:
            logger.warning_once("Casting lat data to np.float32")
            self.lat = self.lat.astype(np.float32)
        if not self.depth.dtype == np.float32:
            logger.warning_once("Casting depth data to np.float32")
            self.depth = self.depth.astype(np.float32)
        if not self.time.dtype == np.float64:
            logger.warning_once("Casting time data to np.float64")
            self.time = self.time.astype(np.float64)
        self.time_origin = time_origin
        self.mesh = mesh
        self.cstruct = None

    @property
    def child_ctypes_struct(self):
        """Returns a ctypes struct object containing all relevant
        pointers and sizes for this grid."""

        class CRectilinearGrid(Structure):
            # z4d is only to have same cstruct as RectilinearSGrid
            _fields_ = [('xdim', c_int), ('ydim', c_int), ('zdim', c_int),
                        ('tdim', c_int), ('z4d', c_int),
                        ('lon', POINTER(c_float)), ('lat', POINTER(c_float)),
                        ('depth', POINTER(c_float)), ('time', POINTER(c_double))
                        ]

        # Create and populate the c-struct object
        if not self.cstruct:  # Not to point to the same grid various times if grid in various fields
            self.cstruct = CRectilinearGrid(self.lon.size, self.lat.size, self.depth.size,
                                            self.time.size, -1,
                                            self.lon.ctypes.data_as(POINTER(c_float)),
                                            self.lat.ctypes.data_as(POINTER(c_float)),
                                            self.depth.ctypes.data_as(POINTER(c_float)),
                                            self.time.ctypes.data_as(POINTER(c_double)))
        return self.cstruct


class RectilinearSGrid(Grid):
    """Rectilinear S Grid. Same horizontal discretisation as a rectilinear z grid,
       but with s vertical coordinates

    :param name: Name of the grid
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

    def __init__(self, name, lon, lat, depth=None, time=None, time_origin=0, mesh='flat'):
        assert(isinstance(lon, np.ndarray) and len(lon.shape) == 1), 'lon is not a numpy vector'
        assert(isinstance(lat, np.ndarray) and len(lat.shape) == 1), 'lat is not a numpy vector'
        assert(isinstance(depth, np.ndarray) and len(depth.shape) in [3, 4]), 'depth is not a 4D numpy array'
        assert (isinstance(time, np.ndarray) or not time), 'time is not a numpy array'
        if isinstance(time, np.ndarray):
            assert(len(time.shape) == 1), 'time is not a vector'

        self.name = name
        self.gtype = GridCode.RectilinearSGrid
        self.lon = lon
        self.lat = lat
        self.depth = depth
        self.z4d = len(depth.shape) == 4
        self.time = np.zeros(1, dtype=np.float64) if time is None else time
        if not self.lon.dtype == np.float32:
            logger.warning_once("Casting lon data to np.float32")
            self.lon = self.lon.astype(np.float32)
        if not self.lat.dtype == np.float32:
            logger.warning_once("Casting lat data to np.float32")
            self.lat = self.lat.astype(np.float32)
        if not self.depth.dtype == np.float32:
            logger.warning_once("Casting depth data to np.float32")
            self.depth = self.depth.astype(np.float32)
        if not self.time.dtype == np.float64:
            logger.warning_once("Casting time data to np.float64")
            self.time = self.time.astype(np.float64)
        self.time_origin = time_origin
        self.mesh = mesh
        self.cstruct = None

    @property
    def child_ctypes_struct(self):
        """Returns a ctypes struct object containing all relevant
        pointers and sizes for this grid."""

        class CRectilinearGrid(Structure):
            _fields_ = [('xdim', c_int), ('ydim', c_int), ('zdim', c_int),
                        ('tdim', c_int), ('z4d', c_int),
                        ('lon', POINTER(c_float)), ('lat', POINTER(c_float)),
                        ('depth', POINTER(c_float)), ('time', POINTER(c_double))
                        ]

        # Create and populate the c-struct object
        if not self.cstruct:  # Not to point to the same grid various times if grid in various fields
            self.cstruct = CRectilinearGrid(self.lon.size, self.lat.size, self.depth.shape[2],
                                            self.time.size, self.z4d,
                                            self.lon.ctypes.data_as(POINTER(c_float)),
                                            self.lat.ctypes.data_as(POINTER(c_float)),
                                            self.depth.ctypes.data_as(POINTER(c_float)),
                                            self.time.ctypes.data_as(POINTER(c_double)))
        return self.cstruct


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
        self.name = grid.name
        self.xi = 0
        self.yi = 0
        self.zi = 0
        self.ti = 0

    @classmethod
    def dtype(cls):
        type_list = [('xi', np.int32), ('yi', np.int32), ('zi', np.int32), ('ti', np.int32)]
        return np.dtype(type_list)

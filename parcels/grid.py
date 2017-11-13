import numpy as np
from ctypes import Structure, c_int, c_float, c_double, POINTER, cast, c_char_p, c_void_p

__all__ = ['StructuredGrid', 'GridIndex', 'CGrid']

#gtype = {'structured': StructuredGrid,
#         'curvilinear': CurvilinearGrid,
#         'unstructured': UnstructuredGrid}


class CGrid(Structure):
    _fields_ = [('name', c_char_p),
                ('xdim', c_int), ('ydim', c_int), ('zdim', c_int),
                ('tdim', c_int), ('tidx', c_int),
                ('lon', POINTER(c_float)), ('lat', POINTER(c_float)),
                ('depth', POINTER(c_float)), ('time', POINTER(c_double))
                ]


class CGridIndex(Structure):
    _fields_ = [('xi', c_int), ('yi', c_int), ('zi', c_int)]


class Grid(object):
    """Grid class that defines a (spatial and temporal) grid on which Fields are defined

    """

    def __init__(self, gtype):
        self.gtype = gtype


class StructuredGrid(Grid):
    """Structured Grid

    :param name:
    :param lon:
    :param lat:
    :param depth:
    :param t
    """

    def __init__(self, name, lon, lat, depth=None, time=None):
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
        self.lon = lon
        self.lat = lat
        self.depth = depth if depth else np.zeros(1, dtype=np.float32)
        self.time = time

    @property
    def ctypes_struct(self):
        """Returns a ctypes struct object containing all relevant
        pointers and sizes for this grid."""

        # Create and populate the c-struct object
        cstruct = CGrid(self.name,
                        self.lon.size, self.lat.size, self.depth.size,
                        self.time.size, 0,
                        self.lon.ctypes.data_as(POINTER(c_float)),
                        self.lat.ctypes.data_as(POINTER(c_float)),
                        self.depth.ctypes.data_as(POINTER(c_float)),
                        self.time.ctypes.data_as(POINTER(c_double)))
        return cstruct


class GVariable(object):
    def __init__(self, name):
        self.name = name

    def __get__(self, instance, cls):
        return instance._cptr.__getitem__(self.name)

    def __set__(self, instance, value):
        if self.name == 'name':
            ptr = cast(value, (c_void_p))
            instance._cptr.__setitem__(self.name, ptr.value)
        else:
            instance._cptr.__setitem__(self.name, value)


class GridIndex(object):
    """GridIndex class that defines the indices of the particle in the grid

    :param grid:

    """
    name = GVariable('name')
    xi = GVariable('xi')
    yi = GVariable('yi')
    zi = GVariable('zi')

    def __init__(self, grid, *args, **kwargs):
        self._cptr = kwargs.pop('cptr', None)
        #self.grid = grid
        self.name = grid.name
        self.xi = 0
        self.yi = 0
        self.zi = 0

    @classmethod
    def dtype(cls):
        type_list = [('name', c_void_p), ('xi', np.int32), ('yi', np.int32), ('zi', np.int32),
                     ('pad', np.int32)]
        return np.dtype(type_list)

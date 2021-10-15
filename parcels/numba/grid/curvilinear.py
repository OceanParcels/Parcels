import numpy as np

from parcels.tools.converters import TimeConverter
from numba.experimental import jitclass
import numba as nb
from copy import deepcopy
from numba.core.typing.asnumbatype import as_numba_type
from parcels.numba.grid.base import BaseGrid, _base_spec, GridCode


_curve_spec = deepcopy(_base_spec)
_curve_spec.extend([
    ("lat", nb.float32[:, :]),
    ("lon", nb.float32[:, :]),
])


class CurvilinearGrid(BaseGrid):
    __init_base = BaseGrid.__init__

    def __init__(self, lon, lat, time=None, time_origin=None, mesh='flat'):
#         assert(isinstance(lon, np.ndarray) and len(lon.squeeze().shape) == 2), 'lon is not a 2D numpy array'
#         assert(isinstance(lat, np.ndarray) and len(lat.squeeze().shape) == 2), 'lat is not a 2D numpy array'
#         assert (isinstance(time, np.ndarray) or not time), 'time is not a numpy array'
#         if isinstance(time, np.ndarray):
#             assert(len(time.shape) == 1), 'time is not a vector'

        lon = lon.squeeze()
        lat = lat.squeeze()
        self.__init_base(lon, lat, time, time_origin, mesh)
        self.xdim = self.lon.shape[1]
        self.ydim = self.lon.shape[0]
        self.tdim = self.time.size

    def get_dlon(self):
        return self.lon[0, 1:] - self.lon[0, :-1]

    def search_indices_curvilinear(self, x, y, z, ti=-1, time=-1, particle=None, search2D=False):
        if particle:
            xi = particle.xi[self.igrid]
            yi = particle.yi[self.igrid]
        else:
            xi = int(self.grid.xdim / 2) - 1
            yi = int(self.grid.ydim / 2) - 1
        xsi = eta = -1
        grid = self.grid
        invA = np.array([[1, 0, 0, 0],
                         [-1, 1, 0, 0],
                         [-1, 0, 0, 1],
                         [1, -1, 1, -1]])
        maxIterSearch = 1e6
        it = 0
        tol = 1.e-10
        if not grid.zonal_periodic:
            if x < grid.lonlat_minmax[0] or x > grid.lonlat_minmax[1]:
                if grid.lon[0, 0] < grid.lon[0, -1]:
                    raise FieldOutOfBoundError(x, y, z, field=self)
                elif x < grid.lon[0, 0] and x > grid.lon[0, -1]:  # This prevents from crashing in [160, -160]
                    raise FieldOutOfBoundError(x, y, z, field=self)
        if y < grid.lonlat_minmax[2] or y > grid.lonlat_minmax[3]:
            raise FieldOutOfBoundError(x, y, z, field=self)

        while xsi < -tol or xsi > 1+tol or eta < -tol or eta > 1+tol:
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
                raise FieldOutOfBoundError(x, y, 0, field=self)
            if xsi > 1 and eta > 1 and xi == grid.xdim-1 and yi == grid.ydim-1:
                raise FieldOutOfBoundError(x, y, 0, field=self)
            if xsi < -tol:
                xi -= 1
            elif xsi > 1+tol:
                xi += 1
            if eta < -tol:
                yi -= 1
            elif eta > 1+tol:
                yi += 1
            (xi, yi) = self.reconnect_bnd_indices(xi, yi, grid.xdim, grid.ydim, grid.mesh)
            it += 1
            if it > maxIterSearch:
                print('Correct cell not found after %d iterations' % maxIterSearch)
                raise FieldOutOfBoundError(x, y, 0, field=self)
        xsi = max(0., xsi)
        eta = max(0., eta)
        xsi = min(1., xsi)
        eta = min(1., eta)

        if grid.zdim > 1 and not search2D:
            if grid.gtype == GridCode.CurvilinearZGrid:
                try:
                    (zi, zeta) = self.search_indices_vertical_z(z)
                except FieldOutOfBoundError:
                    raise FieldOutOfBoundError(x, y, z, field=self)
            elif grid.gtype == GridCode.CurvilinearSGrid:
                (zi, zeta) = self.search_indices_vertical_s(x, y, z, xi, yi, xsi, eta, ti, time)
        else:
            zi = -1
            zeta = 0

        if not ((0 <= xsi <= 1) and (0 <= eta <= 1) and (0 <= zeta <= 1)):
            raise FieldSamplingError(x, y, z, field=self)

        if particle:
            particle.xi[self.igrid] = xi
            particle.yi[self.igrid] = yi
            particle.zi[self.igrid] = zi

        return (xsi, eta, zeta, xi, yi, zi)

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

    def add_periodic_halo(self, zonal, meridional, halosize=5):
        """Add a 'halo' to the Grid, through extending the Grid (and lon/lat)
        similarly to the halo created for the Fields

        :param zonal: Create a halo in zonal direction (boolean)
        :param meridional: Create a halo in meridional direction (boolean)
        :param halosize: size of the halo (in grid points). Default is 5 grid points
        """
        if zonal:
            lonshift = self.lon[:, -1] - 2 * self.lon[:, 0] + self.lon[:, 1]
#             if not np.allclose(self.lon[:, 1]-self.lon[:, 0], self.lon[:, -1]-self.lon[:, -2]):
#                 logger.warning_once("The zonal halo is located at the east and west of current grid, with a dx = lon[:,1]-lon[:,0] between the last nodes of the original grid and the first ones of the halo. In your grid, lon[:,1]-lon[:,0] != lon[:,-1]-lon[:,-2]. Is the halo computed as you expect?")
            self.lon = np.concatenate((self.lon[:, -halosize:] - lonshift[:, np.newaxis],
                                       self.lon, self.lon[:, 0:halosize] + lonshift[:, np.newaxis]),
                                      axis=len(self.lon.shape)-1)
            self.lat = np.concatenate((self.lat[:, -halosize:],
                                       self.lat, self.lat[:, 0:halosize]),
                                      axis=len(self.lat.shape)-1)
            self.xdim = self.lon.shape[1]
            self.ydim = self.lat.shape[0]
            self.zonal_periodic = True
            self.zonal_halo = halosize
        if meridional:
#             if not np.allclose(self.lat[1, :]-self.lat[0, :], self.lat[-1, :]-self.lat[-2, :]):
#                 logger.warning_once("The meridional halo is located at the north and south of current grid, with a dy = lat[1,:]-lat[0,:] between the last nodes of the original grid and the first ones of the halo. In your grid, lat[1,:]-lat[0,:] != lat[-1,:]-lat[-2,:]. Is the halo computed as you expect?")
            latshift = self.lat[-1, :] - 2 * self.lat[0, :] + self.lat[1, :]
            self.lat = np.concatenate((self.lat[-halosize:, :] - latshift[np.newaxis, :],
                                       self.lat, self.lat[0:halosize, :] + latshift[np.newaxis, :]),
                                      axis=len(self.lat.shape)-2)
            self.lon = np.concatenate((self.lon[-halosize:, :],
                                       self.lon, self.lon[0:halosize, :]),
                                      axis=len(self.lon.shape)-2)
            self.xdim = self.lon.shape[1]
            self.ydim = self.lat.shape[0]
            self.meridional_halo = halosize
#         if isinstance(self, CurvilinearSGrid):
#             self.add_Sdepth_periodic_halo(zonal, meridional, halosize)


@jitclass(spec=_base_spec+[("depth", nb.float32[:])])
class CurvilinearZGrid(CurvilinearGrid):
    """Curvilinear Z Grid.

    :param lon: 2D array containing the longitude coordinates of the grid
    :param lat: 2D array containing the latitude coordinates of the grid
    :param depth: Vector containing the vertical coordinates of the grid, which are z-coordinates.
           The depth of the different layers is thus constant.
    :param time: Vector containing the time coordinates of the grid
    :param time_origin: Time origin (TimeConverter object) of the time axis
    :param mesh: String indicating the type of mesh coordinates and
           units used during velocity interpolation:

           1. spherical (default): Lat and lon in degree, with a
              correction for zonal velocity U near the poles.
           2. flat: No conversion, lat/lon are assumed to be in m.
    """
    __init__curv = CurvilinearGrid.__init__

    def __init__(self, lon, lat, depth=None, time=None, time_origin=None, mesh='flat'):
        self.__init__curv(lon, lat, time, time_origin, mesh)
#         if isinstance(depth, np.ndarray):
#             assert(len(depth.shape) == 1), 'depth is not a vector'

        self.gtype = GridCode.CurvilinearZGrid
        self.depth = np.zeros(1, dtype=np.float32) if depth is None else depth
        self.zdim = self.depth.size
#         self.z4d = -1  # only for SGrid
#         if not self.depth.dtype == np.float32:
#             self.depth = self.depth.astype(np.float32)


@jitclass(spec=_base_spec+[("depth", nb.float32[:, :, :, :])])
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
    :param time_origin: Time origin (TimeConverter object) of the time axis
    :param mesh: String indicating the type of mesh coordinates and
           units used during velocity interpolation:

           1. spherical (default): Lat and lon in degree, with a
              correction for zonal velocity U near the poles.
           2. flat: No conversion, lat/lon are assumed to be in m.
    """
    __init__curv = CurvilinearGrid.__init__

    def __init__(self, lon, lat, depth, time=None, time_origin=None, mesh='flat'):
        self.__init__curv(lon, lat, time, time_origin, mesh)
#         assert(isinstance(depth, np.ndarray) and len(depth.shape) in [3, 4]), 'depth is not a 4D numpy array'

        self.gtype = GridCode.CurvilinearSGrid
        self.depth = depth
        self.zdim = self.depth.shape[-3]
#         self.z4d = len(self.depth.shape) == 4
#         if self.z4d:
            # self.depth.shape[0] is 0 for S grids loaded from netcdf file
        assert self.tdim == self.depth.shape[0] or self.depth.shape[0] == 0, 'depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]'
        assert self.xdim == self.depth.shape[-1] or self.depth.shape[-1] == 0, 'depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]'
        assert self.ydim == self.depth.shape[-2] or self.depth.shape[-2] == 0, 'depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]'
#         else:
#             assert self.xdim == self.depth.shape[-1], 'depth dimension has the wrong format. It should be [zdim, ydim, xdim]'
#             assert self.ydim == self.depth.shape[-2], 'depth dimension has the wrong format. It should be [zdim, ydim, xdim]'
#         if not self.depth.dtype == np.float32:
#             self.depth = self.depth.astype(np.float32)

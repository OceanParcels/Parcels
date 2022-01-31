import numpy as np

from numba.experimental import jitclass
import numba as nb
from parcels.numba.grid.base import BaseGrid, GridCode, _base_grid_spec
from parcels.numba.grid.zgrid import BaseZGrid
from parcels.numba.grid.sgrid import BaseSGrid
from parcels.numba.utils import numba_reshape_34


def _rect_grid_spec():
    return _base_grid_spec() + [
        ("lat", nb.float32[:]),
        ("lon", nb.float32[:]),
    ]


class RectilinearGrid(BaseGrid):
    """Rectilinear Grid
       Private base class for RectilinearZGrid and RectilinearSGrid
    """
    __init_base = BaseGrid.__init__

    def __init__(self, lon, lat, time, mesh):
        self.__init_base(lon, lat, time, mesh)
        self.xdim = self.lon.size
        self.ydim = self.lat.size
        self.tdim = self.time.size

    def search_indices(self, x, y, z, ti=-1, time=-1, search2D=False,
                       particle=None, interp_method="linear"):
        """Copied from original code"""
        if self.xdim > 1 and (not self.zonal_periodic):
            if x < self.lonlat_minmax[0] or x > self.lonlat_minmax[1]:
                self.FieldOutOfBoundError(x, y, z)
        if self.ydim > 1 and (y < self.lonlat_minmax[2] or y > self.lonlat_minmax[3]):
                self.FieldOutOfBoundError(x, y, z)

        if self.xdim > 1:
            if self.mesh != 'spherical':
                lon_index = self.lon < x
                if lon_index.all():
                    xi = len(self.lon) - 2
                else:
                    xi = lon_index.argmin() - 1 if lon_index.any() else 0
                xsi = (x-self.lon[xi]) / (self.lon[xi+1]-self.lon[xi])
                if xsi < 0:
                    xi -= 1
                    xsi = (x-self.lon[xi]) / (self.lon[xi+1]-self.lon[xi])
                elif xsi > 1:
                    xi += 1
                    xsi = (x-self.lon[xi]) / (self.lon[xi+1]-self.lon[xi])
            else:
                lon_fixed = self.lon.copy()
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
        else:
            xi, xsi = -1, 0.0

        if self.ydim > 1:
            lat_index = self.lat < y
            if lat_index.all():
                yi = len(self.lat) - 2
            else:
                yi = lat_index.argmin() - 1 if lat_index.any() else 0

            eta = (y-self.lat[yi]) / (self.lat[yi+1]-self.lat[yi])
            if eta < 0:
                yi -= 1
                eta = (y-self.lat[yi]) / (self.lat[yi+1]-self.lat[yi])
            elif eta > 1:
                yi += 1
                eta = (y-self.lat[yi]) / (self.lat[yi+1]-self.lat[yi])
        else:
            yi, eta = -1, 0.0

        if self.zdim > 1 and not search2D:
            (zi, zeta) = self.search_indices_vertical(
                x, y, z, xi, yi, xsi,
                eta, ti, time, interp_method=interp_method)
        else:
            zi, zeta = -1, 0.0

        if not ((0 <= xsi <= 1) and (0 <= eta <= 1) and (0 <= zeta <= 1)):
            self.FieldSamplingError(x, y, z)

        if particle is not None:
            self.xi[particle.id] = xi
            self.yi[particle.id] = yi
            self.zi[particle.id] = zi

        return (xsi, eta, zeta, xi, yi, zi)

    def get_dlon(self):
        return self.lon[1:] - self.lon[:-1]

    def get_pxy(self, xi, yi):
        px = np.array([self.lon[xi], self.lon[xi+1], self.lon[xi+1],
                       self.lon[xi]]).astype(nb.float64)
        py = np.array([self.lat[yi], self.lat[yi], self.lat[yi+1],
                       self.lat[yi+1]]).astype(nb.float64)
        return px, py

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
            self.zonal_periodic = True
            self.zonal_halo = halosize
        if meridional:
            latshift = (self.lat[-1] - 2 * self.lat[0] + self.lat[1])
            self.lat = np.concatenate((self.lat[-halosize:] - latshift,
                                      self.lat, self.lat[0:halosize] + latshift))
            self.ydim = self.lat.size
            self.meridional_halo = halosize
        self.lonlat_minmax = np.array([np.nanmin(self.lon), np.nanmax(self.lon), np.nanmin(self.lat), np.nanmax(self.lat)], dtype=np.float32)
        if isinstance(self, RectilinearSGrid):
            self.add_Sdepth_periodic_halo(zonal, meridional, halosize)


@jitclass(spec=_rect_grid_spec()+[("depth", nb.float32[:])])
class RectilinearZGrid(RectilinearGrid, BaseZGrid):
    """Rectilinear Z Grid

    :param lon: Vector containing the longitude coordinates of the grid
    :param lat: Vector containing the latitude coordinates of the grid
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
    __init__rect = RectilinearGrid.__init__

    def __init__(self, lon, lat, depth=None, time=None, mesh='flat'):
        self.__init__rect(lon, lat, time, mesh)

        self.gtype = GridCode.RectilinearZGrid
        self.depth = np.zeros(1, dtype=np.float32) if depth is None else depth.astype(nb.float32)
        self.zdim = self.depth.size


@jitclass(spec=_rect_grid_spec()+[("depth", nb.float32[:, :, :, :])])
class RectilinearSGrid(RectilinearGrid, BaseSGrid):
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
    :param time_origin: Time origin (TimeConverter object) of the time axis
    :param mesh: String indicating the type of mesh coordinates and
           units used during velocity interpolation:

           1. spherical (default): Lat and lon in degree, with a
              correction for zonal velocity U near the poles.
           2. flat: No conversion, lat/lon are assumed to be in m.
    """
    __init_rect = RectilinearGrid.__init__

    # TODO: description of dimensions is wrong in docstring?

    def __init__(self, lon, lat, depth, time=None, mesh='flat'):
        self.__init_rect(lon, lat, time, mesh)
        assert len(depth.shape) in [3, 4], 'depth is not a 3D or 4D numpy array'
        depth = numba_reshape_34(depth)

        self.gtype = GridCode.RectilinearSGrid
        self.depth = depth.astype(nb.float32)
        self.zdim = self.depth.shape[-3]
        self.z4d = self.depth.shape[0] != 1
        assert self.tdim == self.depth.shape[0] or self.depth.shape[0] == 0, 'depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]'
        assert self.xdim == self.depth.shape[-1] or self.depth.shape[-1] == 0, 'depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]'
        assert self.ydim == self.depth.shape[-2] or self.depth.shape[-2] == 0, 'depth dimension has the wrong format. It should be [tdim, zdim, ydim, xdim]'
        if self.lat_flipped:
            self.depth = self.depth[:, :, ::-1, :]

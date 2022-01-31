import math

from numba.experimental import jitclass
from numba.core.typing.asnumbatype import as_numba_type
import numba as nb
import numpy as np

from .base_vector_field import NumbaBaseVectorField


class NumbaVectorField2D():
    """Python helper class to create 2D fields"""
    @staticmethod
    def _class(U):
        numba_class = U.numba_class
        vfield_class = jitclass(_NumbaVectorField2D, spec=[
            ("U", as_numba_type(numba_class)),
            ("V", as_numba_type(numba_class)),
            ("vector_type", nb.types.string),
            ("name", nb.types.string),
            ("gridindexingtype", nb.types.string),
        ])
        return vfield_class

    def create(self, name, U, V):
        "Create Numba 2D field from python fields"
        return self._class(name, U, V)


class _NumbaVectorField2D(NumbaBaseVectorField):
    """Numba compiled 2D field class"""
    def __init__(self, name, U, V):
        self.U = U
        self.V = V
        self.name = name
        self.gridindexingtype = U.gridindexingtype

    def dist(self, lon1, lon2, lat1, lat2, mesh, lat):
        if mesh == 'spherical':
            rad = np.pi/180.
            deg2m = 1852 * 60.
            return np.sqrt(((lon2-lon1)*deg2m*math.cos(rad * lat))**2 + ((lat2-lat1)*deg2m)**2)
        else:
            return np.sqrt((lon2-lon1)**2 + (lat2-lat1)**2)

    def spatial_slip_interpolation(self, ti, z, y, x, time, particle=None):
        (xsi, eta, zeta, xi, yi, zi) = self.U.grid.search_indices(x, y, z, ti, time)
        di = ti if self.U.grid.zdim == 1 else zi  # general third dimension

        f_u, f_v = 1, 1
        if self._is_land2D(di, yi, xi) and self._is_land2D(di, yi, xi+1) and self._is_land2D(di+1, yi, xi) \
                and self._is_land2D(di+1, yi, xi+1) and eta > 0:
            if self.U.interp_method == 'partialslip':
                f_u = f_u * (.5 + .5 * eta) / eta
            elif self.U.interp_method == 'freeslip':
                f_u = f_u / eta
        if self._is_land2D(di, yi+1, xi) and self._is_land2D(di, yi+1, xi+1) and self._is_land2D(di+1, yi+1, xi) \
                and self._is_land2D(di+1, yi+1, xi+1) and eta < 1:
            if self.U.interp_method == 'partialslip':
                f_u = f_u * (1 - .5 * eta) / (1 - eta)
            elif self.U.interp_method == 'freeslip':
                f_u = f_u / (1 - eta)
        if self._is_land2D(di, yi, xi) and self._is_land2D(di, yi+1, xi) and self._is_land2D(di+1, yi, xi) \
                and self._is_land2D(di+1, yi+1, xi) and xsi > 0:
            if self.U.interp_method == 'partialslip':
                f_v = f_v * (.5 + .5 * xsi) / xsi
            elif self.U.interp_method == 'freeslip':
                f_v = f_v / xsi
        if self._is_land2D(di, yi, xi+1) and self._is_land2D(di, yi+1, xi+1) and self._is_land2D(di+1, yi, xi+1) \
                and self._is_land2D(di+1, yi+1, xi+1) and xsi < 1:
            if self.U.interp_method == 'partialslip':
                f_v = f_v * (1 - .5 * xsi) / (1 - xsi)
            elif self.U.interp_method == 'freeslip':
                f_v = f_v / (1 - xsi)
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
        return u, v

    def spatial_interpolate(self, ti, z, y, x, time, particle):
        if self.U.interp_method == 'cgrid_velocity':
            return self.spatial_c_grid_interpolation2D(ti, z, y, x, time, particle)
        elif self.U.interp_method == 'partial_slip':
            return self.spatial_slip_interpolation(ti, z, y, x, time, particle)
        return self.spatial_slip_interpolation(ti, z, y, x, time, particle)

    def eval(self, time, z, y, x, particle=None):
        if self.U.interp_method not in ['cgrid_velocity', 'partialslip', 'freeslip']:
            u = self.U.eval(time, z, y, x, particle=particle, applyConversion=False)
            v = self.V.eval(time, z, y, x, particle=particle, applyConversion=False)
            return (u, v)
        else:
            grid = self.U.grid
            (ti, periods) = self.U.time_index(time)
            time -= periods*(grid.time_full[-1]-grid.time_full[0])
            if ti < grid.tdim-1 and time > grid.time[ti]:
                t0 = grid.time[ti]
                t1 = grid.time[ti + 1]
                (u0, v0) = self.spatial_interpolate(ti, z, y, x, time, particle=particle)
                (u1, v1) = self.spatial_interpolate(ti + 1, z, y, x, time, particle=particle)
                u = u0 + (u1 - u0) * ((time - t0) / (t1 - t0))
                v = v0 + (v1 - v0) * ((time - t0) / (t1 - t0))
                return (u, v)
            else:
                # Skip temporal interpolation if time is outside
                # of the defined time range or if we have hit an
                # exact value in the time array.
                return self.spatial_interpolate(ti, z, y, x, grid.time[ti], particle=particle)



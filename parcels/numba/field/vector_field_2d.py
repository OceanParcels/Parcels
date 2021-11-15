from numba.experimental import jitclass
from numba.core.typing.asnumbatype import as_numba_type
from .field import NumbaField
import numba as nb
import numpy as np
from parcels.numba.grid.base import GridCode
import math
import parcels.tools.interpolation_utils as ip
from parcels.numba.utils import _numba_isclose


class NumbaBaseVectorField():
    def jacobian(self, xsi, eta, px, py):
        dphidxsi = np.array([eta-1, 1-eta, eta, -eta]).astype(np.float32)
        dphideta = np.array([xsi-1, -xsi, xsi, 1-xsi]).astype(np.float32)

        dxdxsi = np.dot(px, dphidxsi)
        dxdeta = np.dot(px, dphideta)
        dydxsi = np.dot(py, dphidxsi)
        dydeta = np.dot(py, dphideta)
        jac = dxdxsi*dydeta - dxdeta*dydxsi
        return jac

    def spatial_c_grid_interpolation2D(self, ti, z, y, x, time, particle=None):
        grid = self.U.grid
        (xsi, eta, zeta, xi, yi, zi) = grid.search_indices(x, y, z, ti, time)

        px, py = grid.get_pxy(xi, yi)

        if grid.mesh == 'spherical':
            px[0] = px[0]+360 if px[0] < x-225 else px[0]
            px[0] = px[0]-360 if px[0] > x+225 else px[0]
            px[1:] = np.where(px[1:] - px[0] > 180, px[1:]-360, px[1:])
            px[1:] = np.where(-px[1:] + px[0] > 180, px[1:]+360, px[1:])
        xx = (1-xsi)*(1-eta) * px[0] + xsi*(1-eta) * px[1] + xsi*eta * px[2] + (1-xsi)*eta * px[3]
        assert abs(xx-x) < 1e-4
        c1 = self.dist(px[0], px[1], py[0], py[1], grid.mesh, np.dot(ip.phi2D_lin(xsi, 0.), py))
        c2 = self.dist(px[1], px[2], py[1], py[2], grid.mesh, np.dot(ip.phi2D_lin(1., eta), py))
        c3 = self.dist(px[2], px[3], py[2], py[3], grid.mesh, np.dot(ip.phi2D_lin(xsi, 1.), py))
        c4 = self.dist(px[3], px[0], py[3], py[0], grid.mesh, np.dot(ip.phi2D_lin(0., eta), py))

        if self.gridindexingtype == 'nemo':
            U0 = self.U.data[ti, zi, yi+1, xi] * c4
            U1 = self.U.data[ti, zi, yi+1, xi+1] * c2
            V0 = self.V.data[ti, zi, yi, xi+1] * c1
            V1 = self.V.data[ti, zi, yi+1, xi+1] * c3
        elif self.gridindexingtype == 'mitgcm':
            U0 = self.U.data[ti, zi, yi, xi] * c4
            U1 = self.U.data[ti, zi, yi, xi + 1] * c2
            V0 = self.V.data[ti, zi, yi, xi] * c1
            V1 = self.V.data[ti, zi, yi + 1, xi] * c3
        U = (1-xsi) * U0 + xsi * U1
        V = (1-eta) * V0 + eta * V1
        rad = np.pi/180.
        deg2m = 1852 * 60.
        meshJac = (deg2m * deg2m * math.cos(rad * y)) if grid.mesh == 'spherical' else 1
        jac = self.jacobian(xsi, eta, px, py) * meshJac

        u = ((-(1-eta) * U - (1-xsi) * V) * px[0]
             + ((1-eta) * U - xsi * V) * px[1]
             + (eta * U + xsi * V) * px[2]
             + (-eta * U + (1-xsi) * V) * px[3]) / jac
        v = ((-(1-eta) * U - (1-xsi) * V) * py[0]
             + ((1-eta) * U - xsi * V) * py[1]
             + (eta * U + xsi * V) * py[2]
             + (-eta * U + (1-xsi) * V) * py[3]) / jac
        return (u, v)

    def _is_land2D(self, di, yi, xi):
        if di < self.U.grid.zdim and yi < np.shape(self.U.data)[-2] and xi < np.shape(self.U.data)[-1]:
            return _numba_isclose(self.U.data[0, di, yi, xi], 0.) and _numba_isclose(self.V.data[0, di, yi, xi], 0.)
        else:
            return True

    def __getitem__(self, key):
        return self.eval(*key)


@jitclass(spec=[
    ("U", as_numba_type(NumbaField)),
    ("V", as_numba_type(NumbaField)),
    ("vector_type", nb.types.string),
    ("name", nb.types.string),
    ("gridindexingtype", nb.types.string),
])
class NumbaVectorField2D(NumbaBaseVectorField):
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



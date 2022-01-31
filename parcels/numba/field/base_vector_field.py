import math
import numpy as np
import numba as nb

from parcels.numba.utils import _numba_isclose
import parcels.tools.interpolation_utils as ip


class NumbaBaseVectorField():
    """Class with all functions that are shared between vector fields."""
    def jacobian(self, xsi, eta, px, py):
        dphidxsi = np.array([eta-1, 1-eta, eta, -eta]).astype(nb.float64)
        dphideta = np.array([xsi-1, -xsi, xsi, 1-xsi]).astype(nb.float64)

        dxdxsi = np.dot(px, dphidxsi)
        dxdeta = np.dot(px, dphideta)
        dydxsi = np.dot(py, dphidxsi)
        dydeta = np.dot(py, dphideta)
        jac = dxdxsi*dydeta - dxdeta*dydxsi
        return jac

    def spatial_c_grid_interpolation2D(self, ti, z, y, x, time, particle=None):
        """Copied from the originals"""
        grid = self.U.grid
        (xsi, eta, _zeta, xi, yi, zi) = grid.search_indices(x, y, z, ti, time)

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
        """Check if grid cell is on land"""
        if di < self.U.grid.zdim and yi < np.shape(self.U.data)[-2] and xi < np.shape(self.U.data)[-1]:
            return _numba_isclose(self.U.data[0, di, yi, xi], 0.) and _numba_isclose(self.V.data[0, di, yi, xi], 0.)
        else:
            return True

    def __getitem__(self, key):
        return self.eval(*key)



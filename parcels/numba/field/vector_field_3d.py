from numba.experimental import jitclass
from numba.core.typing.asnumbatype import as_numba_type
from .field import NumbaField
import numba as nb
import numpy as np
from parcels.numba.grid.base import GridCode
import math
import parcels.tools.interpolation_utils as ip
from parcels.numba.field.base_vector_field import NumbaBaseVectorField


class NumbaVectorField3D():
    @staticmethod
    def _class(U):
        numba_class = U.numba_class
        vfield_class = jitclass(_NumbaVectorField3D, spec=[
            ("U", as_numba_type(numba_class)),
            ("V", as_numba_type(numba_class)),
            ("W", as_numba_type(numba_class)),
            ("vector_type", nb.types.string),
            ("name", nb.types.string),
            ("gridindexingtype", nb.types.string),
        ])
        return vfield_class

    def create(self, name, U, V, W):
        return self._class(name, U, V, W)

# @jitclass(spec=[
#     ("U", as_numba_type(NumbaField)),
#     ("V", as_numba_type(NumbaField)),
#     ("W", as_numba_type(NumbaField)),
#     ("vector_type", nb.types.string),
#     ("name", nb.types.string),
# ])


class _NumbaVectorField3D(NumbaBaseVectorField):
    def __init__(self, name, U, V, W):
        self.U = U
        self.V = V
        self.W = W
        self.name = name
        self.gridindexingtype = U.gridindexingtype

    def dist(self, lon1, lon2, lat1, lat2, mesh, lat):
        if mesh == 'spherical':
            rad = np.pi/180.
            deg2m = 1852 * 60.
            return np.sqrt(((lon2-lon1)*deg2m*math.cos(rad * lat))**2 + ((lat2-lat1)*deg2m)**2)
        else:
            return np.sqrt((lon2-lon1)**2 + (lat2-lat1)**2)

    def jacobian(self, xsi, eta, px, py):
        dphidxsi = np.array([eta-1, 1-eta, eta, -eta]).astype(nb.float64)
        dphideta = np.array([xsi-1, -xsi, xsi, 1-xsi]).astype(nb.float64)

        dxdxsi = np.dot(px, dphidxsi)
        dxdeta = np.dot(px, dphideta)
        dydxsi = np.dot(py, dphidxsi)
        dydeta = np.dot(py, dphideta)
        jac = dxdxsi*dydeta - dxdeta*dydxsi
        return jac

    def spatial_c_grid_interpolation3D_full(self, ti, z, y, x, time, particle=None):
        grid = self.U.grid
        (xsi, eta, zet, xi, yi, zi) = self.U.grid.search_indices(x, y, z, ti, time, particle=particle)

        px, py = grid.get_pxy(xi, yi)
#         if grid.gtype in [GridCode.RectilinearSGrid, GridCode.RectilinearZGrid]:
#             px = np.array([grid.lon[xi], grid.lon[xi+1], grid.lon[xi+1], grid.lon[xi]])
#             py = np.array([grid.lat[yi], grid.lat[yi], grid.lat[yi+1], grid.lat[yi+1]])
#         else:
#             px = np.array([grid.lon[yi, xi], grid.lon[yi, xi+1], grid.lon[yi+1, xi+1], grid.lon[yi+1, xi]])
#             py = np.array([grid.lat[yi, xi], grid.lat[yi, xi+1], grid.lat[yi+1, xi+1], grid.lat[yi+1, xi]])

        if grid.mesh == 'spherical':
            px[0] = px[0]+360 if px[0] < x-225 else px[0]
            px[0] = px[0]-360 if px[0] > x+225 else px[0]
            px[1:] = np.where(px[1:] - px[0] > 180, px[1:]-360, px[1:])
            px[1:] = np.where(-px[1:] + px[0] > 180, px[1:]+360, px[1:])
        xx = (1-xsi)*(1-eta) * px[0] + xsi*(1-eta) * px[1] + xsi*eta * px[2] + (1-xsi)*eta * px[3]
        assert abs(xx-x) < 1e-4

        px = np.concatenate((px, px))
        py = np.concatenate((py, py))
#         if grid.z4d:
        pz = np.array([grid.depth[0, zi, yi, xi], grid.depth[0, zi, yi, xi+1],
                       grid.depth[0, zi, yi+1, xi+1], grid.depth[0, zi, yi+1, xi],
                       grid.depth[0, zi+1, yi, xi], grid.depth[0, zi+1, yi, xi+1],
                       grid.depth[0, zi+1, yi+1, xi+1], grid.depth[0, zi+1, yi+1, xi]])
#         else:
#             pz = np.array([grid.depth[zi, yi, xi], grid.depth[zi, yi, xi+1], grid.depth[zi, yi+1, xi+1], grid.depth[zi, yi+1, xi],
#                            grid.depth[zi+1, yi, xi], grid.depth[zi+1, yi, xi+1], grid.depth[zi+1, yi+1, xi+1], grid.depth[zi+1, yi+1, xi]])

        u0 = self.U.data[ti, zi, yi+1, xi]
        u1 = self.U.data[ti, zi, yi+1, xi+1]
        v0 = self.V.data[ti, zi, yi, xi+1]
        v1 = self.V.data[ti, zi, yi+1, xi+1]
        w0 = self.W.data[ti, zi, yi+1, xi+1]
        w1 = self.W.data[ti, zi+1, yi+1, xi+1]

        U0 = u0 * ip.jacobian3D_lin_face(px, py, pz, 0, eta, zet, 'zonal', grid.mesh)
        U1 = u1 * ip.jacobian3D_lin_face(px, py, pz, 1, eta, zet, 'zonal', grid.mesh)
        V0 = v0 * ip.jacobian3D_lin_face(px, py, pz, xsi, 0, zet, 'meridional', grid.mesh)
        V1 = v1 * ip.jacobian3D_lin_face(px, py, pz, xsi, 1, zet, 'meridional', grid.mesh)
        W0 = w0 * ip.jacobian3D_lin_face(px, py, pz, xsi, eta, 0, 'vertical', grid.mesh)
        W1 = w1 * ip.jacobian3D_lin_face(px, py, pz, xsi, eta, 1, 'vertical', grid.mesh)

        # Computing fluxes in half left hexahedron -> flux_u05
        xx = np.array([px[0], (px[0]+px[1])/2, (px[2]+px[3])/2, px[3], px[4], (px[4]+px[5])/2, (px[6]+px[7])/2, px[7]]).astype(nb.float64)
        yy = np.array([py[0], (py[0]+py[1])/2, (py[2]+py[3])/2, py[3], py[4], (py[4]+py[5])/2, (py[6]+py[7])/2, py[7]]).astype(nb.float64)
        zz = np.array([pz[0], (pz[0]+pz[1])/2, (pz[2]+pz[3])/2, pz[3], pz[4], (pz[4]+pz[5])/2, (pz[6]+pz[7])/2, pz[7]]).astype(nb.float64)
        flux_u0 = u0 * ip.jacobian3D_lin_face(xx, yy, zz, 0, .5, .5, 'zonal', grid.mesh)
        flux_v0_halfx = v0 * ip.jacobian3D_lin_face(xx, yy, zz, .5, 0, .5, 'meridional', grid.mesh)
        flux_v1_halfx = v1 * ip.jacobian3D_lin_face(xx, yy, zz, .5, 1, .5, 'meridional', grid.mesh)
        flux_w0_halfx = w0 * ip.jacobian3D_lin_face(xx, yy, zz, .5, .5, 0, 'vertical', grid.mesh)
        flux_w1_halfx = w1 * ip.jacobian3D_lin_face(xx, yy, zz, .5, .5, 1, 'vertical', grid.mesh)
        flux_u05 = flux_u0 + flux_v0_halfx - flux_v1_halfx + flux_w0_halfx - flux_w1_halfx

        # Computing fluxes in half front hexahedron -> flux_v05
        xx = np.array([px[0], px[1], (px[1]+px[2])/2, (px[0]+px[3])/2, px[4], px[5], (px[5]+px[6])/2, (px[4]+px[7])/2]).astype(nb.float64)
        yy = np.array([py[0], py[1], (py[1]+py[2])/2, (py[0]+py[3])/2, py[4], py[5], (py[5]+py[6])/2, (py[4]+py[7])/2]).astype(nb.float64)
        zz = np.array([pz[0], pz[1], (pz[1]+pz[2])/2, (pz[0]+pz[3])/2, pz[4], pz[5], (pz[5]+pz[6])/2, (pz[4]+pz[7])/2]).astype(nb.float64)
        flux_u0_halfy = u0 * ip.jacobian3D_lin_face(xx, yy, zz, 0, .5, .5, 'zonal', grid.mesh)
        flux_u1_halfy = u1 * ip.jacobian3D_lin_face(xx, yy, zz, 1, .5, .5, 'zonal', grid.mesh)
        flux_v0 = v0 * ip.jacobian3D_lin_face(xx, yy, zz, .5, 0, .5, 'meridional', grid.mesh)
        flux_w0_halfy = w0 * ip.jacobian3D_lin_face(xx, yy, zz, .5, .5, 0, 'vertical', grid.mesh)
        flux_w1_halfy = w1 * ip.jacobian3D_lin_face(xx, yy, zz, .5, .5, 1, 'vertical', grid.mesh)
        flux_v05 = flux_u0_halfy - flux_u1_halfy + flux_v0 + flux_w0_halfy - flux_w1_halfy

        # Computing fluxes in half lower hexahedron -> flux_w05
        xx = np.array([px[0], px[1], px[2], px[3], (px[0]+px[4])/2, (px[1]+px[5])/2, (px[2]+px[6])/2, (px[3]+px[7])/2]).astype(nb.float64)
        yy = np.array([py[0], py[1], py[2], py[3], (py[0]+py[4])/2, (py[1]+py[5])/2, (py[2]+py[6])/2, (py[3]+py[7])/2]).astype(nb.float64)
        zz = np.array([pz[0], pz[1], pz[2], pz[3], (pz[0]+pz[4])/2, (pz[1]+pz[5])/2, (pz[2]+pz[6])/2, (pz[3]+pz[7])/2]).astype(nb.float64)
        flux_u0_halfz = u0 * ip.jacobian3D_lin_face(xx, yy, zz, 0, .5, .5, 'zonal', grid.mesh)
        flux_u1_halfz = u1 * ip.jacobian3D_lin_face(xx, yy, zz, 1, .5, .5, 'zonal', grid.mesh)
        flux_v0_halfz = v0 * ip.jacobian3D_lin_face(xx, yy, zz, .5, 0, .5, 'meridional', grid.mesh)
        flux_v1_halfz = v1 * ip.jacobian3D_lin_face(xx, yy, zz, .5, 1, .5, 'meridional', grid.mesh)
        flux_w0 = w0 * ip.jacobian3D_lin_face(xx, yy, zz, .5, .5, 0, 'vertical', grid.mesh)
        flux_w05 = flux_u0_halfz - flux_u1_halfz + flux_v0_halfz - flux_v1_halfz + flux_w0

        surf_u05 = ip.jacobian3D_lin_face(px, py, pz, .5, .5, .5, 'zonal', grid.mesh)
        jac_u05 = ip.jacobian3D_lin_face(px, py, pz, .5, eta, zet, 'zonal', grid.mesh)
        U05 = flux_u05 / surf_u05 * jac_u05

        surf_v05 = ip.jacobian3D_lin_face(px, py, pz, .5, .5, .5, 'meridional', grid.mesh)
        jac_v05 = ip.jacobian3D_lin_face(px, py, pz, xsi, .5, zet, 'meridional', grid.mesh)
        V05 = flux_v05 / surf_v05 * jac_v05

        surf_w05 = ip.jacobian3D_lin_face(px, py, pz, .5, .5, .5, 'vertical', grid.mesh)
        jac_w05 = ip.jacobian3D_lin_face(px, py, pz, xsi, eta, .5, 'vertical', grid.mesh)
        W05 = flux_w05 / surf_w05 * jac_w05

        jac = ip.jacobian3D_lin(px, py, pz, xsi, eta, zet, grid.mesh)
        dxsidt = ip.interpolate(ip.phi1D_quad, np.array([U0, U05, U1]).astype(nb.float64), xsi) / jac
        detadt = ip.interpolate(ip.phi1D_quad, np.array([V0, V05, V1]).astype(nb.float64), eta) / jac
        dzetdt = ip.interpolate(ip.phi1D_quad, np.array([W0, W05, W1]).astype(nb.float64), zet) / jac

        dphidxsi, dphideta, dphidzet = ip.dphidxsi3D_lin(xsi, eta, zet)

        u = np.dot(dphidxsi, px) * dxsidt + np.dot(dphideta, px) * detadt + np.dot(dphidzet, px) * dzetdt
        v = np.dot(dphidxsi, py) * dxsidt + np.dot(dphideta, py) * detadt + np.dot(dphidzet, py) * dzetdt
        w = np.dot(dphidxsi, pz) * dxsidt + np.dot(dphideta, pz) * detadt + np.dot(dphidzet, pz) * dzetdt

#         if isinstance(u, da.core.Array):
#             u = u.compute()
#             v = v.compute()
#             w = w.compute()
        return (u, v, w)

    def spatial_c_grid_interpolation3D(self, ti, z, y, x, time, particle=None):
        """
        +---+---+---+
        |   |V1 |   |
        +---+---+---+
        |U0 |   |U1 |
        +---+---+---+
        |   |V0 |   |
        +---+---+---+

        The interpolation is done in the following by
        interpolating linearly U depending on the longitude coordinate and
        interpolating linearly V depending on the latitude coordinate.
        Curvilinear grids are treated properly, since the element is projected to a rectilinear parent element.
        """
        if self.U.grid.gtype in [GridCode.RectilinearSGrid, GridCode.CurvilinearSGrid]:
            (u, v, w) = self.spatial_c_grid_interpolation3D_full(ti, z, y, x, time, particle=particle)
        else:
            (u, v) = self.spatial_c_grid_interpolation2D(ti, z, y, x, time, particle=particle)
            w = self.W.eval(time, z, y, x, particle=particle, applyConversion=False)
            # w = self.W.units.to_target(w, x, y, z)
        return (u, v, w)

    def spatial_slip_interpolation(self, ti, z, y, x, time, particle=None):
        (xsi, eta, zeta, xi, yi, zi) = self.U.grid.search_indices(x, y, z, ti, time, particle=particle)
        di = ti if self.U.grid.zdim == 1 else zi  # general third dimension

        f_u, f_v, f_w = 1, 1, 1
        if self._is_land2D(di, yi, xi) and self._is_land2D(di, yi, xi+1) and self._is_land2D(di+1, yi, xi) \
                and self._is_land2D(di+1, yi, xi+1) and eta > 0:
            if self.U.interp_method == 'partialslip':
                f_u = f_u * (.5 + .5 * eta) / eta
                f_w = f_w * (.5 + .5 * eta) / eta
            elif self.U.interp_method == 'freeslip':
                f_u = f_u / eta
                f_w = f_w / eta
        if self._is_land2D(di, yi+1, xi) and self._is_land2D(di, yi+1, xi+1) and self._is_land2D(di+1, yi+1, xi) \
                and self._is_land2D(di+1, yi+1, xi+1) and eta < 1:
            if self.U.interp_method == 'partialslip':
                f_u = f_u * (1 - .5 * eta) / (1 - eta)
                f_w = f_w * (1 - .5 * eta) / (1 - eta)
            elif self.U.interp_method == 'freeslip':
                f_u = f_u / (1 - eta)
                f_w = f_w / (1 - eta)
        if self._is_land2D(di, yi, xi) and self._is_land2D(di, yi+1, xi) and self._is_land2D(di+1, yi, xi) \
                and self._is_land2D(di+1, yi+1, xi) and xsi > 0:
            if self.U.interp_method == 'partialslip':
                f_v = f_v * (.5 + .5 * xsi) / xsi
                f_w = f_w * (.5 + .5 * xsi) / xsi
            elif self.U.interp_method == 'freeslip':
                f_v = f_v / xsi
                f_w = f_w / xsi
        if self._is_land2D(di, yi, xi+1) and self._is_land2D(di, yi+1, xi+1) and self._is_land2D(di+1, yi, xi+1) \
                and self._is_land2D(di+1, yi+1, xi+1) and xsi < 1:
            if self.U.interp_method == 'partialslip':
                f_v = f_v * (1 - .5 * xsi) / (1 - xsi)
                f_w = f_w * (1 - .5 * xsi) / (1 - xsi)
            elif self.U.interp_method == 'freeslip':
                f_v = f_v / (1 - xsi)
                f_w = f_w / (1 - xsi)
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
        w = f_w * self.W.eval(time, z, y, x, particle)
        return u, v, w

    def spatial_interpolate(self, ti, z, y, x, time, particle):
        if self.U.interp_method == 'cgrid_velocity':
            return self.spatial_c_grid_interpolation3D(ti, z, y, x, time, particle)
        elif self.U.interp_method == 'partial_slip':
            return self.spatial_slip_interpolation(ti, z, y, x, time, particle)
        return self.spatial_slip_interpolation(ti, z, y, x, time, particle)

    def eval(self, time, z, y, x, particle=None):
        if self.U.interp_method not in ['cgrid_velocity', 'partialslip', 'freeslip']:
            u = self.U.eval(time, z, y, x, particle=particle, applyConversion=False)
            v = self.V.eval(time, z, y, x, particle=particle, applyConversion=False)
            # u = self.U.units.to_target(u, x, y, z)
            # v = self.V.units.to_target(v, x, y, z)
            w = self.W.eval(time, z, y, x, particle=particle, applyConversion=False)
                # w = self.W.units.to_target(w, x, y, z)
            return (u, v, w)
        else:
            grid = self.U.grid
            (ti, periods) = self.U.time_index(time)
            time -= periods*(grid.time_full[-1]-grid.time_full[0])
            if ti < grid.tdim-1 and time > grid.time[ti]:
                t0 = grid.time[ti]
                t1 = grid.time[ti + 1]
                (u0, v0, w0) = self.spatial_interpolate(ti, z, y, x, time, particle=particle)
                (u1, v1, w1) = self.spatial_interpolate(ti + 1, z, y, x, time, particle=particle)
                w = w0 + (w1 - w0) * ((time - t0) / (t1 - t0))
                u = u0 + (u1 - u0) * ((time - t0) / (t1 - t0))
                v = v0 + (v1 - v0) * ((time - t0) / (t1 - t0))
                return (u, v, w)
            else:
                # Skip temporal interpolation if time is outside
                # of the defined time range or if we have hit an
                # exact value in the time array.
                return self.spatial_interpolate(ti, z, y, x, grid.time[ti], particle=particle)

    def __getitem__(self, key):
        return self.eval(*key)

#     def __getitem__(self, key):
#         if _isParticle(key):
#             return self.eval(key.time, key.depth, key.lat, key.lon, key)
#         else:
#             return self.eval(*key)

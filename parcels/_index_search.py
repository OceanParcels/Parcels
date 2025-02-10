from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from parcels._typing import (
    GridIndexingType,
    InterpMethodOption,
)
from parcels.tools.statuscodes import (
    FieldOutOfBoundError,
    FieldOutOfBoundSurfaceError,
    _raise_field_out_of_bound_error,
    _raise_field_out_of_bound_surface_error,
    _raise_field_sampling_error,
)

from .grid import GridType

if TYPE_CHECKING:
    from .field import Field
    from .grid import Grid


def search_indices_vertical_z(grid: Grid, gridindexingtype: GridIndexingType, z: float):
    if grid.depth[-1] > grid.depth[0]:
        if z < grid.depth[0]:
            # Since MOM5 is indexed at cell bottom, allow z at depth[0] - dz where dz = (depth[1] - depth[0])
            if gridindexingtype == "mom5" and z > 2 * grid.depth[0] - grid.depth[1]:
                return (-1, z / grid.depth[0])
            else:
                _raise_field_out_of_bound_surface_error(z, None, None)
        elif z > grid.depth[-1]:
            # In case of CROCO, allow particles in last (uppermost) layer using depth[-1]
            if gridindexingtype in ["croco"] and z < 0:
                return (-2, 1)
            _raise_field_out_of_bound_error(z, None, None)
        depth_indices = grid.depth < z
        if z >= grid.depth[-1]:
            zi = len(grid.depth) - 2
        else:
            zi = depth_indices.argmin() - 1 if z > grid.depth[0] else 0
    else:
        if z > grid.depth[0]:
            _raise_field_out_of_bound_surface_error(z, None, None)
        elif z < grid.depth[-1]:
            _raise_field_out_of_bound_error(z, None, None)
        depth_indices = grid.depth > z
        if z <= grid.depth[-1]:
            zi = len(grid.depth) - 2
        else:
            zi = depth_indices.argmin() - 1 if z < grid.depth[0] else 0
    zeta = (z - grid.depth[zi]) / (grid.depth[zi + 1] - grid.depth[zi])
    while zeta > 1:
        zi += 1
        zeta = (z - grid.depth[zi]) / (grid.depth[zi + 1] - grid.depth[zi])
    while zeta < 0:
        zi -= 1
        zeta = (z - grid.depth[zi]) / (grid.depth[zi + 1] - grid.depth[zi])
    return (zi, zeta)


def search_indices_vertical_s(
    grid: Grid,
    interp_method: InterpMethodOption,
    time: float,
    z: float,
    y: float,
    x: float,
    ti: int,
    yi: int,
    xi: int,
    eta: float,
    xsi: float,
):
    if interp_method in ["bgrid_velocity", "bgrid_w_velocity", "bgrid_tracer"]:
        xsi = 1
        eta = 1
    if time < grid.time[ti]:
        ti -= 1
    if grid._z4d:  # type: ignore[attr-defined]
        if ti == len(grid.time) - 1:
            depth_vector = (
                (1 - xsi) * (1 - eta) * grid.depth[-1, :, yi, xi]
                + xsi * (1 - eta) * grid.depth[-1, :, yi, xi + 1]
                + xsi * eta * grid.depth[-1, :, yi + 1, xi + 1]
                + (1 - xsi) * eta * grid.depth[-1, :, yi + 1, xi]
            )
        else:
            dv2 = (
                (1 - xsi) * (1 - eta) * grid.depth[ti : ti + 2, :, yi, xi]
                + xsi * (1 - eta) * grid.depth[ti : ti + 2, :, yi, xi + 1]
                + xsi * eta * grid.depth[ti : ti + 2, :, yi + 1, xi + 1]
                + (1 - xsi) * eta * grid.depth[ti : ti + 2, :, yi + 1, xi]
            )
            tt = (time - grid.time[ti]) / (grid.time[ti + 1] - grid.time[ti])
            assert tt >= 0 and tt <= 1, "Vertical s grid is being wrongly interpolated in time"
            depth_vector = dv2[0, :] * (1 - tt) + dv2[1, :] * tt
    else:
        depth_vector = (
            (1 - xsi) * (1 - eta) * grid.depth[:, yi, xi]
            + xsi * (1 - eta) * grid.depth[:, yi, xi + 1]
            + xsi * eta * grid.depth[:, yi + 1, xi + 1]
            + (1 - xsi) * eta * grid.depth[:, yi + 1, xi]
        )
    z = np.float32(z)  # type: ignore # TODO: remove type ignore once we migrate to float64

    if depth_vector[-1] > depth_vector[0]:
        if z < depth_vector[0]:
            _raise_field_out_of_bound_error(z, None, None)
        elif z > depth_vector[-1]:
            _raise_field_out_of_bound_error(z, None, None)
        depth_indices = depth_vector < z
        if z >= depth_vector[-1]:
            zi = len(depth_vector) - 2
        else:
            zi = depth_indices.argmin() - 1 if z > depth_vector[0] else 0
    else:
        if z > depth_vector[0]:
            _raise_field_out_of_bound_error(z, None, None)
        elif z < depth_vector[-1]:
            _raise_field_out_of_bound_error(z, None, None)
        depth_indices = depth_vector > z
        if z <= depth_vector[-1]:
            zi = len(depth_vector) - 2
        else:
            zi = depth_indices.argmin() - 1 if z < depth_vector[0] else 0
    zeta = (z - depth_vector[zi]) / (depth_vector[zi + 1] - depth_vector[zi])
    while zeta > 1:
        zi += 1
        zeta = (z - depth_vector[zi]) / (depth_vector[zi + 1] - depth_vector[zi])
    while zeta < 0:
        zi -= 1
        zeta = (z - depth_vector[zi]) / (depth_vector[zi + 1] - depth_vector[zi])
    return (zi, zeta)


def _search_indices_rectilinear(
    field: Field, time: float, z: float, y: float, x: float, ti=-1, particle=None, search2D=False
):
    grid = field.grid

    if grid.xdim > 1 and (not grid.zonal_periodic):
        if x < grid.lonlat_minmax[0] or x > grid.lonlat_minmax[1]:
            _raise_field_out_of_bound_error(z, y, x)
    if grid.ydim > 1 and (y < grid.lonlat_minmax[2] or y > grid.lonlat_minmax[3]):
        _raise_field_out_of_bound_error(z, y, x)

    if grid.xdim > 1:
        if grid.mesh != "spherical":
            lon_index = grid.lon < x
            if lon_index.all():
                xi = len(grid.lon) - 2
            else:
                xi = lon_index.argmin() - 1 if lon_index.any() else 0
            xsi = (x - grid.lon[xi]) / (grid.lon[xi + 1] - grid.lon[xi])
            if xsi < 0:
                xi -= 1
                xsi = (x - grid.lon[xi]) / (grid.lon[xi + 1] - grid.lon[xi])
            elif xsi > 1:
                xi += 1
                xsi = (x - grid.lon[xi]) / (grid.lon[xi + 1] - grid.lon[xi])
        else:
            lon_fixed = grid.lon.copy()
            indices = lon_fixed >= lon_fixed[0]
            if not indices.all():
                lon_fixed[indices.argmin() :] += 360
            if x < lon_fixed[0]:
                lon_fixed -= 360

            lon_index = lon_fixed < x
            if lon_index.all():
                xi = len(lon_fixed) - 2
            else:
                xi = lon_index.argmin() - 1 if lon_index.any() else 0
            xsi = (x - lon_fixed[xi]) / (lon_fixed[xi + 1] - lon_fixed[xi])
            if xsi < 0:
                xi -= 1
                xsi = (x - lon_fixed[xi]) / (lon_fixed[xi + 1] - lon_fixed[xi])
            elif xsi > 1:
                xi += 1
                xsi = (x - lon_fixed[xi]) / (lon_fixed[xi + 1] - lon_fixed[xi])
    else:
        xi, xsi = -1, 0

    if grid.ydim > 1:
        lat_index = grid.lat < y
        if lat_index.all():
            yi = len(grid.lat) - 2
        else:
            yi = lat_index.argmin() - 1 if lat_index.any() else 0

        eta = (y - grid.lat[yi]) / (grid.lat[yi + 1] - grid.lat[yi])
        if eta < 0:
            yi -= 1
            eta = (y - grid.lat[yi]) / (grid.lat[yi + 1] - grid.lat[yi])
        elif eta > 1:
            yi += 1
            eta = (y - grid.lat[yi]) / (grid.lat[yi + 1] - grid.lat[yi])
    else:
        yi, eta = -1, 0

    if grid.zdim > 1 and not search2D:
        if grid._gtype == GridType.RectilinearZGrid:
            try:
                (zi, zeta) = search_indices_vertical_z(field.grid, field.gridindexingtype, z)
            except FieldOutOfBoundError:
                _raise_field_out_of_bound_error(z, y, x)
            except FieldOutOfBoundSurfaceError:
                _raise_field_out_of_bound_surface_error(z, y, x)
        elif grid._gtype == GridType.RectilinearSGrid:
            (zi, zeta) = search_indices_vertical_s(field.grid, field.interp_method, time, z, y, x, ti, yi, xi, eta, xsi)
    else:
        zi, zeta = -1, 0

    if not ((0 <= xsi <= 1) and (0 <= eta <= 1) and (0 <= zeta <= 1)):
        _raise_field_sampling_error(z, y, x)

    if particle:
        particle.xi[field.igrid] = xi
        particle.yi[field.igrid] = yi
        particle.zi[field.igrid] = zi

    return (zeta, eta, xsi, zi, yi, xi)


def _search_indices_curvilinear(field: Field, time, z, y, x, ti=-1, particle=None, search2D=False):
    if particle:
        xi = particle.xi[field.igrid]
        yi = particle.yi[field.igrid]
    else:
        xi = int(field.grid.xdim / 2) - 1
        yi = int(field.grid.ydim / 2) - 1
    xsi = eta = -1.0
    grid = field.grid
    invA = np.array([[1, 0, 0, 0], [-1, 1, 0, 0], [-1, 0, 0, 1], [1, -1, 1, -1]])
    maxIterSearch = 1e6
    it = 0
    tol = 1.0e-10
    if not grid.zonal_periodic:
        if x < grid.lonlat_minmax[0] or x > grid.lonlat_minmax[1]:
            if grid.lon[0, 0] < grid.lon[0, -1]:
                _raise_field_out_of_bound_error(z, y, x)
            elif x < grid.lon[0, 0] and x > grid.lon[0, -1]:  # This prevents from crashing in [160, -160]
                _raise_field_out_of_bound_error(z, y, x)
    if y < grid.lonlat_minmax[2] or y > grid.lonlat_minmax[3]:
        _raise_field_out_of_bound_error(z, y, x)

    while xsi < -tol or xsi > 1 + tol or eta < -tol or eta > 1 + tol:
        px = np.array([grid.lon[yi, xi], grid.lon[yi, xi + 1], grid.lon[yi + 1, xi + 1], grid.lon[yi + 1, xi]])
        if grid.mesh == "spherical":
            px[0] = px[0] + 360 if px[0] < x - 225 else px[0]
            px[0] = px[0] - 360 if px[0] > x + 225 else px[0]
            px[1:] = np.where(px[1:] - px[0] > 180, px[1:] - 360, px[1:])
            px[1:] = np.where(-px[1:] + px[0] > 180, px[1:] + 360, px[1:])
        py = np.array([grid.lat[yi, xi], grid.lat[yi, xi + 1], grid.lat[yi + 1, xi + 1], grid.lat[yi + 1, xi]])
        a = np.dot(invA, px)
        b = np.dot(invA, py)

        aa = a[3] * b[2] - a[2] * b[3]
        bb = a[3] * b[0] - a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + x * b[3] - y * a[3]
        cc = a[1] * b[0] - a[0] * b[1] + x * b[1] - y * a[1]
        if abs(aa) < 1e-12:  # Rectilinear cell, or quasi
            eta = -cc / bb
        else:
            det2 = bb * bb - 4 * aa * cc
            if det2 > 0:  # so, if det is nan we keep the xsi, eta from previous iter
                det = np.sqrt(det2)
                eta = (-bb + det) / (2 * aa)
        if abs(a[1] + a[3] * eta) < 1e-12:  # this happens when recti cell rotated of 90deg
            xsi = ((y - py[0]) / (py[1] - py[0]) + (y - py[3]) / (py[2] - py[3])) * 0.5
        else:
            xsi = (x - a[0] - a[2] * eta) / (a[1] + a[3] * eta)
        if xsi < 0 and eta < 0 and xi == 0 and yi == 0:
            _raise_field_out_of_bound_error(0, y, x)
        if xsi > 1 and eta > 1 and xi == grid.xdim - 1 and yi == grid.ydim - 1:
            _raise_field_out_of_bound_error(0, y, x)
        if xsi < -tol:
            xi -= 1
        elif xsi > 1 + tol:
            xi += 1
        if eta < -tol:
            yi -= 1
        elif eta > 1 + tol:
            yi += 1
        (yi, xi) = _reconnect_bnd_indices(yi, xi, grid.ydim, grid.xdim, grid.mesh)
        it += 1
        if it > maxIterSearch:
            print(f"Correct cell not found after {maxIterSearch} iterations")
            _raise_field_out_of_bound_error(0, y, x)
    xsi = max(0.0, xsi)
    eta = max(0.0, eta)
    xsi = min(1.0, xsi)
    eta = min(1.0, eta)

    if grid.zdim > 1 and not search2D:
        if grid._gtype == GridType.CurvilinearZGrid:
            try:
                (zi, zeta) = search_indices_vertical_z(field.grid, field.gridindexingtype, z)
            except FieldOutOfBoundError:
                _raise_field_out_of_bound_error(z, y, x)
        elif grid._gtype == GridType.CurvilinearSGrid:
            (zi, zeta) = search_indices_vertical_s(field.grid, field.interp_method, time, z, y, x, ti, yi, xi, eta, xsi)
    else:
        zi = -1
        zeta = 0

    if not ((0 <= xsi <= 1) and (0 <= eta <= 1) and (0 <= zeta <= 1)):
        _raise_field_sampling_error(z, y, x)

    if particle:
        particle.xi[field.igrid] = xi
        particle.yi[field.igrid] = yi
        particle.zi[field.igrid] = zi

    return (zeta, eta, xsi, zi, yi, xi)


def _reconnect_bnd_indices(yi: int, xi: int, ydim: int, xdim: int, sphere_mesh: bool):
    if xi < 0:
        if sphere_mesh:
            xi = xdim - 2
        else:
            xi = 0
    if xi > xdim - 2:
        if sphere_mesh:
            xi = 0
        else:
            xi = xdim - 2
    if yi < 0:
        yi = 0
    if yi > ydim - 2:
        yi = ydim - 2
        if sphere_mesh:
            xi = xdim - xi
    return yi, xi

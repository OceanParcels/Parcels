from __future__ import annotations

from datetime import datetime
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
    _raise_time_extrapolation_error,
)

from .basegrid import GridType

if TYPE_CHECKING:
    from parcels.xgrid import XGrid

    from .field import Field


def _search_time_index(field: Field, time: datetime):
    """Find and return the index and relative coordinate in the time array associated with a given time.

    Parameters
    ----------
    field: Field

    time: datetime
        This is the amount of time, in seconds (time_delta), in unix epoch
    Note that we normalize to either the first or the last index
    if the sampled value is outside the time value range.
    """
    if field.time_interval is None:
        return np.zeros(shape=time.shape, dtype=np.float32), np.zeros(shape=time.shape, dtype=np.int32)

    if not field.time_interval.is_all_time_in_interval(time):
        _raise_time_extrapolation_error(time, field=None)

    ti = np.searchsorted(field.data.time.data, time, side="right") - 1
    tau = (time - field.data.time.data[ti]) / (field.data.time.data[ti + 1] - field.data.time.data[ti])
    return np.atleast_1d(tau), np.atleast_1d(ti)


def search_indices_vertical_z(depth, gridindexingtype: GridIndexingType, z: float):
    if depth[-1] > depth[0]:
        if z < depth[0]:
            # Since MOM5 is indexed at cell bottom, allow z at depth[0] - dz where dz = (depth[1] - depth[0])
            if gridindexingtype == "mom5" and z > 2 * depth[0] - depth[1]:
                return (-1, z / depth[0])
            else:
                _raise_field_out_of_bound_surface_error(z, None, None)
        elif z > depth[-1]:
            # In case of CROCO, allow particles in last (uppermost) layer using depth[-1]
            if gridindexingtype in ["croco"] and z < 0:
                return (-2, 1)
            _raise_field_out_of_bound_error(z, None, None)
        depth_indices = depth < z
        if z >= depth[-1]:
            zi = len(depth) - 2
        else:
            zi = depth_indices.argmin() - 1 if z > depth[0] else 0
    else:
        if z > depth[0]:
            _raise_field_out_of_bound_surface_error(z, None, None)
        elif z < depth[-1]:
            _raise_field_out_of_bound_error(z, None, None)
        depth_indices = depth > z
        if z <= depth[-1]:
            zi = len(depth) - 2
        else:
            zi = depth_indices.argmin() - 1 if z < depth[0] else 0
    zeta = (z - depth[zi]) / (depth[zi + 1] - depth[zi])
    while zeta > 1:
        zi += 1
        zeta = (z - depth[zi]) / (depth[zi + 1] - depth[zi])
    while zeta < 0:
        zi -= 1
        zeta = (z - depth[zi]) / (depth[zi + 1] - depth[zi])
    return (zi, zeta)


## TODO :  Still need to implement the search_indices_vertical_s function
def search_indices_vertical_s(
    field: Field,
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
    if time < field.time[ti]:
        ti -= 1
    if field._z4d:  # type: ignore[attr-defined]
        if ti == len(field.time) - 1:
            depth_vector = (
                (1 - xsi) * (1 - eta) * field.depth[-1, :, yi, xi]
                + xsi * (1 - eta) * field.depth[-1, :, yi, xi + 1]
                + xsi * eta * field.depth[-1, :, yi + 1, xi + 1]
                + (1 - xsi) * eta * field.depth[-1, :, yi + 1, xi]
            )
        else:
            dv2 = (
                (1 - xsi) * (1 - eta) * field.depth[ti : ti + 2, :, yi, xi]
                + xsi * (1 - eta) * field.depth[ti : ti + 2, :, yi, xi + 1]
                + xsi * eta * field.depth[ti : ti + 2, :, yi + 1, xi + 1]
                + (1 - xsi) * eta * field.depth[ti : ti + 2, :, yi + 1, xi]
            )
            tt = (time - field.time[ti]) / (field.time[ti + 1] - field.time[ti])
            assert tt >= 0 and tt <= 1, "Vertical s grid is being wrongly interpolated in time"
            depth_vector = dv2[0, :] * (1 - tt) + dv2[1, :] * tt
    else:
        depth_vector = (
            (1 - xsi) * (1 - eta) * field.depth[:, yi, xi]
            + xsi * (1 - eta) * field.depth[:, yi, xi + 1]
            + xsi * eta * field.depth[:, yi + 1, xi + 1]
            + (1 - xsi) * eta * field.depth[:, yi + 1, xi]
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
    field: Field, time: datetime, z: float, y: float, x: float, ti: int, ei: int | None = None, search2D=False
):
    # TODO : If ei is provided, check if particle is in the same cell
    if field.xdim > 1 and (not field.zonal_periodic):
        if x < field.lonlat_minmax[0] or x > field.lonlat_minmax[1]:
            _raise_field_out_of_bound_error(z, y, x)
    if field.ydim > 1 and (y < field.lonlat_minmax[2] or y > field.lonlat_minmax[3]):
        _raise_field_out_of_bound_error(z, y, x)

    if field.xdim > 1:
        if field._mesh_type != "spherical":
            lon_index = field.lon < x
            if lon_index.all():
                xi = len(field.lon) - 2
            else:
                xi = lon_index.argmin() - 1 if lon_index.any() else 0
            xsi = (x - field.lon[xi]) / (field.lon[xi + 1] - field.lon[xi])
            if xsi < 0:
                xi -= 1
                xsi = (x - field.lon[xi]) / (field.lon[xi + 1] - field.lon[xi])
            elif xsi > 1:
                xi += 1
                xsi = (x - field.lon[xi]) / (field.lon[xi + 1] - field.lon[xi])
        else:
            lon_fixed = field.lon.copy()
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

    if field.ydim > 1:
        lat_index = field.lat < y
        if lat_index.all():
            yi = len(field.lat) - 2
        else:
            yi = lat_index.argmin() - 1 if lat_index.any() else 0

        eta = (y - field.lat[yi]) / (field.lat[yi + 1] - field.lat[yi])
        if eta < 0:
            yi -= 1
            eta = (y - field.lat[yi]) / (field.lat[yi + 1] - field.lat[yi])
        elif eta > 1:
            yi += 1
            eta = (y - field.lat[yi]) / (field.lat[yi + 1] - field.lat[yi])
    else:
        yi, eta = -1, 0

    if field.zdim > 1 and not search2D:
        if field._gtype == GridType.RectilinearZGrid:
            try:
                (zi, zeta) = search_indices_vertical_z(field.grid, field.gridindexingtype, z)
            except FieldOutOfBoundError:
                _raise_field_out_of_bound_error(z, y, x)
            except FieldOutOfBoundSurfaceError:
                _raise_field_out_of_bound_surface_error(z, y, x)
        elif field._gtype == GridType.RectilinearSGrid:
            ## TODO :  Still need to implement the search_indices_vertical_s function
            (zi, zeta) = search_indices_vertical_s(field.grid, field.interp_method, time, z, y, x, ti, yi, xi, eta, xsi)
    else:
        zi, zeta = -1, 0

    if not ((0 <= xsi <= 1) and (0 <= eta <= 1) and (0 <= zeta <= 1)):
        _raise_field_sampling_error(z, y, x)

    _ei = field.ravel_index(zi, yi, xi)

    return (zeta, eta, xsi, _ei)


def _search_indices_curvilinear_2d(
    grid: XGrid, y: float, x: float, yi_guess: int | None = None, xi_guess: int | None = None
):  # TODO fix typing instructions to make clear that y, x etc need to be ndarrays
    yi, xi = yi_guess, xi_guess
    if yi is None or xi is None:
        faces = grid.get_spatial_hash().query(np.column_stack((y, x)))
        yi, xi = faces[0]

    xsi = eta = -1.0 * np.ones(len(x), dtype=float)
    invA = np.array(
        [
            [1, 0, 0, 0],
            [-1, 1, 0, 0],
            [-1, 0, 0, 1],
            [1, -1, 1, -1],
        ]
    )
    maxIterSearch = 1e6
    it = 0
    tol = 1.0e-10

    # # ! Error handling for out of bounds
    # TODO: Re-enable in some capacity
    # if x < field.lonlat_minmax[0] or x > field.lonlat_minmax[1]:
    #     if grid.lon[0, 0] < grid.lon[0, -1]:
    #         _raise_field_out_of_bound_error(y, x)
    #     elif x < grid.lon[0, 0] and x > grid.lon[0, -1]:  # This prevents from crashing in [160, -160]
    #         _raise_field_out_of_bound_error(z, y, x)

    # if y < field.lonlat_minmax[2] or y > field.lonlat_minmax[3]:
    #     _raise_field_out_of_bound_error(z, y, x)

    while np.any(xsi < -tol) or np.any(xsi > 1 + tol) or np.any(eta < -tol) or np.any(eta > 1 + tol):
        px = np.array([grid.lon[yi, xi], grid.lon[yi, xi + 1], grid.lon[yi + 1, xi + 1], grid.lon[yi + 1, xi]])

        py = np.array([grid.lat[yi, xi], grid.lat[yi, xi + 1], grid.lat[yi + 1, xi + 1], grid.lat[yi + 1, xi]])
        a = np.dot(invA, px)
        b = np.dot(invA, py)

        aa = a[3] * b[2] - a[2] * b[3]
        bb = a[3] * b[0] - a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + x * b[3] - y * a[3]
        cc = a[1] * b[0] - a[0] * b[1] + x * b[1] - y * a[1]

        det2 = bb * bb - 4 * aa * cc
        det = np.where(det2 > 0, np.sqrt(det2), eta)
        eta = np.where(abs(aa) < 1e-12, -cc / bb, np.where(det2 > 0, (-bb + det) / (2 * aa), eta))

        xsi = np.where(
            abs(a[1] + a[3] * eta) < 1e-12,
            ((y - py[0]) / (py[1] - py[0]) + (y - py[3]) / (py[2] - py[3])) * 0.5,
            (x - a[0] - a[2] * eta) / (a[1] + a[3] * eta),
        )

        xi = np.where(xsi < -tol, xi - 1, np.where(xsi > 1 + tol, xi + 1, xi))
        yi = np.where(eta < -tol, yi - 1, np.where(eta > 1 + tol, yi + 1, yi))

        (yi, xi) = _reconnect_bnd_indices(yi, xi, grid.ydim, grid.xdim, grid.mesh)
        it += 1
        if it > maxIterSearch:
            print(f"Correct cell not found after {maxIterSearch} iterations")
            _raise_field_out_of_bound_error(0, y, x)
    xsi = np.where(xsi < 0.0, 0.0, np.where(xsi > 1.0, 1.0, xsi))
    eta = np.where(eta < 0.0, 0.0, np.where(eta > 1.0, 1.0, eta))

    if np.any((xsi < 0) | (xsi > 1) | (eta < 0) | (eta > 1)):
        _raise_field_sampling_error(y, x)
    return (yi, eta, xi, xsi)


## TODO :  Still need to implement the search_indices_curvilinear
def _search_indices_curvilinear(field, time, z, y, x, ti, particle=None, search2D=False):
    if particle:
        zi, yi, xi = field.unravel_index(particle.ei)
    else:
        xi = int(field.xdim / 2) - 1
        yi = int(field.ydim / 2) - 1
    xsi = eta = -1.0
    grid = field.grid
    invA = np.array([[1, 0, 0, 0], [-1, 1, 0, 0], [-1, 0, 0, 1], [1, -1, 1, -1]])
    maxIterSearch = 1e6
    it = 0
    tol = 1.0e-10
    if not grid.zonal_periodic:
        if x < field.lonlat_minmax[0] or x > field.lonlat_minmax[1]:
            if grid.lon[0, 0] < grid.lon[0, -1]:
                _raise_field_out_of_bound_error(z, y, x)
            elif x < grid.lon[0, 0] and x > grid.lon[0, -1]:  # This prevents from crashing in [160, -160]
                _raise_field_out_of_bound_error(z, y, x)
    if y < field.lonlat_minmax[2] or y > field.lonlat_minmax[3]:
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
        particle.ei[field.igrid] = field.ravel_index(zi, yi, xi)

    return (zeta, eta, xsi, zi, yi, xi)


def _reconnect_bnd_indices(yi: int, xi: int, ydim: int, xdim: int, sphere_mesh: bool):
    xi = np.where(xi < 0, (xdim - 2) if sphere_mesh else 0, xi)
    xi = np.where(xi > xdim - 2, 0 if sphere_mesh else (xdim - 2), xi)

    xi = np.where(yi > ydim - 2, xdim - xi if sphere_mesh else xi, xi)

    yi = np.where(yi < 0, 0, yi)
    yi = np.where(yi > ydim - 2, ydim - 2, yi)

    return yi, xi

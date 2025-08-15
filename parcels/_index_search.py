from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from parcels.tools.statuscodes import (
    _raise_field_out_of_bound_error,
    _raise_field_sampling_error,
    _raise_time_extrapolation_error,
)

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
        return 0, 0

    if time not in field.time_interval:
        _raise_time_extrapolation_error(time, field=None)

    time_index = field.data.time <= time

    if time_index.all():
        # If given time > last known field time, use
        # the last field frame without interpolation
        ti = len(field.data.time) - 1

    elif np.logical_not(time_index).all():
        # If given time < any time in the field, use
        # the first field frame without interpolation
        ti = 0
    else:
        ti = int(time_index.argmin() - 1) if time_index.any() else 0
    if len(field.data.time) == 1:
        tau = 0
    elif ti == len(field.data.time) - 1:
        tau = 1
    else:
        tau = (
            (time - field.data.time[ti]).dt.total_seconds().values
            / (field.data.time[ti + 1] - field.data.time[ti]).dt.total_seconds().values
            if field.data.time[ti] != field.data.time[ti + 1]
            else 0
        )
    return tau, ti


def _search_indices_curvilinear_2d(
    grid: XGrid, y: float, x: float, yi_guess: int | None = None, xi_guess: int | None = None
):
    yi, xi = yi_guess, xi_guess
    if yi is None or xi is None:
        faces = grid.get_spatial_hash().query(np.column_stack((y, x)))
        yi, xi = faces[0]

    xsi = eta = -1.0
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

    while xsi < -tol or xsi > 1 + tol or eta < -tol or eta > 1 + tol:
        px = np.array([grid.lon[yi, xi], grid.lon[yi, xi + 1], grid.lon[yi + 1, xi + 1], grid.lon[yi + 1, xi]])

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

    if not ((0 <= xsi <= 1) and (0 <= eta <= 1)):
        _raise_field_sampling_error(y, x)
    return (yi, eta, xi, xsi)


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

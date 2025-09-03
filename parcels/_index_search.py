from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from parcels._typing import Mesh
from parcels.tools.statuscodes import (
    _raise_grid_searching_error,
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
        return np.zeros(shape=time.shape, dtype=np.float32), np.zeros(shape=time.shape, dtype=np.int32)

    if not field.time_interval.is_all_time_in_interval(time):
        _raise_time_extrapolation_error(time, field=None)

    ti = np.searchsorted(field.data.time.data, time, side="right") - 1
    tau = (time - field.data.time.data[ti]) / (field.data.time.data[ti + 1] - field.data.time.data[ti])
    return np.atleast_1d(tau), np.atleast_1d(ti)


def _search_indices_curvilinear_2d(
    grid: XGrid, y: float, x: float, yi_guess: int | None = None, xi_guess: int | None = None
):  # TODO fix typing instructions to make clear that y, x etc need to be ndarrays
    yi, xi = yi_guess, xi_guess
    if yi is None or xi is None:
        yi, xi = grid.get_spatial_hash().query(y, x)

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
    #         _raise_grid_searching_error(y, x)
    #     elif x < grid.lon[0, 0] and x > grid.lon[0, -1]:  # This prevents from crashing in [160, -160]
    #         _raise_grid_searching_error(z, y, x)

    # if y < field.lonlat_minmax[2] or y > field.lonlat_minmax[3]:
    #     _raise_grid_searching_error(z, y, x)

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

        (yi, xi) = _reconnect_bnd_indices(yi, xi, grid.ydim, grid.xdim, grid._mesh)
        it += 1
        if it > maxIterSearch:
            print(f"Correct cell not found after {maxIterSearch} iterations")
            _raise_grid_searching_error(0, y, x)
    xsi = np.where(xsi < 0.0, 0.0, np.where(xsi > 1.0, 1.0, xsi))
    eta = np.where(eta < 0.0, 0.0, np.where(eta > 1.0, 1.0, eta))

    if np.any((xsi < 0) | (xsi > 1) | (eta < 0) | (eta > 1)):
        _raise_grid_searching_error(y, x)
    return (yi, eta, xi, xsi)


def _reconnect_bnd_indices(yi: int, xi: int, ydim: int, xdim: int, mesh: Mesh):
    xi = np.where(xi < 0, (xdim - 2) if mesh == "spherical" else 0, xi)
    xi = np.where(xi > xdim - 2, 0 if mesh == "spherical" else (xdim - 2), xi)

    xi = np.where(yi > ydim - 2, xdim - xi if mesh == "spherical" else xi, xi)

    yi = np.where(yi < 0, 0, yi)
    yi = np.where(yi > ydim - 2, ydim - 2, yi)

    return yi, xi

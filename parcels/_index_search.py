from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from parcels.tools.statuscodes import _raise_time_extrapolation_error

if TYPE_CHECKING:
    from parcels.xgrid import XGrid

    from .field import Field


GRID_SEARCH_ERROR = -3


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


def curvilinear_point_in_cell(grid, y: np.ndarray, x: np.ndarray, yi: np.ndarray, xi: np.ndarray):
    xsi = eta = -1.0 * np.ones(len(x), dtype=float)
    invA = np.array(
        [
            [1, 0, 0, 0],
            [-1, 1, 0, 0],
            [-1, 0, 0, 1],
            [1, -1, 1, -1],
        ]
    )

    px = np.array([grid.lon[yi, xi], grid.lon[yi, xi + 1], grid.lon[yi + 1, xi + 1], grid.lon[yi + 1, xi]])
    py = np.array([grid.lat[yi, xi], grid.lat[yi, xi + 1], grid.lat[yi + 1, xi + 1], grid.lat[yi + 1, xi]])

    a, b = np.dot(invA, px), np.dot(invA, py)
    aa = a[3] * b[2] - a[2] * b[3]
    bb = a[3] * b[0] - a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + x * b[3] - y * a[3]
    cc = a[1] * b[0] - a[0] * b[1] + x * b[1] - y * a[1]
    det2 = bb * bb - 4 * aa * cc

    with np.errstate(divide="ignore", invalid="ignore"):
        det = np.where(det2 > 0, np.sqrt(det2), eta)
        eta = np.where(abs(aa) < 1e-12, -cc / bb, np.where(det2 > 0, (-bb + det) / (2 * aa), eta))

        xsi = np.where(
            abs(a[1] + a[3] * eta) < 1e-12,
            ((y - py[0]) / (py[1] - py[0]) + (y - py[3]) / (py[2] - py[3])) * 0.5,
            (x - a[0] - a[2] * eta) / (a[1] + a[3] * eta),
        )

    is_in_cell = np.where((xsi >= 0) & (xsi <= 1) & (eta >= 0) & (eta <= 1), 1, 0)

    return is_in_cell, np.column_stack((xsi, eta))


def _search_indices_curvilinear_2d(
    grid: XGrid, y: np.ndarray, x: np.ndarray, yi: np.ndarray | None = None, xi: np.ndarray | None = None
):
    """Searches a grid for particle locations in 2D curvilinear coordinates.

    Parameters
    ----------
    grid : XGrid
        The curvilinear grid to search within.
    y : np.ndarray
        Array of latitude-coordinates of the points to locate.
    x : np.ndarray
        Array of longitude-coordinates of the points to locate.
    yi : np.ndarray | None, optional
        Array of initial guesses for the j indices of the points to locate.
    xi : np.ndarray | None, optional
        Array of initial guesses for the i indices of the points to locate.

    Returns
    -------
    tuple
        A tuple containing four elements:
        - yi (np.ndarray): Array of found j-indices corresponding to the input coordinates.
        - eta (np.ndarray): Array of barycentric coordinates in the j-direction within the found grid cells.
        - xi (np.ndarray): Array of found i-indices corresponding to the input cooordinates.
        - xsi (np.ndarray): Array of barycentric coordinates in the i-direction within the found grid cells.
    """
    if np.any(xi):
        # If an initial guess is provided, we first perform a point in cell check for all guessed indices
        is_in_cell, coords = curvilinear_point_in_cell(grid, y, x, yi, xi)
        y_check = y[is_in_cell == 0]
        x_check = x[is_in_cell == 0]
        zero_indices = np.where(is_in_cell == 0)[0]
    else:
        # Otherwise, we need to check all points
        yi = np.full(len(y), GRID_SEARCH_ERROR, dtype=np.int32)
        xi = np.full(len(x), GRID_SEARCH_ERROR, dtype=np.int32)
        y_check = y
        x_check = x
        coords = -1.0 * np.ones((len(y), 2), dtype=np.float32)
        zero_indices = np.arange(len(y))

    # If there are any points that were not found in the first step, we query the spatial hash for those points
    if len(zero_indices) > 0:
        yi_q, xi_q, coords_q = grid.get_spatial_hash().query(y_check, x_check)
        # Only those points that were not found in the first step are updated
        coords[zero_indices, :] = coords_q
        yi[zero_indices] = yi_q
        xi[zero_indices] = xi_q

    xsi = coords[:, 0]
    eta = coords[:, 1]

    return (yi, eta, xi, xsi)

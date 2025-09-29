from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from parcels._core.statuscodes import _raise_time_extrapolation_error

if TYPE_CHECKING:
    from parcels._core.field import Field
    from parcels.xgrid import XGrid


GRID_SEARCH_ERROR = -3
LEFT_OUT_OF_BOUNDS = -2
RIGHT_OUT_OF_BOUNDS = -1


def _search_1d_array(
    arr: np.array,
    x: float,
) -> tuple[int, int]:
    """
    Searches for particle locations in a 1D array and returns barycentric coordinate along dimension.

    Assumptions:
    - array is strictly monotonically increasing.

    Parameters
    ----------
    arr : np.array
        1D array to search in.
    x : float
        Position in the 1D array to search for.

    Returns
    -------
    array of int
        Index of the element just before the position x in the array. Note that this index is -2 if the index is left out of bounds and -1 if the index is right out of bounds.
    array of float
        Barycentric coordinate.
    """
    # TODO v4: We probably rework this to deal with 0D arrays before this point (as we already know field dimensionality)
    if len(arr) < 2:
        return np.zeros(shape=x.shape, dtype=np.int32), np.zeros_like(x)
    index = np.searchsorted(arr, x, side="right") - 1
    # Use broadcasting to avoid repeated array access
    arr_index = arr[index]
    arr_next = arr[np.clip(index + 1, 1, len(arr) - 1)]  # Ensure we don't go out of bounds
    bcoord = (x - arr_index) / (arr_next - arr_index)

    # TODO check how we can avoid searchsorted when grid spacing is uniform
    # dx = arr[1] - arr[0]
    # index = ((x - arr[0]) / dx).astype(int)
    # index = np.clip(index, 0, len(arr) - 2)
    # bcoord = (x - arr[index]) / dx

    index = np.where(x < arr[0], LEFT_OUT_OF_BOUNDS, index)
    index = np.where(x >= arr[-1], RIGHT_OUT_OF_BOUNDS, index)

    return np.atleast_1d(index), np.atleast_1d(bcoord)


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


def uxgrid_point_in_cell(grid, y: np.ndarray, x: np.ndarray, yi: np.ndarray, xi: np.ndarray):
    """Check if points are inside the grid cells defined by the given face indices.

    Parameters
    ----------
    grid : ux.grid.Grid
        The uxarray grid object containing the unstructured grid data.
    y : np.ndarray
        Array of latitudes of the points to check.
    x : np.ndarray
        Array of longitudes of the points to check.
    yi : np.ndarray
        Array of face indices corresponding to the points.
    xi : np.ndarray
        Not used, but included for compatibility with other search functions.

    Returns
    -------
    is_in_cell : np.ndarray
        An array indicating whether each point is inside (1) or outside (0) the corresponding cell.
    coords : np.ndarray
        Barycentric coordinates of the points within their respective cells.
    """
    if grid._mesh == "spherical":
        lon_rad = np.deg2rad(x)
        lat_rad = np.deg2rad(y)
        x_cart, y_cart, z_cart = _latlon_rad_to_xyz(lat_rad, lon_rad)
        points = np.column_stack((x_cart.flatten(), y_cart.flatten(), z_cart.flatten()))

        # Get the vertex indices for each face
        nids = grid.uxgrid.face_node_connectivity[yi].values
        face_vertices = np.stack(
            (
                grid.uxgrid.node_x[nids.ravel()].values.reshape(nids.shape),
                grid.uxgrid.node_y[nids.ravel()].values.reshape(nids.shape),
                grid.uxgrid.node_z[nids.ravel()].values.reshape(nids.shape),
            ),
            axis=-1,
        )
    else:
        nids = grid.uxgrid.face_node_connectivity[yi].values
        face_vertices = np.stack(
            (
                grid.uxgrid.node_lon[nids.ravel()].values.reshape(nids.shape),
                grid.uxgrid.node_lat[nids.ravel()].values.reshape(nids.shape),
            ),
            axis=-1,
        )
        points = np.stack((x, y), axis=-1)

    M = len(points)

    is_in_cell = np.zeros(M, dtype=np.int32)

    coords = _barycentric_coordinates(face_vertices, points)
    is_in_cell = np.where(np.all((coords >= -1e-6) & (coords <= 1 + 1e-6), axis=1), 1, 0)

    return is_in_cell, coords


def _triangle_area(A, B, C):
    """Compute the area of a triangle given by three points."""
    d1 = B - A
    d2 = C - A
    if A.shape[-1] == 2:
        # 2D case: cross product reduces to scalar z-component
        cross = d1[..., 0] * d2[..., 1] - d1[..., 1] * d2[..., 0]
        area = 0.5 * np.abs(cross)
    elif A.shape[-1] == 3:
        # 3D case: full vector cross product
        cross = np.cross(d1, d2)
        area = 0.5 * np.linalg.norm(cross, axis=-1)
    else:
        raise ValueError(f"Expected last dim=2 or 3, got {A.shape[-1]}")

    return area


def _barycentric_coordinates(nodes, points, min_area=1e-8):
    """
    Compute the barycentric coordinates of a point P inside a convex polygon using area-based weights.
    So that this method generalizes to n-sided polygons, we use the Waschpress points as the generalized
    barycentric coordinates, which is only valid for convex polygons.

    Parameters
    ----------
        nodes : numpy.ndarray
            Polygon verties per query of shape (M, 3, 2/3) where M is the number of query points. The second dimension corresponds to the number
            of vertices
            The last dimension can be either 2 or 3, where 3 corresponds to the (z, y, x) coordinates of each vertex and 2 corresponds to the
            (lat, lon) coordinates of each vertex.

        points : numpy.ndarray
            Spherical coordinates of the point (M,2/3) where M is the number of query points.

    Returns
    -------
    numpy.ndarray
        Barycentric coordinates corresponding to each vertex.

    """
    M, K = nodes.shape[:2]

    # roll(-1) to get vi+1, roll(+1) to get vi-1
    vi = nodes  # (M,K,2)
    vi1 = np.roll(nodes, shift=-1, axis=1)  # (M,K,2)
    vim1 = np.roll(nodes, shift=+1, axis=1)  # (M,K,2)

    # a0 = area(v_{i-1}, v_i, v_{i+1})
    a0 = _triangle_area(vim1, vi, vi1)  # (M,K)

    # a1 = area(P, v_{i-1}, v_i); a2 = area(P, v_i, v_{i+1})
    P = points[:, None, :]  # (M,1,2) -> (M,K,2)
    a1 = _triangle_area(P, vim1, vi)
    a2 = _triangle_area(P, vi, vi1)

    # clamp tiny denominators for stability
    a1c = np.maximum(a1, min_area)
    a2c = np.maximum(a2, min_area)

    wi = a0 / (a1c * a2c)  # (M,K)

    sum_wi = wi.sum(axis=1, keepdims=True)  # (M,1)
    # Avoid 0/0: if sum_wi==0 (degenerate), keep zeros
    with np.errstate(invalid="ignore", divide="ignore"):
        bcoords = wi / sum_wi

    return bcoords


def _latlon_rad_to_xyz(
    lat,
    lon,
):
    """Converts Spherical latitude and longitude coordinates into Cartesian x,
    y, z coordinates.
    """
    x = np.cos(lon) * np.cos(lat)
    y = np.sin(lon) * np.cos(lat)
    z = np.sin(lat)

    return x, y, z

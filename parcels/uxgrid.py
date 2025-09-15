from __future__ import annotations

from typing import Literal

import numpy as np
import uxarray as ux

from parcels._index_search import GRID_SEARCH_ERROR
from parcels._typing import assert_valid_mesh
from parcels.xgrid import _search_1d_array

from .basegrid import BaseGrid

_UXGRID_AXES = Literal["Z", "FACE"]


class UxGrid(BaseGrid):
    """
    Extension of uxarray's Grid class that supports point-location search
    for interpolation on unstructured grids.
    """

    def __init__(self, grid: ux.grid.Grid, z: ux.UxDataArray, mesh="flat") -> UxGrid:
        """
        Initializes the UxGrid with a uxarray grid and vertical coordinate array.

        Parameters
        ----------
        grid : ux.grid.Grid
            The uxarray grid object containing the unstructured grid data.
        z : ux.UxDataArray
            A 1D array of vertical coordinates (depths) associated with the layer interface heights (not the mid-layer depths).
            While uxarray allows nz to be spatially and temporally varying, the parcels.UxGrid class considers the case where
            the vertical coordinate is constant in time and space. This implies flat bottom topography and no moving ALE vertical grid.
        mesh : str, optional
            The type of mesh used for the grid. Either "flat" (default) or "spherical".
        """
        self.uxgrid = grid
        if not isinstance(z, ux.UxDataArray):
            raise TypeError("z must be an instance of ux.UxDataArray")
        if z.ndim != 1:
            raise ValueError("z must be a 1D array of vertical coordinates")
        self.z = z
        self._mesh = mesh
        self._spatialhash = None

        assert_valid_mesh(mesh)

    @property
    def depth(self):
        """
        Note
        ----
        Included for compatibility with v3 codebase. May be removed in future.
        TODO v4: Evaluate
        """
        try:
            _ = self.z.values
        except KeyError:
            return np.zeros(1)
        return self.z.values

    @property
    def axes(self) -> list[_UXGRID_AXES]:
        return ["Z", "FACE"]

    def get_axis_dim(self, axis: _UXGRID_AXES) -> int:
        if axis not in self.axes:
            raise ValueError(f"Axis {axis!r} is not part of this grid. Available axes: {self.axes}")

        if axis == "Z":
            return len(self.z.values)
        elif axis == "FACE":
            return self.uxgrid.n_face

    def search(self, z, y, x, ei=None, tol=1e-6):
        """
        Search for the grid cell (face) and vertical layer that contains the given points.

        Parameters
        ----------
        z : float or np.ndarray
            The vertical coordinate(s) (depth) of the point(s).
        y : float or np.ndarray
            The latitude(s) of the point(s).
        x : float or np.ndarray
            The longitude(s) of the point(s).
        ei : np.ndarray, optional
            Precomputed horizontal indices (face indices) for the points.

            TO BE IMPLEMENTED : If provided, we'll check
            if the points are within the faces specified by these indices. For cells where the particles
            are not found, a nearest neighbor search will be performed. As a last resort, the spatial hash will be used.
        tol : float, optional
            Tolerance for barycentric coordinate checks. Default is 1e-6.
        """
        indices = self.unravel_index(ei)
        fi = indices["FACE"]
        zi = indices["Z"]
        zi, zeta = _search_1d_array(self.z.values, z)
        if np.any(ei):
            is_in_cell, coords = uxgrid_point_in_cell(self.uxgrid, y, x, fi, fi)
            y_check = y[is_in_cell == 0]
            x_check = x[is_in_cell == 0]
            zero_indices = np.where(is_in_cell == 0)[0]
        else:
            # Otherwise, we need to check all points
            fi = np.full(len(y), GRID_SEARCH_ERROR, dtype=np.int32)
            y_check = y
            x_check = x
            coords = -1.0 * np.ones((len(y), 2), dtype=np.float32)
            zero_indices = np.arange(len(y))

        if len(zero_indices) > 0:
            face_ids_q, _, coords_q = self.uxgrid.get_spatial_hash().query(y_check, x_check)
            coords[zero_indices, :] = coords_q
            fi[zero_indices] = face_ids_q

        return {"Z": (zi, zeta), "FACE": (fi, coords)}


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
    if grid.mesh == "spherical":
        lon_rad = np.deg2rad(grid.lon.values)
        lat_rad = np.deg2rad(grid.lat.values)
        x_cart, y_cart, z_cart = _lonlat_rad_to_xyz(lon_rad, lat_rad)
        points = np.column_stack((x_cart.flatten(), y_cart.flatten(), z_cart.flatten()))

        # Get the vertex indices for each face
        nodeids = grid.face_node_connectivity[yi, :].values
        face_vertices = np.column_stack(
            (grid.node_x[nodeids].values, grid.node_y[nodeids].values, grid.node_z[nodeids].values)
        )
    else:
        nodeids = grid.face_node_connectivity[yi, :].values
        face_vertices = np.column_stack(
            (grid.node_lon[nodeids].values.flatten(), grid.node_lat[nodeids].values.flatten())
        )
        points = np.column_stack((x, y))

    M = len(points)

    is_in_cell = np.zeros(M, dtype=np.int32)

    coords = _barycentric_coordinates(face_vertices, points)
    is_in_cell = np.where(np.all((coords >= -1e-6) & (coords <= 1 + 1e-6), axis=1), 1, 0)

    return is_in_cell, coords


def _triangle_area(A, B, C):
    """Compute the area of a triangle given by three points."""
    d1 = B - A
    d2 = C - A
    d3 = np.cross(d1, d2)
    return 0.5 * np.linalg.norm(d3)


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


def _lonlat_rad_to_xyz(
    lon,
    lat,
):
    """Converts Spherical latitude and longitude coordinates into Cartesian x,
    y, z coordinates.
    """
    x = np.cos(lon) * np.cos(lat)
    y = np.sin(lon) * np.cos(lat)
    z = np.sin(lat)

    return x, y, z

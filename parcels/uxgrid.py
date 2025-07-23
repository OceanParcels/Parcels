from __future__ import annotations

from typing import Literal

import numpy as np
import uxarray as ux
from uxarray.grid.coordinates import _lonlat_rad_to_xyz
from uxarray.grid.neighbors import _barycentric_coordinates

from parcels.field import FieldOutOfBoundError  # Adjust import as necessary
from parcels.xgrid import _search_1d_array

from .basegrid import BaseGrid

_UXGRID_AXES = Literal["Z", "FACE"]


class UxGrid(BaseGrid):
    """
    Extension of uxarray's Grid class that supports point-location search
    for interpolation on unstructured grids.
    """

    def __init__(self, grid: ux.grid.Grid, z: ux.UxDataArray) -> UxGrid:
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
        """
        self.uxgrid = grid
        if not isinstance(z, ux.UxDataArray):
            raise TypeError("z must be an instance of ux.UxDataArray")
        if z.ndim != 1:
            raise ValueError("z must be a 1D array of vertical coordinates")
        self.z = z

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
        def try_face(fid):
            bcoords, err = self._get_barycentric_coordinates_latlon(y, x, fid)
            if (bcoords >= 0).all() and (bcoords <= 1).all() and err < tol:
                return bcoords
            else:
                bcoords, err = self._get_barycentric_coordinates_cartesian(y, x, fid)
                if (bcoords >= 0).all() and (bcoords <= 1).all() and err < tol:
                    return bcoords

            return None

        zi, zeta = _search_1d_array(self.z.values, z)

        if ei is not None:
            _, fi = self.unravel_index(ei)
            bcoords = try_face(fi)
            if bcoords is not None:
                return bcoords, self.ravel_index(zi, fi)
            # Try neighbors of current face
            for neighbor in self.uxgrid.face_face_connectivity[fi, :]:
                if neighbor == -1:
                    continue
                bcoords = try_face(neighbor)
                if bcoords is not None:
                    return bcoords, self.ravel_index(zi, neighbor)

        # Global fallback as last ditch effort
        face_ids = self.uxgrid.get_faces_containing_point([x, y], return_counts=False)[0]
        fi = face_ids[0] if len(face_ids) > 0 else -1
        if fi == -1:
            raise FieldOutOfBoundError(z, y, x)
        bcoords = try_face(fi)
        if bcoords is None:
            raise FieldOutOfBoundError(z, y, x)

        return {"Z": (zi, zeta), "FACE": (fi, bcoords)}

    def _get_barycentric_coordinates_latlon(self, y, x, fi):
        """Checks if a point is inside a given face id on a UxGrid."""
        # Check if particle is in the same face, otherwise search again.

        n_nodes = self.uxgrid.n_nodes_per_face[fi].to_numpy()
        node_ids = self.uxgrid.face_node_connectivity[fi, 0:n_nodes]
        nodes = np.column_stack(
            (
                np.deg2rad(self.uxgrid.node_lon[node_ids].to_numpy()),
                np.deg2rad(self.uxgrid.node_lat[node_ids].to_numpy()),
            )
        )

        coord = np.deg2rad([x, y])
        bcoord = np.asarray(_barycentric_coordinates(nodes, coord))
        err = abs(np.dot(bcoord, nodes[:, 0]) - coord[0]) + abs(np.dot(bcoord, nodes[:, 1]) - coord[1])
        return bcoord, err

    def _get_barycentric_coordinates_cartesian(self, y, x, fi):
        n_nodes = self.uxgrid.n_nodes_per_face[fi].to_numpy()
        node_ids = self.uxgrid.face_node_connectivity[fi, 0:n_nodes]

        coord = np.deg2rad([x, y])
        x, y, z = _lonlat_rad_to_xyz(coord[0], coord[1])
        cart_coord = np.array([x, y, z]).T
        # Second attempt to find barycentric coordinates using cartesian coordinates
        nodes = np.stack(
            (
                self._source_grid.node_x[node_ids].values(),
                self._source_grid.node_y[node_ids].values(),
                self._source_grid.node_z[node_ids].values(),
            ),
            axis=-1,
        )

        bcoord = np.asarray(_barycentric_coordinates_cartesian(nodes, cart_coord))
        proj_uv = np.dot(bcoord, nodes)
        err = np.linalg.norm(proj_uv - coord)
        face_center = np.stack(
            (
                self._source_grid.face_x[fi].values(),
                self._source_grid.face_y[fi].values(),
                self._source_grid.face_z[fi].values(),
            ),
            axis=-1,
        )
        # Compute and remove the local projection error
        err -= np.abs(_local_projection_error(nodes, face_center))
        return bcoord, err


def _barycentric_coordinates_cartesian(nodes, point, min_area=1e-8):
    """
    Compute the barycentric coordinates of a point P inside a convex polygon using area-based weights.
    So that this method generalizes to n-sided polygons, we use the Waschpress points as the generalized
    barycentric coordinates, which is only valid for convex polygons.

    Parameters
    ----------
        nodes : numpy.ndarray
            Cartesian coordinates (x,y,z) of each corner node of a face
        point : numpy.ndarray
            Cartesian coordinates (x,y,z) of the point

    Returns
    -------
    numpy.ndarray
        Barycentric coordinates corresponding to each vertex.

    """
    n = len(nodes)
    sum_wi = 0
    w = []

    for i in range(0, n):
        vim1 = nodes[i - 1]
        vi = nodes[i]
        vi1 = nodes[(i + 1) % n]
        a0 = _triangle_area_cartesian(vim1, vi, vi1)
        a1 = max(_triangle_area_cartesian(point, vim1, vi), min_area)
        a2 = max(_triangle_area_cartesian(point, vi, vi1), min_area)
        sum_wi += a0 / (a1 * a2)
        w.append(a0 / (a1 * a2))

    barycentric_coords = [w_i / sum_wi for w_i in w]

    return barycentric_coords


def _local_projection_error(nodes, point):
    """
    Computes the size of the local projection error that arises from
    assuming planar faces. Effectively, a planar face on a spherical
    manifold is local linearization of the spherical coordinate
    transformation. Since query points and nodes are converted to
    cartesian coordinates using the full spherical coordinate transformation,
    the local projection error will likely be non-zero but related to the discretiztaion.
    """
    a = nodes[1] - nodes[0]
    b = nodes[2] - nodes[0]
    normal = np.cross(a, b)
    normal /= np.linalg.norm(normal)
    d = point - nodes[0]
    return abs(np.dot(d, normal))


def _triangle_area_cartesian(A, B, C):
    """Compute the area of a triangle given by three points."""
    d1 = B - A
    d2 = C - A
    d3 = np.cross(d1, d2)
    return 0.5 * np.linalg.norm(d3)

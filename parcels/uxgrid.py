from __future__ import annotations

from typing import Literal

import numpy as np
import uxarray as ux

from parcels.spatialhash import _barycentric_coordinates
from parcels.tools.statuscodes import FieldOutOfBoundError
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
                bcoords = self._get_barycentric_coordinates_cartesian(y, x, fid)
                if (bcoords >= 0).all() and (bcoords <= 1).all():
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
        points = np.column_stack((x, y))
        face_ids = self.uxgrid.get_faces_containing_point(points, return_counts=False)[0]
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

        coord = np.deg2rad(np.column_stack((x, y)))
        bcoord = np.asarray(_barycentric_coordinates(nodes, coord))
        proj_coord = np.matmul(np.transpose(nodes), bcoord)
        err = np.linalg.norm(proj_coord - coord)
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
                self.uxgrid.node_x[node_ids].values,
                self.uxgrid.node_y[node_ids].values,
                self.uxgrid.node_z[node_ids].values,
            ),
            axis=-1,
        )

        bcoord = np.asarray(_barycentric_coordinates(nodes, cart_coord))

        return bcoord


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

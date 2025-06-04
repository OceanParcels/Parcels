from __future__ import annotations

import numpy as np
import uxarray as ux
from uxarray.grid.neighbors import _barycentric_coordinates

from parcels.field import FieldOutOfBoundError  # Adjust import as necessary

from .basegrid import BaseGrid


class UxGrid(BaseGrid):
    """
    Extension of uxarray's Grid class that supports point-location search
    for interpolation on unstructured grids.
    """

    def __init__(self, grid: ux.grid.Grid) -> UxGrid:
        self.uxgrid = grid

    def search(
        self, field, z: float, y: float, x: float, ei: int | None = None, search2D: bool = False
    ) -> tuple[np.ndarray, int]:
        tol = 1e-10

        def try_face(fid):
            bcoords, err = self.uxgrid._get_barycentric_coordinates(y, x, fid)
            if (bcoords >= 0).all() and (bcoords <= 1).all() and err < tol:
                return bcoords, field.ravel_index(0, 0, fid)  # Z and time indices are 0 for now
            return None, None

        if ei is not None:
            zi, fi = field.unravel_index(ei)
            bcoords, ei_new = try_face(fi)
            if bcoords is not None:
                return bcoords, ei_new

            # Try neighbors of current face
            for neighbor in self.uxgrid.face_face_connectivity[fi, :]:
                if neighbor == -1:
                    continue
                bcoords, ei_new = try_face(neighbor)
                if bcoords is not None:
                    return bcoords, ei_new

        # Global fallback using spatial hash
        fi, bcoords = self.uxgrid.get_spatial_hash().query([[x, y]])
        if fi == -1:
            raise FieldOutOfBoundError(z, y, x)

        return bcoords, field.ravel_index(0, 0, fi)

    def _get_barycentric_coordinates(self, y, x, fi):
        """Checks if a point is inside a given face id on a UxGrid."""
        # Check if particle is in the same face, otherwise search again.
        n_nodes = self.uxgrid.n_nodes_per_face[fi].to_numpy()
        node_ids = self.uxgrid.face_node_connectivity[fi, 0:n_nodes]
        nodes = np.column_stack(
            (
                np.deg2rad(self.uxgrid.grid.node_lon[node_ids].to_numpy()),
                np.deg2rad(self.uxgrid.grid.node_lat[node_ids].to_numpy()),
            )
        )

        coord = np.deg2rad([x, y])
        bcoord = np.asarray(_barycentric_coordinates(nodes, coord))
        err = abs(np.dot(bcoord, nodes[:, 0]) - coord[0]) + abs(np.dot(bcoord, nodes[:, 1]) - coord[1])
        return bcoord, err

    def ravel_index(self, zi, yi, xi):
        return xi + self.uxgrid.n_face * zi

    def unravel_index(self, ei):
        zi = ei // self.uxgrid.n_face
        fi = ei % self.uxgrid.n_face
        return zi, fi

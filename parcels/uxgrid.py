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

    def __init__(self, grid: ux.grid.Grid, z: ux.UxDataArray) -> UxGrid:
        """
        Initializes the UxGrid with a uxarray grid and vertical coordinate array.

        Parameters
        ----------
        grid : ux.grid.Grid
            The uxarray grid object containing the unstructured grid data.
        z : ux.UxDataArray
            A 1D array of vertical coordinates (depths) corresponding to the vertical position of layer faces of the grid
            While uxarray allows nz to be spatially and temporally varying, the parcels.UxGrid class considers the case where
            the vertical coordinate is constant in time and space. This implies flat bottom topography and no moving ALE vertical grid.
        """
        self.uxgrid = grid
        if not isinstance(z, ux.UxDataArray):
            raise TypeError("z must be an instance of ux.UxDataArray")
        if z.ndim != 1:
            raise ValueError("z must be a 1D array of vertical coordinates")
        self.z = z

    def search(
        self, z: float, y: float, x: float, ei: int | None = None, search2D: bool = False
    ) -> tuple[np.ndarray, int]:
        tol = 1e-10

        def try_face(fid):
            bcoords, err = self.uxgrid._get_barycentric_coordinates(y, x, fid)
            if (bcoords >= 0).all() and (bcoords <= 1).all() and err < tol:
                return bcoords, fid
            return None, None

        def find_vertical_index() -> int:
            nz = self.z.shape[0]
            if nz == 1:
                return 0
            zf = self.z.values
            zi = np.searchsorted(zf, z, side="right") - 1  # Search assumes that z is positive and increasing with i
            if zi < 0 or zi >= nz - 1:
                raise FieldOutOfBoundError(z, y, x)
            return zi

        if not search2D:
            zi = find_vertical_index()  # Find the vertical cell center nearest to z
        else:
            zi = 0

        if ei is not None:
            zi, fi = self.unravel_index(ei)
            bcoords, fi_new = try_face(fi)
            if bcoords is not None:
                return bcoords, self.ravel_index(zi, fi_new)
            # Try neighbors of current face
            for neighbor in self.uxgrid.face_face_connectivity[fi, :]:
                if neighbor == -1:
                    continue
                bcoords, fi_new = try_face(neighbor)
                if bcoords is not None:
                    return bcoords, self.ravel_index(zi, fi_new)

        # Global fallback using spatial hash
        fi, bcoords = self.uxgrid.get_spatial_hash().query([[x, y]])
        if fi == -1:
            raise FieldOutOfBoundError(z, y, x)

        return bcoords, self.ravel_index(zi, fi)

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

    def ravel_index(self, zi, fi):
        """
        Converts a face index and a vertical index into a single encoded index.

        Parameters
        ----------
        zi : int
            Vertical index (not used in unstructured grids, but kept for compatibility).
        fi : int
            Face index.

        Returns
        -------
        int
            Encoded index combining the face index and vertical index.
        """
        return fi + self.uxgrid.n_face * zi

    def unravel_index(self, ei):
        """
        Converts a single encoded index back into a vertical index and face index.

        Parameters
        ----------
        ei : int
            Encoded index to be unraveled.

        Returns
        -------
        zi : int
            Vertical index.
        fi : int
            Face index.
        """
        zi = ei // self.uxgrid.n_face
        fi = ei % self.uxgrid.n_face
        return zi, fi

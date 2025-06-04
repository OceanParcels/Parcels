import numpy as np
import uxarray as ux
from uxarray.grid.neighbors import _barycentric_coordinates

from parcels.field import FieldOutOfBoundError  # Adjust import as necessary


class UxGrid(ux.grid.Grid):
    """
    Extension of uxarray's Grid class that supports point-location search
    for interpolation on unstructured grids.
    """

    @classmethod
    def from_uxgrid(cls, grid: ux.grid.Grid) -> "UxGrid":
        """
        Create a UxGrid instance from an existing uxarray Grid instance.

        Parameters
        ----------
        grid : uxarray.grid.Grid
            A previously constructed uxarray Grid object.

        Returns
        -------
        UxGrid
            A new UxGrid object with the same internal state.
        """
        if isinstance(grid, cls):
            return grid  # Already an extended grid

        new = cls.__new__(cls)
        new.__dict__.update(grid.__dict__)
        return new

    def search(
        self, field, z: float, y: float, x: float, ei: int | None = None, search2D: bool = False
    ) -> tuple[np.ndarray, int]:
        """
        Locate the unstructured grid face containing the point (x, y),
        returning interpolation weights and a face-based encoded index.

        Parameters
        ----------
        field : parcels.Field
            The field requesting the search. Used to access unravel_index(),
            ravel_index(), and igrid metadata.
        z : float
            Vertical coordinate of the query point. Currently ignored.
        y : float
            Latitude of the query point.
        x : float
            Longitude of the query point.
        ei : int, optional
            Encoded index to test reuse of previous face. If valid, neighbors
            of that face are also checked before falling back to global search.
        search2D : bool, default=False
            Ignored for now. Included for interface compatibility.

        Returns
        -------
        bcoords : np.ndarray
            Barycentric coordinates of the point in the containing face.
        ei : int
            Encoded index (e.g., raveled face index) corresponding to the face found.

        Raises
        ------
        FieldOutOfBoundError
            If no containing face is found within tolerance.
        """
        tol = 1e-10

        def try_face(fid):
            bcoords, err = self._get_barycentric_coordinates(y, x, fid)
            if (bcoords >= 0).all() and (bcoords <= 1).all() and err < tol:
                return bcoords, field.ravel_index(0, 0, fid)  # Z and time indices are 0 for now
            return None, None

        if ei is not None:
            zi, fi = field.unravel_index(ei)
            bcoords, ei_new = try_face(fi)
            if bcoords is not None:
                return bcoords, ei_new

            # Try neighbors of current face
            for neighbor in self.face_face_connectivity[fi, :]:
                if neighbor == -1:
                    continue
                bcoords, ei_new = try_face(neighbor)
                if bcoords is not None:
                    return bcoords, ei_new

        # Global fallback using spatial hash
        fi, bcoords = self.get_spatial_hash().query([[x, y]])
        if fi == -1:
            raise FieldOutOfBoundError(z, y, x)

        return bcoords, field.ravel_index(0, 0, fi)

    def _get_barycentric_coordinates(self, y, x, fi):
        """Checks if a point is inside a given face id on a UxGrid."""
        # Check if particle is in the same face, otherwise search again.
        n_nodes = self.n_nodes_per_face[fi].to_numpy()
        node_ids = self.face_node_connectivity[fi, 0:n_nodes]
        nodes = np.column_stack(
            (
                np.deg2rad(self.grid.node_lon[node_ids].to_numpy()),
                np.deg2rad(self.grid.node_lat[node_ids].to_numpy()),
            )
        )

        coord = np.deg2rad([x, y])
        bcoord = np.asarray(_barycentric_coordinates(nodes, coord))
        err = abs(np.dot(bcoord, nodes[:, 0]) - coord[0]) + abs(np.dot(bcoord, nodes[:, 1]) - coord[1])
        return bcoord, err

    def ravel_index(self, zi, yi, xi):
        """Return the flat index of the given grid points.

        Parameters
        ----------
        zi : int
            z index
        yi : int
            y index
        xi : int
            x index. When using an unstructured grid, this is the face index (fi)

        Returns
        -------
        int
            flat index
        """
        return xi + self.n_face * zi

    def unravel_index(self, ei):
        """Return the zi, yi, xi indices for a given flat index.
        Only used when working with fields on a structured grid.

        Parameters
        ----------
        ei : int
            The flat index to be unraveled.

        Returns
        -------
        zi : int
            The z index.
        yi : int
            The y index.
        xi : int
            The x index.
        """
        zi = ei // self.n_face
        fi = ei % self.n_face
        return zi, fi


def ensure_uxgrid(grid: ux.grid.Grid) -> UxGrid:
    """
    Ensure a given uxarray grid is an instance of UxGrid.

    Parameters
    ----------
    grid : uxarray.grid.Grid

    Returns
    -------
    UxGrid
    """
    return UxGrid.from_uxgrid(grid)

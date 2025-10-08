from __future__ import annotations

from typing import Literal

import numpy as np
import uxarray as ux

from parcels._core.basegrid import BaseGrid
from parcels._core.index_search import GRID_SEARCH_ERROR, _search_1d_array, uxgrid_point_in_cell
from parcels._typing import assert_valid_mesh

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
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        z = np.asarray(z, dtype=np.float32)

        zi, zeta = _search_1d_array(self.z.values, z)

        if np.any(ei):
            indices = self.unravel_index(ei)
            fi = indices.get("FACE")
            is_in_cell, coords = uxgrid_point_in_cell(self.uxgrid, y, x, fi, fi)
            y_check = y[is_in_cell == 0]
            x_check = x[is_in_cell == 0]
            zero_indices = np.where(is_in_cell == 0)[0]
        else:
            # Otherwise, we need to check all points
            fi = np.full(len(y), GRID_SEARCH_ERROR, dtype=np.int32)
            y_check = y
            x_check = x
            coords = -1.0 * np.ones((len(y), 3), dtype=np.float32)
            zero_indices = np.arange(len(y))

        if len(zero_indices) > 0:
            _, face_ids_q, coords_q = self.get_spatial_hash().query(y_check, x_check)
            coords[zero_indices, :] = coords_q
            fi[zero_indices] = face_ids_q

        return {"Z": (zi, zeta), "FACE": (fi, coords)}

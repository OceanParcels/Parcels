from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class BaseGrid(ABC):
    @abstractmethod
    def search(self, z: float, y: float, x: float, ei=None) -> dict[str, tuple[int, float | np.ndarray]]:
        """
        Perform a spatial (and optionally vertical) search to locate the grid element
        that contains a given point (x, y, z).

        This method delegates to grid-type-specific logic (e.g., structured or unstructured)
        to determine the appropriate indices and barycentric coordinates for evaluating a field.

        Parameters
        ----------
        z : float
            Vertical coordinate of the query point. If `search2D=True`, this may be ignored.
        y : float
            Latitude or vertical index, depending on grid type and projection.
        x : float
            Longitude or horizontal index, depending on grid type and projection.
        ei : int, optional
            A previously computed encoded index (e.g., raveled face or cell index). If provided,
            the search will first attempt to validate and reuse it before falling back to
            a global or local search strategy.
        search2D : bool, default=False
            If True, perform only a 2D search (x, y), ignoring the vertical component z.

        Returns
        -------
        dict
            A dictionary mapping spatial axis names to tuples of (index, barycentric_coordinates).
            The returned axes depend on the grid dimensionality and type:

            - 3D structured grid: {"Z": (zi, zeta), "Y": (yi, eta), "X": (xi, xsi)}
            - 2D structured grid: {"Y": (yi, eta), "X": (xi, xsi)}
            - 1D structured grid (depth): {"Z": (zi, zeta)}
            - Unstructured grid: {"Z": (zi, zeta), "FACE": (fi, bcoords)}

            Where:
            - index (int): The cell position of a particle along the given axis
            - barycentric_coordinates (float or np.ndarray): The coordinates defining
              a particle's position within the grid cell. For structured grids, this
              is a single coordinate per axis; for unstructured grids, this can be
              an array of coordinates for the face polygon.

        Raises
        ------
        FieldOutOfBoundError
            Raised when the queried point lies outside the bounds of the grid.
        NotImplementedError
            Raised if the search method is not implemented for the current grid type.
        """
        ...

    @abstractmethod
    def ravel_index(self, axis_indices: dict[str, int]) -> int:
        """
        Convert a dictionary of axis indices to a single encoded index (ei).

        This method takes the individual indices for each spatial axis and combines them
        into a single integer that uniquely identifies a grid cell. This encoded
        index can be used for efficient caching and lookup operations.

        Parameters
        ----------
        axis_indices : dict[str, int]
            A dictionary mapping axis names to their corresponding indices.
            The expected keys depend on the grid dimensionality and type:

            - 3D structured grid: {"Z": zi, "Y": yi, "X": xi}
            - 2D structured grid: {"Y": yi, "X": xi}
            - 1D structured grid: {"Z": zi}
            - Unstructured grid: {"Z": zi, "FACE": fi}

        Returns
        -------
        int
            The encoded index (ei) representing the unique grid cell or face.

        Raises
        ------
        KeyError
            Raised when required axis keys are missing from axis_indices.
        ValueError
            Raised when index values are out of bounds for the grid.
        NotImplementedError
            Raised if the method is not implemented for the current grid type.
        """
        ...

    @abstractmethod
    def unravel_index(self, ei: int) -> dict[str, int]:
        """
        Convert a single encoded index (ei) back to a dictionary of axis indices.

        This method is the inverse of ravel_index, taking an encoded index and
        decomposing it back into the individual indices for each spatial axis.

        Parameters
        ----------
        ei : int
            The encoded index representing a unique grid cell or face.

        Returns
        -------
        dict[str, int]
            A dictionary mapping axis names to their corresponding indices.
            The returned keys depend on the grid dimensionality and type:

            - 3D structured grid: {"Z": zi, "Y": yi, "X": xi}
            - 2D structured grid: {"Y": yi, "X": xi}
            - 1D structured grid: {"Z": zi}
            - Unstructured grid: {"Z": zi, "FACE": fi}

        Raises
        ------
        ValueError
            Raised when the encoded index is out of bounds or invalid for the grid.
        NotImplementedError
            Raised if the method is not implemented for the current grid type.
        """
        ...

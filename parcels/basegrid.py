from abc import ABC, abstractmethod


class BaseGrid(ABC):
    @abstractmethod
    def ravel_index(self, zi: int, yi: int, xi: int):
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
        ...

    @abstractmethod
    def unravel_index(self, ei: int):
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
        ...

    @abstractmethod
    def search(self, z: float, y: float, x: float, ei=None, search2D: bool = False):
        """
        Perform a spatial (and optionally vertical) search to locate the grid element
        that contains a given point (x, y, z).

        This method delegates to grid-type-specific logic (e.g., structured or unstructured)
        to determine the appropriate indices and interpolation coordinates for evaluating a field.

        Parameters
        ----------
        field : Field
            The field requesting the search. Used to extract grid-specific parameters,
            unravel index metadata, or perform coordinate system-specific operations.
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
        bcoords : np.ndarray or tuple
            Interpolation weights or barycentric coordinates within the containing cell/face.
            The interpretation of `bcoords` depends on the grid type.
        ei : int
            Encoded index of the identified grid cell or face. This value can be cached for
            future lookups to accelerate repeated searches.

        Raises
        ------
        FieldOutOfBoundError
            Raised when the queried point lies outside the bounds of the grid.
        NotImplementedError
            Raised if the search method is not implemented for the current grid type.
        """
        ...

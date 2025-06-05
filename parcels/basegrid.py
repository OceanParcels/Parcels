from abc import ABC, abstractmethod


class BaseGrid(ABC):
    @abstractmethod
    def search(self, z: float, y: float, x: float, ei=None, search2D: bool = False):
        """
        Perform a spatial (and optionally vertical) search to locate the grid element
        that contains a given point (x, y, z).

        This method delegates to grid-type-specific logic (e.g., structured or unstructured)
        to determine the appropriate indices and interpolation coordinates for evaluating a field.

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

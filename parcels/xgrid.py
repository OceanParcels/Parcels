from typing import Literal

import numpy as np
import numpy.typing as npt

from parcels import xgcm
from parcels.basegrid import BaseGrid
from parcels.tools.converters import TimeConverter


def get_dimensionality(axis: xgcm.Axis | None) -> int:
    if axis is None:
        return 1
    first_coord = list(axis.coords.items())[0]
    pos, coord = first_coord

    pos_to_dim = {  # TODO: These could do with being explicitly tested
        "center": lambda x: x,
        "left": lambda x: x,
        "right": lambda x: x,
        "inner": lambda x: x + 1,
        "outer": lambda x: x - 1,
    }

    n = axis._ds[coord].size
    return pos_to_dim[pos](n)


def get_time(axis: xgcm.Axis) -> npt.NDArray:
    return axis._ds[axis.coords["center"]].values


class XGrid(BaseGrid):
    """
    Class to represent a structured grid in Parcels. Wraps a xgcm-like Grid object (we use a trimmed down version of the xgcm.Grid class that is vendored with Parcels).

    This class provides methods and properties required for indexing and interpolating on the grid.
    """

    def __init__(self, grid: xgcm.Grid, mesh="flat"):
        self.xgcm_grid = grid
        self.mesh = mesh

        # ! Not ideal... Triggers computation on a throwaway item. Keeping for now for v3 compat, will be removed in v4.
        self.lonlat_minmax = np.array(
            [
                np.nanmin(self.xgcm_grid._ds["lon"]),
                np.nanmax(self.xgcm_grid._ds["lon"]),
                np.nanmin(self.xgcm_grid._ds["lat"]),
                np.nanmax(self.xgcm_grid._ds["lat"]),
            ]
        )

    @property
    def lon(self):
        """
        Note
        ----
        Included for compatibility with v3 codebase. May be removed in future.
        TODO v4: Evaluate
        """
        try:
            _ = self.xgcm_grid.axes["X"]
        except KeyError:
            return np.zeros(1)
        return self.xgcm_grid._ds["lon"].values

    @property
    def lat(self):
        """
        Note
        ----
        Included for compatibility with v3 codebase. May be removed in future.
        TODO v4: Evaluate
        """
        try:
            _ = self.xgcm_grid.axes["Y"]
        except KeyError:
            return np.zeros(1)
        return self.xgcm_grid._ds["lat"].values

    @property
    def depth(self):
        """
        Note
        ----
        Included for compatibility with v3 codebase. May be removed in future.
        TODO v4: Evaluate
        """
        try:
            _ = self.xgcm_grid.axes["Z"]
        except KeyError:
            return np.zeros(1)
        return self.xgcm_grid._ds["depth"].values

    @property
    def _datetimes(self):
        try:
            axis = self.xgcm_grid.axes["T"]
        except KeyError:
            return np.zeros(1)
        return get_time(axis)

    @property
    def time(self):
        return self._datetimes.astype(np.float64) / 1e9

    @property
    def xdim(self):
        return get_dimensionality(self.xgcm_grid.axes.get("X"))

    @property
    def ydim(self):
        return get_dimensionality(self.xgcm_grid.axes.get("Y"))

    @property
    def zdim(self):
        return get_dimensionality(self.xgcm_grid.axes.get("Z"))

    @property
    def tdim(self):
        return get_dimensionality(self.xgcm_grid.axes.get("T"))

    @property
    def time_origin(self):
        """
        Note
        ----
        Included for compatibility with v3 codebase. May be removed in future.
        TODO v4: Evaluate
        """
        return TimeConverter(self._datetimes[0])

    @property
    def _z4d(self) -> Literal[0, 1]:
        """
        Note
        ----
        Included for compatibility with v3 codebase. May be removed in future.
        TODO v4: Evaluate
        """
        return 1 if self.depth.shape == 4 else 0

    @property
    def zonal_periodic(self): ...  # ? hmmm, from v3, do we still need this?

    @property
    def _gtype(self):
        """This class is created *purely* for compatibility with v3 code and will be removed
        or changed in future.

        TODO: Remove
        """
        from parcels.grid import GridType

        if len(self.lon.shape) <= 1:
            if self.depth is None or len(self.depth.shape) <= 1:
                return GridType.RectilinearZGrid
            else:
                return GridType.RectilinearSGrid
        else:
            if self.depth is None or len(self.depth.shape) <= 1:
                return GridType.CurvilinearZGrid
            else:
                return GridType.CurvilinearSGrid

    def search(self, z, y, x, ei=None, search2D=False): ...

from typing import Literal

import numpy as np
import numpy.typing as npt

from parcels.tools.converters import TimeConverter
from parcels.v4.grid import Axis
from parcels.v4.grid import Grid as NewGrid


def get_dimensionality(axis: Axis | None) -> int:
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


def get_time(axis: Axis) -> npt.NDArray:
    return axis._ds[axis.coords["center"]].values


class GridAdapter:
    def __init__(self, grid: NewGrid, mesh="flat"):
        self.grid = grid
        self.mesh = mesh

        # ! Not ideal... Triggers computation on a throwaway item. If adapter is still needed in codebase, and this is prohibitively expensive, perhaps store GridAdapter on Field object instead of Grid
        self.lonlat_minmax = np.array(
            [
                np.nanmin(self.grid._ds["lon"]),
                np.nanmax(self.grid._ds["lon"]),
                np.nanmin(self.grid._ds["lat"]),
                np.nanmax(self.grid._ds["lat"]),
            ]
        )

    @property
    def lon(self):
        try:
            _ = self.grid.axes["X"]
        except KeyError:
            return np.zeros(1)
        return self.grid._ds["lon"].values

    @property
    def lat(self):
        try:
            _ = self.grid.axes["Y"]
        except KeyError:
            return np.zeros(1)
        return self.grid._ds["lat"].values

    @property
    def depth(self):
        try:
            _ = self.grid.axes["Z"]
        except KeyError:
            return np.zeros(1)
        return self.grid._ds["depth"].values

    @property
    def time(self):
        try:
            axis = self.grid.axes["T"]
        except KeyError:
            return np.zeros(1)
        return get_time(axis)

    @property
    def xdim(self):
        return get_dimensionality(self.grid.axes.get("X"))

    @property
    def ydim(self):
        return get_dimensionality(self.grid.axes.get("Y"))

    @property
    def zdim(self):
        return get_dimensionality(self.grid.axes.get("Z"))

    @property
    def tdim(self):
        return get_dimensionality(self.grid.axes.get("T"))

    @property
    def time_origin(self):
        return TimeConverter(self.time[0])

    @property
    def _z4d(self) -> Literal[0, 1]:
        return 1 if self.depth.shape == 4 else 0

    @property
    def zonal_periodic(self): ...  # ? hmmm

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

    @staticmethod
    def create_grid(lon, lat, depth, time, time_origin, mesh, **kwargs): ...  # ? hmmm

    def _check_zonal_periodic(self): ...  # ? hmmm

    def _add_Sdepth_periodic_halo(self, zonal, meridional, halosize): ...  # ? hmmm

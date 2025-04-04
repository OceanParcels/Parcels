from typing import Literal

import numpy as np
import numpy.typing as npt

from parcels.tools.converters import TimeConverter
from parcels.v4.grid import Axis, Grid


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


class GridAdapter(Grid):
    def __init__(self, ds, mesh="flat", *args, **kwargs):
        super().__init__(ds, *args, **kwargs)
        self.mesh = mesh

    @property
    def lon(self):
        try:
            _ = self.axes["X"]
        except KeyError:
            return np.zeros(1)
        return self._ds["lon"].values

    @property
    def lat(self):
        try:
            _ = self.axes["Y"]
        except KeyError:
            return np.zeros(1)
        return self._ds["lat"].values

    @property
    def depth(self):
        try:
            _ = self.axes["Z"]
        except KeyError:
            return np.zeros(1)
        return self._ds["depth"].values

    @property
    def time(self):
        try:
            axis = self.axes["T"]
        except KeyError:
            return np.zeros(1)
        return get_time(axis)

    @property
    def xdim(self):
        return get_dimensionality(self.axes.get("X"))

    @property
    def ydim(self):
        return get_dimensionality(self.axes.get("Y"))

    @property
    def zdim(self):
        return get_dimensionality(self.axes.get("Z"))

    @property
    def tdim(self):
        return get_dimensionality(self.axes.get("T"))

    @property
    def time_origin(self):
        return TimeConverter(self.time[0])

    @property
    def _z4d(self) -> Literal[0, 1]:
        return 1 if self.depth.shape == 4 else 0

    @property
    def zonal_periodic(self): ...  # ? hmmm

    @property
    def lonlat_minmax(self): ...  # ? hmmm

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

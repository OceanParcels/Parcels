from ctypes import Structure, c_int, c_void_p
from enum import IntEnum

import numpy as np

from parcels.tools.converters import Geographic, GeographicPolar, UnitConverter

__all__ = [
    "CGrid",
    "CurvilinearSGrid",
    "CurvilinearZGrid",
    "Grid",
    "GridCode",
    "GridType",
    "RectilinearSGrid",
    "RectilinearZGrid",
]


class GridType(IntEnum):
    RectilinearZGrid = 0
    RectilinearSGrid = 1
    CurvilinearZGrid = 2
    CurvilinearSGrid = 3


# GridCode has been renamed to GridType for consistency.
# TODO: Remove alias in Parcels v4
GridCode = GridType


class CGrid(Structure):
    _fields_ = [("gtype", c_int), ("grid", c_void_p)]


class Grid:
    def __init__(self, time_full=None):
        self.time_full = time_full


class RectilinearGrid(Grid): ...


class RectilinearZGrid(RectilinearGrid): ...


class RectilinearSGrid(RectilinearGrid): ...


class CurvilinearGrid(Grid): ...


class CurvilinearZGrid(CurvilinearGrid): ...


class CurvilinearSGrid(CurvilinearGrid): ...


def _calc_cell_edge_sizes(grid: RectilinearGrid) -> None:
    """Method to calculate cell sizes based on numpy.gradient method.

    Currently only works for Rectilinear Grids. Operates in place adding a `cell_edge_sizes`
    attribute to the grid.
    """
    if not grid.cell_edge_sizes:
        if grid._gtype in (GridType.RectilinearZGrid, GridType.RectilinearSGrid):
            grid.cell_edge_sizes["x"] = np.zeros(
                (grid.ydim, grid.xdim), dtype=np.float32
            )
            grid.cell_edge_sizes["y"] = np.zeros(
                (grid.ydim, grid.xdim), dtype=np.float32
            )

            x_conv = GeographicPolar() if grid.mesh == "spherical" else UnitConverter()
            y_conv = Geographic() if grid.mesh == "spherical" else UnitConverter()
            for y, (lat, dy) in enumerate(
                zip(grid.lat, np.gradient(grid.lat), strict=False)
            ):
                for x, (lon, dx) in enumerate(
                    zip(grid.lon, np.gradient(grid.lon), strict=False)
                ):
                    grid.cell_edge_sizes["x"][y, x] = x_conv.to_source(
                        dx, grid.depth[0], lat, lon
                    )
                    grid.cell_edge_sizes["y"][y, x] = y_conv.to_source(
                        dy, grid.depth[0], lat, lon
                    )
        else:
            raise ValueError(
                f"_cell_edge_sizes() not implemented for {grid._gtype} grids. "
                "You can provide Field.grid.cell_edge_sizes yourself by in, e.g., "
                "NEMO using the e1u fields etc from the mesh_mask.nc file."
            )


def _calc_cell_areas(grid: RectilinearGrid) -> np.ndarray:
    if not grid.cell_edge_sizes:
        _calc_cell_edge_sizes(grid)
    return grid.cell_edge_sizes["x"] * grid.cell_edge_sizes["y"]

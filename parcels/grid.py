import numpy as np

from parcels.numba.grid.curvilinear import CurvilinearSGrid, CurvilinearZGrid
from parcels.numba.grid.rectilinear import RectilinearSGrid, RectilinearZGrid
from parcels.numba.grid import GridStatus


class Grid():
    @staticmethod
    def create_grid(lon, lat, depth, time, time_origin, mesh, **kwargs):
        if not isinstance(lon, np.ndarray):
            lon = np.array(lon)
        if not isinstance(lat, np.ndarray):
            lat = np.array(lat)
        if not (depth is None or isinstance(depth, np.ndarray)):
            depth = np.array(depth)
        if len(lon.shape) <= 1:
            if depth is None or len(depth.shape) <= 1:
                return RectilinearZGrid(
                    lon, lat, depth, time, time_origin=time_origin, mesh=mesh,
                    **kwargs)
            else:
                return RectilinearSGrid(
                    lon, lat, depth, time, time_origin=time_origin, mesh=mesh,
                    **kwargs)
        else:
            if depth is None or len(depth.shape) <= 1:
                return CurvilinearZGrid(
                    lon, lat, depth, time, time_origin=time_origin, mesh=mesh,
                    **kwargs)
            else:
                return CurvilinearSGrid(
                    lon, lat, depth, time, time_origin=time_origin, mesh=mesh,
                    **kwargs)

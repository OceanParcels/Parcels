import functools
import warnings
from ctypes import POINTER, Structure, c_double, c_float, c_int, c_void_p, cast, pointer
from enum import IntEnum
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from parcels._typing import Mesh, UpdateStatus, assert_valid_mesh
from parcels.tools._helpers import deprecated_made_private
from parcels.tools.converters import TimeConverter
from parcels.tools.warnings import FieldSetWarning

class CGrid(Structure):
    _fields_ = [("gtype", c_int), ("grid", c_void_p)]

class BaseGrid(ABC):
    """Abstract Base Class for Grid types."""
    def __init__(
        self,
        lon: npt.NDArray,
        lat: npt.NDArray,
        time: npt.NDArray | None,
        time_origin: TimeConverter | None,
        mesh: Mesh,
    ):
        self._ti = -1
        self._update_status: UpdateStatus | None = None
        if not lon.flags["C_CONTIGUOUS"]:
            lon = np.array(lon, order="C")
        if not lat.flags["C_CONTIGUOUS"]:
            lat = np.array(lat, order="C")
        time = np.zeros(1, dtype=np.float64) if time is None else time
        if not time.flags["C_CONTIGUOUS"]:
            time = np.array(time, order="C")
        if not lon.dtype == np.float32:
            lon = lon.astype(np.float32)
        if not lat.dtype == np.float32:
            lat = lat.astype(np.float32)
        if not time.dtype == np.float64:
            assert isinstance(
                time[0], (np.integer, np.floating, float, int)
            ), "Time vector must be an array of int or floats"
            time = time.astype(np.float64)

        self._lon = lon
        self._lat = lat
        self.time = time
        self.time_full = self.time  # needed for deferred_loaded Fields
        self._time_origin = TimeConverter() if time_origin is None else time_origin
        assert isinstance(self.time_origin, TimeConverter), "time_origin needs to be a TimeConverter object"
        assert_valid_mesh(mesh)
        self._mesh = mesh
        self._cstruct = None
        self._cell_edge_sizes: dict[str, npt.NDArray] = {}
        # self._zonal_periodic = False
        # self._zonal_halo = 0
        # self._meridional_halo = 0
        # self._lat_flipped = False
        self._defer_load = False
        self._lonlat_minmax = np.array(
            [np.nanmin(lon), np.nanmax(lon), np.nanmin(lat), np.nanmax(lat)], dtype=np.float32
        )
        self.periods = 0
        self._load_chunk: npt.NDArray = np.array([])
        self.chunk_info = None
        self.chunksize = None
        self._add_last_periodic_data_timestep = False
        self.depth_field = None

    def __repr__(self):
        with np.printoptions(threshold=5, suppress=True, linewidth=120, formatter={"float": "{: 0.2f}".format}):
            return (
                f"{type(self).__name__}("
                f"lon={self.lon!r}, lat={self.lat!r}, time={self.time!r}, "
                f"time_origin={self.time_origin!r}, mesh={self.mesh!r})"
            )
        
    @property
    def lon(self):
        return self._lon

    @property
    def lat(self):
        return self._lat

    @property
    def depth(self):
        return self._depth
    
    @property
    def mesh(self):
        return self._mesh
    
    @property
    def lonlat_minmax(self):
        return self._lonlat_minmax

    @property
    def cell_edge_sizes(self):
        return self._cell_edge_sizes
    
    @property
    def defer_load(self):
        return self._defer_load
    
    @property
    def time_origin(self):
        return self._time_origin

    @property
    def ctypes_struct(self):
        # This is unnecessary for the moment, but it could be useful when going will fully unstructured grids
        self._cgrid = cast(pointer(self._child_ctypes_struct), c_void_p)
        cstruct = CGrid(self._gtype, self._cgrid.value)
        return cstruct

    @property
    @abstractmethod
    def _child_ctypes_struct(self):
        pass

    @abstractmethod
    def lon_grid_to_target(self):
        pass

    @abstractmethod
    def lon_grid_to_source(self):
        pass
    
    @abstractmethod
    def lon_particle_to_target(self, lon):
        pass

    def _computeTimeChunk(self, f, time, signdt):
        nextTime_loc = np.inf if signdt >= 0 else -np.inf
        periods = self.periods.value if isinstance(self.periods, c_int) else self.periods
        prev_time_indices = self.time
        if self._update_status == "not_updated":
            if self._ti >= 0:
                if (
                    time - periods * (self.time_full[-1] - self.time_full[0]) < self.time[0]
                    or time - periods * (self.time_full[-1] - self.time_full[0]) > self.time[1]
                ):
                    self._ti = -1  # reset
                elif signdt >= 0 and (
                    time - periods * (self.time_full[-1] - self.time_full[0]) < self.time_full[0]
                    or time - periods * (self.time_full[-1] - self.time_full[0]) >= self.time_full[-1]
                ):
                    self._ti = -1  # reset
                elif signdt < 0 and (
                    time - periods * (self.time_full[-1] - self.time_full[0]) <= self.time_full[0]
                    or time - periods * (self.time_full[-1] - self.time_full[0]) > self.time_full[-1]
                ):
                    self._ti = -1  # reset
                elif (
                    signdt >= 0
                    and time - periods * (self.time_full[-1] - self.time_full[0]) >= self.time[1]
                    and self._ti < len(self.time_full) - 2
                ):
                    self._ti += 1
                    self.time = self.time_full[self._ti : self._ti + 2]
                    self._update_status = "updated"
                elif (
                    signdt < 0
                    and time - periods * (self.time_full[-1] - self.time_full[0]) <= self.time[0]
                    and self._ti > 0
                ):
                    self._ti -= 1
                    self.time = self.time_full[self._ti : self._ti + 2]
                    self._update_status = "updated"
            if self._ti == -1:
                self.time = self.time_full
                self._ti, _ = f._time_index(time)
                periods = self.periods.value if isinstance(self.periods, c_int) else self.periods
                if (
                    signdt == -1
                    and self._ti == 0
                    and (time - periods * (self.time_full[-1] - self.time_full[0])) == self.time[0]
                    and f.time_periodic
                ):
                    self._ti = len(self.time) - 1
                    periods -= 1
                if signdt == -1 and self._ti > 0 and self.time_full[self._ti] == time:
                    self._ti -= 1
                if self._ti >= len(self.time_full) - 1:
                    self._ti = len(self.time_full) - 2

                self.time = self.time_full[self._ti : self._ti + 2]
                self.tdim = 2
                if prev_time_indices is None or len(prev_time_indices) != 2 or len(prev_time_indices) != len(self.time):
                    self._update_status = "first_updated"
                elif functools.reduce(
                    lambda i, j: i and j, map(lambda m, k: m == k, self.time, prev_time_indices), True
                ) and len(prev_time_indices) == len(self.time):
                    self._update_status = "not_updated"
                elif functools.reduce(
                    lambda i, j: i and j, map(lambda m, k: m == k, self.time[:1], prev_time_indices[:1]), True
                ) and len(prev_time_indices) == len(self.time):
                    self._update_status = "updated"
                else:
                    self._update_status = "first_updated"
            if signdt >= 0 and (self._ti < len(self.time_full) - 2 or not f.allow_time_extrapolation):
                nextTime_loc = self.time[1] + periods * (self.time_full[-1] - self.time_full[0])
            elif signdt < 0 and (self._ti > 0 or not f.allow_time_extrapolation):
                nextTime_loc = self.time[0] + periods * (self.time_full[-1] - self.time_full[0])
        return nextTime_loc

    @property
    def _chunk_not_loaded(self):
        return 0

    @property
    def _chunk_loading_requested(self):
        return 1

    @property
    def _chunk_loaded_touched(self):
        return 2
    
    @property
    def _chunk_deprecated(self):
        return 3
    
    @property
    def _chunk_loaded(self):
        return [2, 3]
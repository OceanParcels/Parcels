import functools
import warnings
from ctypes import POINTER, Structure, c_double, c_float, c_int, c_void_p, cast, pointer
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from parcels._typing import Mesh, UpdateStatus, assert_valid_mesh
from parcels.tools._helpers import deprecated_made_private
from parcels.tools.converters import TimeConverter
from parcels.tools.warnings import FieldSetWarning
from parcels.basegrid import BaseGrid
from parcels.basegrid import CGrid

# Note :
# For variable placement in FESOM - see https://fesom2.readthedocs.io/en/latest/geometry.html
__all__ = [
    "CGrid",
    "UGrid"
]

class UGrid(BaseGrid):
    """Grid class that defines a (spatial and temporal) unstructured grid on which Fields are defined."""

    def __init__(self,face_node_connectivity,*args, **kwargs):
        self._face_node_connectivity = face_node_connectivity # nface_node_connectivity x n_nodes_per_element array listing the vertex ids of each face_node_connectivity.
        super().__init__(*args, **kwargs)

        if not isinstance(self.lon, np.ndarray):
            raise TypeError("lon must be a NumPy array.")
        
        if self.lon.ndim != 1:
            raise ValueError("lon must be a 1-D array.")
        
        if not isinstance(self.lat, np.ndarray):
            raise TypeError("lat must be a NumPy array.")
        
        if self.lat.ndim != 1:
            raise ValueError("lat must be a 1-D array.")
        
        if self.lon.shape != self.lat.shape:
            raise ValueError("lon and lat must have the same shape.")
        
        if not isinstance(self.face_node_connectivity, np.ndarray):
            raise TypeError("face_node_connectivity must be a NumPy array.")
        
        if self.face_node_connectivity.ndim != 2:
            raise ValueError("face_node_connectivity must be a 2-D array.")
        
        if self.face_node_connectivity.shape[1] > 3: # Enforce triangle face_node_connectivity
            raise ValueError("face_node_connectivity must be a 2-D array with at most 3 columns.")
        
        self._n_face = self.face_node_connectivity.shape[0]
        self._nodes_per_element = face_node_connectivity.shape[1] 
        self._n_vertices = self.lon.shape[0]

    # The lon and lat fields are assumed identical to the 
    # node_lon and node_lat fields in UXArray.Grid
    # data structure.
    @property
    def node_lon(self):
        return self.lon
    
    @property
    def node_lat(self): 
        return self.lat
    
    @property
    def face_node_connectivity(self):
        return self._face_node_connectivity
    
    @property
    def n_face(self):
        return self._n_face

    @property
    def nodes_per_element(self):
        return self._nodes_per_element
    
    @property
    def n_vertices(self):
        return self._n_vertices

        

    @staticmethod
    def create_grid(
        lon: npt.ArrayLike,
        lat: npt.ArrayLike,
        face_node_connectivity: npt.ArrayLike,
        depth,
        time,
        time_origin,
        mesh: Mesh,
        **kwargs,
    ):
        lon = np.array(lon)
        lat = np.array(lat)
        face_node_connectivity = np.array(face_node_connectivity)

        if depth is not None:
            depth = np.array(depth)

        return UGrid(lon, lat, face_node_connectivity, depth, time, time_origin=time_origin, mesh=mesh, **kwargs)

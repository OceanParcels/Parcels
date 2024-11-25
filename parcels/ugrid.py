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
from parcels.hashgrid import HashGrid

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
        self._hashgrid = None
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

    @staticmethod
    def create_hashgrid(self, hash_cell_size_scalefac=1.0):
        """Create the hashgrid attribute and populates the ugrid_element lookup table that
        relates the hash cell indices to the associated ugrid elements. The hashgrid 
        spacing is based on the median bounding box diagonal length of all triangles in the mesh.
        The `hash_cell_size_scalefac` parameter can be used to scale the hash cell size. Values
        greater than 1 will result in larger hash cells, likely having more unstructured elements
        per hash cell.
        """
        import numpy as np
        
        # Initialize a list to store bounding box diagonals
        diagonals = np.zeros(self.face_node_connectivity.shape[0])
        vertices = np.column_stack((self.lon,self.lat))

        # Loop over each triangle element
        k = 0
        for triangle in self.face_node_connectivity:
            # Get the coordinates of the triangle's vertices
            triangle_vertices = vertices[triangle]
            
            # Calculate the bounding box of the triangle
            x_min, y_min = np.min(triangle_vertices, axis=0)
            x_max, y_max = np.max(triangle_vertices, axis=0)
            
            # Calculate the diagonal length of the bounding box
            diagonal = np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)
            
            # Store the diagonal length
            diagonals[k] = diagonal

            k+=1
        
        # Use the median diagonal as a basis for the cell size
        dh = np.median(diagonals)*hash_cell_size_scalefac
        
        Nx = int((self.lon.max() - self.lon.min()) / dh) + 1
        Ny = int((self.lat.max() - self.lat.min()) / dh) + 1
        self.hashgrid = HashGrid(self.lon.min(), self.lat.min(), dh, dh, Nx, Ny)
        self.hashgrid.populate_ugrid_elements(self.lon, self.lat, self.face_node_connectivity)

    @staticmethod
    def barycentric_coordinates(self,xP, yP):
        """
        Compute the barycentric coordinates of a particle in a triangular element
        
        Parameters:
        - xP, yP: The coordinates of the particle
        - triangle_vertices (np.ndarray) : The vertices of the triangle as a (3,2) array.
        
        Returns:
        - The barycentric coordinates (l1,l2,l3)
        - True if the point is inside the triangle, False otherwise.
        """
        
        xv = np.squeeze(self.lon)
        yv = np.squeeze(self.lat[:,1])

        A_ABC = xv[0]*(yv[1]-yv[2]) + xv[1]*(yv[2]-yv[0]) + xv[2]*(yv[0]-yv[1])
        A_BCP = xv[1]*(yv[2]-yP   ) + xv[2]*(yP   -yv[1]) + xP   *(yv[1]-yv[2])
        A_CAP = xv[2]*(yv[0]-yP   ) + xv[0]*(yP   -yv[2]) + xP   *(yv[2]-yv[0])
        A_ABP = xv[0]*(yv[1]-yP   ) + xv[1]*(yP   -yv[0]) + xP   *(yv[0]-yv[1])

        # Compute the vectors
        l1 = A_BCP/A_ABC
        l2 = A_CAP/A_ABC
        l3 = A_ABP/A_ABC
        
        inside_triangle = all( [l1 >= 0.0, l1 <= 1.0, 
                                l2 >= 0.0, l2 <= 1.0,
                                l3 >= 0.0, l3 <= 1.0] )
        
        return l1,l2,l3,inside_triangle
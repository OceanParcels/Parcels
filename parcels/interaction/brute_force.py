import numpy as np

from .geo_utils import fast_distance
from .geo_utils import fast_3d_distance
from .base_neighbor import BaseNeighborSearchGeo
from .base_neighbor import BaseNeighborSearchGeo3D


class BruteGeoNSearch(BaseNeighborSearchGeo):
    '''Brute force implementation to find the neighbors.'''
    def find_neighbors(self, particle_id):
        distances = fast_distance(*self._values[:, particle_id],
                                  self._values[0, :], self._values[1, :])
        idx = np.where(distances < self.max_dist)[0]
        return idx


class BruteNeighborSearch(BaseNeighborSearchGeo3D):
    '''Brute force implementation to find the neighbors.'''
    name = "brute force"

    def __init__(self, values, max_dist, depth_factor=1):
        super(BruteNeighborSearch, self).__init__(values, max_dist)
        self.depth_factor = depth_factor

    def find_neighbors(self, particle_id):
        distances = fast_3d_distance(*self._values[:, particle_id],
                                     self._values[0, :], self._values[1, :],
                                     self._values[2, :],
                                     depth_factor=self.depth_factor)
        idx = np.where(distances < self.max_dist)[0]
        return idx

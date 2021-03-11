import numpy as np

from .geo_utils import fast_distance
from .geo_utils import relative_3d_distance
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

    def find_neighbors_by_idx(self, particle_id):
        distances = relative_3d_distance(
            *self._values[:, particle_id],
            self._values[0, :], self._values[1, :],
            self._values[2, :],
            interaction_distance=self.interaction_distance,
            interaction_depth=self.interaction_depth)
        idx = np.where(distances < 1)[0]
        return idx

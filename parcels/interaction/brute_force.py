import numpy as np

from parcels.interaction.spherical_utils import relative_3d_distance
from .base_neighbor import BaseFlatNeighborSearch
from .base_neighbor import BaseSphericalNeighborSearch


class BruteFlatNeighborSearch(BaseFlatNeighborSearch):
    '''Brute force implementation to find the neighbors.'''
    name = "brute force"

    def find_neighbors_by_coor(self, coor):
#         coor = self.values[:, particle_idx].reshape(3, 1)
        distances = np.sqrt(np.sum(((self._values-coor)/self.inter_dist)**2, axis=0))
#         distances = fast_distance(*self._values[:, particle_idx],
#                                   self._values[0, :], self._values[1, :])
        print(distances.shape)
        idx = np.where(distances < 1)[0]
        return idx


class BruteSphericalNeighborSearch(BaseSphericalNeighborSearch):
    '''Brute force implementation to find the neighbors.'''
    name = "brute force"

    def find_neighbors_by_coor(self, coor):
        distances = relative_3d_distance(
            *coor,
            self._values[0, :], self._values[1, :],
            self._values[2, :],
            interaction_distance=self.interaction_distance,
            interaction_depth=self.interaction_depth)
        idx = np.where(distances < 1)[0]
        return idx

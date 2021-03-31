import numpy as np

from parcels.interaction.spherical_utils import relative_3d_distance
from .base_neighbor import BaseFlatNeighborSearch
from .base_neighbor import BaseSphericalNeighborSearch


class BruteFlatNeighborSearch(BaseFlatNeighborSearch):
    '''Brute force implementation to find the neighbors.'''
    name = "brute force"

    def find_neighbors_by_coor(self, coor):
        active_values = self._values[:, self._active_idx]
        distances = np.sqrt(np.sum(((active_values-coor)/self.inter_dist)**2, axis=0))
        idx = np.where(distances < 1)[0]
        return self._active_idx[idx]


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

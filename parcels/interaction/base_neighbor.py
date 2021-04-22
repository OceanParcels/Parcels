from abc import ABC, abstractmethod

import numpy as np
from parcels.interaction.distance_utils import spherical_distance


class BaseNeighborSearch(ABC):
    name = "unknown"

    def __init__(self, interaction_distance, interaction_depth, max_depth=100000):
        self.interaction_depth = interaction_depth
        self.interaction_distance = interaction_distance
        self.inter_dist = np.array(
            [self.interaction_distance, self.interaction_distance,
             self.interaction_depth]).reshape(3, 1)
        self.max_depth = max_depth
        self._values = None
        self._active_mask = None

    def find_neighbors_by_idx(self, particle_idx):
        '''Find neighbors with particle_id.'''
        coor = self._values[:, particle_idx].reshape(3, 1)
        return self.find_neighbors_by_coor(coor)

    @abstractmethod
    def find_neighbors_by_coor(self, coor):
        raise NotImplementedError

    def update_values(self, new_values, new_active_mask=None):
        self.rebuild(new_values, new_active_mask)

    def rebuild(self, values, active_mask=-1):
        if values is not None:
            self._values = values
        if active_mask is None:
            self._active_mask = np.arange(self._values.shape[1])
        if not (isinstance(active_mask, int) and active_mask == -1):
            self._active_mask = active_mask
        self._active_idx = self.active_idx

    @property
    def active_idx(self):
        if self._active_mask is None:
            return np.arange(self._values.shape[1])
        return np.where(self._active_mask)[0]

    @abstractmethod
    def _distance(self, coor, subset_idx):
        raise NotImplementedError

    def _get_close_neighbor_dist(self, coor, subset_idx):
        surf_distance, depth_distance = self._distance(coor, subset_idx)
        rel_distances = np.sqrt((surf_distance/self.interaction_distance)**2
                                + (depth_distance/self.interaction_depth)**2)
        rel_neighbor_idx = np.where(rel_distances < 1)[0]
        neighbor_idx = subset_idx[rel_neighbor_idx]
        distances = np.vstack((surf_distance[rel_neighbor_idx],
                               depth_distance[rel_neighbor_idx]))
        return neighbor_idx, distances


class BaseFlatNeighborSearch(BaseNeighborSearch):
    def _distance(self, coor, subset_idx):
        surf_distance = np.sqrt(np.sum((
            self._values[:2, subset_idx] - coor[:2])**2,
            axis=0))
        depth_distance = np.abs(self._values[2, subset_idx]-coor[2])
        return (surf_distance, depth_distance)


class BaseSphericalNeighborSearch(BaseNeighborSearch):
    def _distance(self, coor, subset_idx):
        return spherical_distance(
            *coor,
            self._values[0, subset_idx],
            self._values[1, subset_idx],
            self._values[2, subset_idx],
        )

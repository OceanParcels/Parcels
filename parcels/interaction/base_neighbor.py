from abc import ABC, abstractmethod

import numpy as np
from parcels.interaction.distance_utils import spherical_distance


class BaseNeighborSearch(ABC):
    """Base class for searching particles in the neighborhood."""
    name = "unknown"

    def __init__(self, interaction_distance, interaction_depth,
                 max_depth=100000):
        self.interaction_depth = interaction_depth
        self.interaction_distance = interaction_distance
        self.inter_dist = np.array(
            [self.interaction_distance, self.interaction_distance,
             self.interaction_depth]).reshape(3, 1)
        self.max_depth = max_depth
        self._values = None
        self._active_mask = None

    @abstractmethod
    def find_neighbors_by_coor(self, coor):
        '''Get the neighbors around a certain location.

        :param coor: Numpy array with [lat, long, depth].
        :returns List of particle indices.
        '''
        raise NotImplementedError

    def find_neighbors_by_idx(self, particle_idx):
        '''Get the neighbors around a certain particle.

        Mainly useful for Structure of Array (SoA) datastructure

        :param particle_idx: index of the particle (SoA).
        :returns List of particle indices
        '''
        coor = self._values[:, particle_idx].reshape(3, 1)
        return self.find_neighbors_by_coor(coor)

    def update_values(self, new_values, new_active_mask=None):
        '''Update the coordinates of the particles.

        This is a default implementation simply rebuilds the structure.
        If the rebuilding is slow, a faster implementation can be provided.

        :param new_values: new coordinates of the particles.
        :param new_active_mask: boolean array indicating active particles.
        '''
        self.rebuild(new_values, new_active_mask)

    def rebuild(self, values, active_mask=-1):
        """Rebuild the neighbor structure from scratch.

        :param values: new coordinates of the particles.
        :param active_mask: boolean array indicating active particles.
        """
        if values is not None:
            self._values = values
        if active_mask is None:
            self._active_mask = np.arange(self._values.shape[1])

        # If active_mask == -1, then don't update the active mask.
        if not (isinstance(active_mask, int) and active_mask == -1):
            self._active_mask = active_mask
        self._active_idx = self.active_idx

    @property
    def active_idx(self):
        "Indices of the currently active mask."
        if self._active_mask is None:
            return np.arange(self._values.shape[1])
        return np.where(self._active_mask)[0]

    @abstractmethod
    def _distance(self, coor, subset_idx):
        """Distance between a coordinate and particles

        Distance depends on the mesh (spherical/flat).

        :param coor: Numpy array with 3D coordinates.
        :param subset_idx: Indices of the particles to compute the distance to.
        :returns surface_dist: distance along the surface
        :returns depth_dist: distance in the z-direction.
        """
        raise NotImplementedError

    def _get_close_neighbor_dist(self, coor, subset_idx):
        """Compute distances and remove non-neighbors.

        :param coor: Numpy array with 3D coordinates.
        :param subset_idx: Indices of the particles to compute the distance to.
        :returns neighbor_idx: Indices within the interaction distance.
        :returns distances: Distance between coor and the neighbor particles.
        """
        surf_distance, depth_distance = self._distance(coor, subset_idx)
        rel_distances = np.sqrt((surf_distance/self.interaction_distance)**2
                                + (depth_distance/self.interaction_depth)**2)
        rel_neighbor_idx = np.where(rel_distances < 1)[0]
        neighbor_idx = subset_idx[rel_neighbor_idx]
        distances = np.vstack((surf_distance[rel_neighbor_idx],
                               depth_distance[rel_neighbor_idx]))
        return neighbor_idx, distances


class BaseFlatNeighborSearch(BaseNeighborSearch):
    "Base class for neighbor searches with a flat mesh."
    def _distance(self, coor, subset_idx):
        coor = coor.reshape(3, 1)
        surf_distance = np.sqrt(np.sum((
            self._values[:2, subset_idx] - coor[:2])**2,
            axis=0))
        depth_distance = np.abs(self._values[2, subset_idx]-coor[2])
        return (surf_distance, depth_distance)


class BaseSphericalNeighborSearch(BaseNeighborSearch):
    "Base class for a neighbor search with a spherical mesh."
    def _distance(self, coor, subset_idx):
        return spherical_distance(
            *coor,
            self._values[0, subset_idx],
            self._values[1, subset_idx],
            self._values[2, subset_idx],
        )

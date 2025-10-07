from abc import ABC, abstractmethod

import numpy as np

from parcels.interaction.neighborsearch.distanceutils import spherical_distance


class BaseNeighborSearch(ABC):
    """Base class for searching particles in the neighborhood.

    The data structure of the class (and subclasses) only contain spatial
    information. Additionally its input is in array format (3, n_particles),
    which makes it the most efficient with the SoA (structure of arrays) data
    structure.
    """

    def __init__(self, inter_dist_vert, inter_dist_horiz, max_depth=100000, periodic_domain_zonal=None):
        """Initialize neighbor search


        Parameters
        ----------
        inter_dist_vert : float
            Interaction distance (vertical) in m.
        inter_dist_horiz : float
            interaction distance (horizontal) in m
        max_depth : float, optional
            Maximum depth of the particles (default is 100000m).
        """
        self.inter_dist_vert = inter_dist_vert
        self.inter_dist_horiz = inter_dist_horiz
        self.inter_dist = np.array([inter_dist_vert, inter_dist_horiz, inter_dist_horiz]).reshape(3, 1)
        self.max_depth = max_depth  # Maximum depth of particles.
        self._values = None  # Coordinates of the particles.

        # Boolean array denoting active particles.
        # These are particles 1) already started at the current time and
        # 2) are set to a positive state (Success/Evaluate).
        # Thus, this mask allows for particles do be deactivated without
        # needing to completely rebuild the tree.
        self._active_mask = None
        self.periodic_domain_zonal = periodic_domain_zonal

    @abstractmethod
    def find_neighbors_by_coor(self, coor):
        """Get the neighbors around a certain location.

        Parameters
        ----------
        coor :
            Numpy array with [depth, lat, lon].

        Returns
        -------
        type
            List of particle indices.

        """
        raise NotImplementedError

    def find_neighbors_by_idx(self, particle_idx):
        """Get the neighbors around a certain particle.

        Mainly useful for Structure of Array (SoA) datastructure

        Parameters
        ----------
        particle_idx :
            index of the particle (SoA).

        Returns
        -------
        type
            List of particle indices

        """
        coor = self._values[:, particle_idx].reshape(3, 1)
        return self.find_neighbors_by_coor(coor)

    def update_values(self, new_values, new_active_mask=None):
        """Update the coordinates of the particles.

        This is a default implementation simply rebuilds the structure.
        If the rebuilding is slow, a faster implementation can be provided.

        Parameters
        ----------
        new_values :
            numpy array ([depth, lat, lon], n_particles) with
            new coordinates of the particles.
        new_active_mask :
            boolean array indicating active particles. (Default value = None)
        """
        self.rebuild(new_values, new_active_mask)

    def rebuild(self, values, active_mask=-1):
        """Rebuild the neighbor structure from scratch.

        Parameters
        ----------
        values :
            numpy array with coordinates of particles
            (same as update).
        active_mask :
            boolean array indicating active particles. (Default value = -1)
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
        """Indices of the currently active mask."""
        # See __init__ comments for a more detailed explanation.
        if self._active_mask is None:
            return np.arange(self._values.shape[1])
        return np.where(self._active_mask)[0]

    @abstractmethod
    def _distance(self, coor, subset_idx):
        """Distance between a coordinate and particles

        Distance depends on the mesh (spherical/flat).

        Parameters
        ----------
        coor :
            Numpy array with 3D coordinates ([depth, lat, lon]).
        subset_idx :
            Indices of the particles to compute the distance to.

        Returns
        -------
        type
            horiz_dist: distance in the horizontal direction

        """
        raise NotImplementedError

    def _get_close_neighbor_dist(self, coor, subset_idx):
        """Compute distances and remove non-neighbors.

        Parameters
        ----------
        coor :
            Numpy array with 3D coordinates ([depth, lat, lon]).
        subset_idx :
            Indices of the particles to compute the distance to.

        Returns
        -------
        type
            neighbor_idx: Indices within the interaction distance.

        """
        vert_distance, horiz_distance = self._distance(coor, subset_idx)
        rel_distances = np.sqrt(
            (horiz_distance / self.inter_dist_horiz) ** 2 + (vert_distance / self.inter_dist_vert) ** 2
        )
        rel_neighbor_idx = np.where(rel_distances < 1)[0]
        neighbor_idx = subset_idx[rel_neighbor_idx]
        distances = np.vstack((vert_distance[rel_neighbor_idx], horiz_distance[rel_neighbor_idx]))
        return neighbor_idx, distances


class BaseFlatNeighborSearch(BaseNeighborSearch):
    """Base class for neighbor searches with a flat mesh."""

    def _distance(self, coor, subset_idx):
        coor = coor.reshape(3, 1)
        horiz_distance = np.sqrt(np.sum((self._values[1:, subset_idx] - coor[1:]) ** 2, axis=0))
        if self.periodic_domain_zonal:
            # If zonal periodic boundaries
            coor[2, 0] -= self.periodic_domain_zonal
            # distance through Western boundary
            hd2 = np.sqrt(np.sum((self._values[1:, subset_idx] - coor[1:]) ** 2, axis=0))
            coor[2, 0] += 2 * self.periodic_domain_zonal
            # distance through Eastern boundary
            hd3 = np.sqrt(np.sum((self._values[1:, subset_idx] - coor[1:]) ** 2, axis=0))
            coor[2, 0] -= self.periodic_domain_zonal
        else:
            hd2 = np.full(len(horiz_distance), np.inf)
            hd3 = np.full(len(horiz_distance), np.inf)

        horiz_distance = np.column_stack((horiz_distance, hd2, hd3))
        horiz_distance = np.min(horiz_distance, axis=1)
        vert_distance = np.abs(self._values[0, subset_idx] - coor[0])
        return (vert_distance, horiz_distance)


class BaseSphericalNeighborSearch(BaseNeighborSearch):
    """Base class for a neighbor search with a spherical mesh."""

    def _distance(self, coor, subset_idx):
        vert_distances, horiz_distances = spherical_distance(
            *coor,
            self._values[0, subset_idx],
            self._values[1, subset_idx],
            self._values[2, subset_idx],
        )

        if self.periodic_domain_zonal:
            # If zonal periodic boundaries
            coor[2, 0] -= self.periodic_domain_zonal
            # distance through Western boundary
            hd2 = spherical_distance(
                *coor, self._values[0, subset_idx], self._values[1, subset_idx], self._values[2, subset_idx]
            )[1]
            coor[2, 0] += 2 * self.periodic_domain_zonal
            # distance through Eastern boundary
            hd3 = spherical_distance(
                *coor, self._values[0, subset_idx], self._values[1, subset_idx], self._values[2, subset_idx]
            )[1]
            coor[2, 0] -= self.periodic_domain_zonal
        else:
            hd2 = np.full(len(horiz_distances), np.inf)
            hd3 = np.full(len(horiz_distances), np.inf)

        horiz_distances = np.column_stack((horiz_distances, hd2, hd3))
        horiz_distances = np.min(horiz_distances, axis=1)
        return (vert_distances, horiz_distances)

from math import ceil

import numpy as np

from parcels.interaction.neighborsearch.base import BaseSphericalNeighborSearch
from parcels.interaction.neighborsearch.basehash import (
    BaseHashNeighborSearch,
    hash_split,
)


class HashSphericalNeighborSearch(BaseHashNeighborSearch, BaseSphericalNeighborSearch):
    """Neighbor search using a hashtable (similar to octtrees).


    Parameters
    ----------
    inter_dist_vert : float
        Interaction distance (vertical) in m.
    inter_dist_horiz : float
        interaction distance (horizontal) in m
    max_depth : float, optional
        Maximum depth of the particles (default is 100000m).
    """

    def __init__(self, inter_dist_vert, inter_dist_horiz, max_depth=100000):
        super().__init__(inter_dist_vert, inter_dist_horiz, max_depth)

        self._init_structure()

    def _find_neighbors(self, hash_id, coor):
        """Get neighbors from hash_id and location."""
        # Get the neighboring cells.
        neighbor_blocks = geo_hash_to_neighbors(hash_id, coor, self._bits, self.inter_arc_dist)
        all_neighbor_points = []

        # Get the particles from the neighboring cells.
        for block in neighbor_blocks:
            try:
                all_neighbor_points.extend(self._hashtable[block])
            except KeyError:
                pass

        potential_neighbors = np.array(all_neighbor_points, dtype=int)
        return self._get_close_neighbor_dist(coor, potential_neighbors)

    def _values_to_hashes(self, values, active_idx=None):
        """Convert coordinates to cell ids.

        Parameters
        ----------
        values :
            array of positions of particles to convert
            ([depth, lat, lon], # of particles to convert).
        active_idx :
             (Default value = None)

        Returns
        -------
        type
            array of cell ids.

        """
        if active_idx is None:
            active_idx = np.arange(values.shape[1], dtype=int)
        depth = values[0, active_idx]
        lat = values[1, active_idx]
        lon = values[2, active_idx]

        # Southern or Northern hemisphere.
        lat_sign = (lat > 0).astype(int)

        # Find the latitude part of the cell id.
        i_depth = np.floor(depth / self.inter_dist_vert).astype(int)
        i_lat = np.floor(np.abs(lat) / self.inter_degree_dist).astype(int)

        # Get the arc length of the smaller circle around the earth.
        circ_small = 2 * np.pi * np.cos((i_lat + 1) * self.inter_arc_dist)
        n_lon = np.floor(circ_small / self.inter_arc_dist).astype(int)
        n_lon[n_lon < 1] = 1
        d_lon = 360 / n_lon

        # Get the longitude part of the cell id.
        i_lon = np.floor(lon / d_lon).astype(int)

        # Merge the 4 parts of the cell into one id.
        point_hash = i_3d_to_hash(i_depth, i_lat, i_lon, lat_sign, self._bits)
        point_array = np.empty(values.shape[1], dtype=int)
        point_array[active_idx] = point_hash
        return point_array

    def rebuild(self, values, active_mask=-1):
        """Recreate the tree with new values.

        Parameters
        ----------
        values :
            positions of the particles.
        active_mask :
             (Default value = -1)
        """
        super().rebuild(values, active_mask)
        active_idx = self.active_idx

        # Compute the hash values:
        self._particle_hashes = np.empty(self._values.shape[1], dtype=int)
        self._particle_hashes[active_idx] = self._values_to_hashes(values[:, active_idx])

        # Create the hashtable.
        self._hashtable = hash_split(self._particle_hashes, active_idx=active_idx)

        # Keep track of the position of a particle index within a cell.
        self._hash_idx = np.empty_like(self._particle_hashes, dtype=int)
        for idx_array in self._hashtable.values():
            self._hash_idx[idx_array] = np.arange(len(idx_array))

    def _init_structure(self):
        """Initialize the basic tree properties without building"""
        epsilon = 1e-12
        R_earth = 6371000

        self.inter_arc_dist = self.inter_dist_horiz / R_earth
        self.inter_degree_dist = 180 * self.inter_arc_dist / np.pi
        n_lines_depth = int(ceil(self.max_depth / self.inter_dist_vert + epsilon))
        n_lines_lat = int(ceil(np.pi / self.inter_arc_dist + epsilon))
        n_lines_lon = int(ceil(2 * np.pi / self.inter_arc_dist + epsilon))
        n_bits_lat = ceil(np.log(n_lines_lat) / np.log(2))
        n_bits_lon = ceil(np.log(n_lines_lon) / np.log(2))
        n_bits_depth = ceil(np.log(n_lines_depth) / np.log(2))
        self._bits = np.array([n_bits_depth, n_bits_lat, n_bits_lon])


def i_3d_to_hash(i_depth, i_lat, i_lon, lat_sign, bits):
    """Convert longitude and latitude id's to hash"""
    point_hash = lat_sign
    point_hash = np.bitwise_or(point_hash, np.left_shift(i_depth, 1))
    point_hash = np.bitwise_or(point_hash, np.left_shift(i_lat, 1 + bits[0]))
    point_hash = np.bitwise_or(point_hash, np.left_shift(i_lon, 1 + bits[0] + bits[1]))
    return point_hash


def geo_hash_to_neighbors(hash_id, coor, bits, inter_arc_dist):
    """Compute the hashes of all neighboring cells in a 3x3x3 neighborhood."""
    lat_sign = hash_id & 0x1
    i_depth = (hash_id >> 1) & ((1 << bits[0]) - 1)
    i_lat = (hash_id >> (1 + bits[0])) & ((1 << bits[1]) - 1)

    def all_neigh_depth(i_lat, i_lon, lat_sign):
        hashes = []
        for d_depth in [-1, 0, 1]:
            new_depth = i_depth + d_depth
            if new_depth < 0:
                continue
            hashes.append(i_3d_to_hash(new_depth, i_lat, i_lon, lat_sign, bits))
        return hashes

    neighbors = []
    # Loop over lower row, middle row, upper row
    for i_d_lat in [-1, 0, 1]:
        new_lat_sign = lat_sign
        new_i_lat = i_lat + i_d_lat
        if new_i_lat == -1:
            new_i_lat = 0
            new_lat_sign = 1 - lat_sign

        min_lat = new_i_lat + 1
        circ_small = 2 * np.pi * np.cos(min_lat * inter_arc_dist)
        n_new_lon = int(max(1, np.floor(circ_small / inter_arc_dist)))
        d_lon = 360 / n_new_lon
        if n_new_lon <= 3:
            for new_i_lon in range(n_new_lon):
                neighbors.extend(all_neigh_depth(new_i_lat, new_i_lon, new_lat_sign))
        else:
            start_i_lon = int(np.floor(coor[2][0] / d_lon))
            for delta_lon in [-1, 0, 1]:
                new_i_lon = (start_i_lon + delta_lon + n_new_lon) % n_new_lon
                neighbors.extend(all_neigh_depth(new_i_lat, new_i_lon, new_lat_sign))
    return neighbors

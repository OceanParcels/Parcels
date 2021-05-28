from math import ceil

# from numba.core.decorators import njit
import numpy as np

from parcels.interaction.base_neighbor import BaseSphericalNeighborSearch
from parcels.interaction.base_hash import BaseHashNeighborSearch, hash_split


class HashSphericalNeighborSearch(BaseHashNeighborSearch,
                                  BaseSphericalNeighborSearch):
    '''Neighbor search using a hashtable (similar to octtrees).'''
    name = "hash"

    def __init__(self, interaction_distance, interaction_depth,
                 max_depth=100000):
        '''Initialize the neighbor data structure.

        :param interaction_distance: maximum horizontal interaction distance.
        :param interaction_depth: maximum depth of interaction.
        :param values: lat, long, depth values for particles.
        :param max_depth: maximum depth of the ocean.
        '''
        super().__init__(interaction_distance, interaction_depth, max_depth)

        self._init_structure()

    def _find_neighbors(self, hash_id, coor):
        '''Get neighbors from hash_id and location.'''
        # Get the neighboring cells.
        neighbor_blocks = geo_hash_to_neighbors(
            hash_id, coor, self._bits, self.inter_arc_dist)
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
        '''Convert coordinates to cell ids.

        :param values: positions of particles to convert.
        :returns array of cell ids.
        '''
        if active_idx is None:
            active_idx = np.arange(values.shape[1], dtype=int)
        lat = values[0, active_idx]
        long = values[1, active_idx]
        depth = values[2, active_idx]

        # Southern or Nothern hemisphere.
        lat_sign = (lat > 0).astype(int)

        # Find the lattitude part of the cell id.
        i_lat = np.floor(np.abs(lat)/self.inter_degree_dist).astype(int)
        i_depth = np.floor(depth/self.interaction_depth).astype(int)

        # Get the arc length of the smaller circle around the earth.
        circ_small = 2*np.pi*np.cos((i_lat+1)*self.inter_arc_dist)
        n_long = np.floor(circ_small/self.inter_arc_dist).astype(int)
        n_long[n_long < 1] = 1
        d_long = 360/n_long

        # Get the longitude part of the cell id.
        i_long = np.floor(long/d_long).astype(int)

        # Merge the 4 parts of the cell into one id.
        point_hash = i_3d_to_hash(i_lat, i_long, i_depth, lat_sign, self._bits)
        point_array = np.empty(values.shape[1], dtype=int)
        point_array[active_idx] = point_hash
        print(values.shape[1])
        return point_array

    def rebuild(self, values, active_mask=-1):
        '''Recreate the tree with new values.

        :param values: positions of the particles.
        '''
        super().rebuild(values, active_mask)
        active_idx = self.active_idx

        # Compute the hash values:
        self._particle_hashes = np.empty(self._values.shape[1], dtype=int)
        self._particle_hashes[active_idx] = self._values_to_hashes(
            values[:, active_idx])

        # Create the hashtable.
        self._hashtable = hash_split(self._particle_hashes,
                                     active_idx=active_idx)

        # Keep track of the position of a particle index within a cell.
        self._hash_idx = np.empty_like(self._particle_hashes, dtype=int)
        for idx_array in self._hashtable.values():
            self._hash_idx[idx_array] = np.arange(len(idx_array))

    def _init_structure(self):
        '''Initialize the basic tree properties without building'''
        epsilon = 1e-12
        R_earth = 6371000

        self.inter_arc_dist = self.interaction_distance/R_earth
        self.inter_degree_dist = 180*self.inter_arc_dist/np.pi
        n_lines_lat = int(ceil(np.pi/self.inter_arc_dist+epsilon))
        n_lines_long = int(ceil(2*np.pi/self.inter_arc_dist+epsilon))
        n_lines_depth = int(ceil(
            self.max_depth/self.interaction_depth + epsilon))
        n_bits_lat = ceil(np.log(n_lines_lat)/np.log(2))
        n_bits_long = ceil(np.log(n_lines_long)/np.log(2))
        n_bits_depth = ceil(np.log(n_lines_depth)/np.log(2))
        self._bits = np.array([n_bits_lat, n_bits_long, n_bits_depth])


# @njit
def i_3d_to_hash(i_lat, i_long, i_depth, lat_sign, bits):
    '''Convert longitude and lattitude id's to hash'''
    point_hash = lat_sign
    point_hash = np.bitwise_or(point_hash, np.left_shift(i_lat, 1))
    point_hash = np.bitwise_or(point_hash, np.left_shift(i_long, 1+bits[0]))
    point_hash = np.bitwise_or(point_hash,
                               np.left_shift(i_depth, 1+bits[0]+bits[1]))
    return point_hash


def geo_hash_to_neighbors(hash_id, coor, bits, inter_arc_dist):
    '''Compute the hashes of all neighboring cells.'''
    lat_sign = hash_id & 0x1
    i_lat = (hash_id >> 1) & ((1 << bits[0])-1)
    i_depth = (hash_id >> (1+bits[0]+bits[1])) & ((1 << bits[2])-1)

    def all_neigh_depth(i_lat, i_long, lat_sign):
        hashes = []
        for d_depth in [-1, 0, 1]:
            new_depth = i_depth + d_depth
            if new_depth < 0:
                continue
            hashes.append(
                i_3d_to_hash(i_lat, i_long, new_depth, lat_sign, bits))
        return hashes
    neighbors = []
    # Loop over lower row, middle row, upper row
    for i_d_lat in [-1, 0, 1]:
        new_lat_sign = lat_sign
        new_i_lat = i_lat + i_d_lat
        if new_i_lat == -1:
            new_i_lat = 0
            new_lat_sign = (1-lat_sign)

        min_lat = new_i_lat + 1
        circ_small = 2*np.pi*np.cos(min_lat*inter_arc_dist)
        n_new_long = int(max(1, np.floor(circ_small/inter_arc_dist)))
        d_long = 360/n_new_long
        if n_new_long <= 3:
            for new_i_long in range(n_new_long):
                neighbors.extend(
                    all_neigh_depth(new_i_lat, new_i_long, new_lat_sign))
        else:
            start_i_long = int(np.floor(coor[1]/d_long))
            for delta_long in [-1, 0, 1]:
                new_i_long = (start_i_long+delta_long+n_new_long) % n_new_long
                neighbors.extend(
                    all_neigh_depth(new_i_lat, new_i_long, new_lat_sign))
    return neighbors

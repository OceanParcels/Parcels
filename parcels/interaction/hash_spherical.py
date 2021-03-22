from math import ceil
from timeit import timeit

from numba.core.decorators import njit
import numpy as np

from parcels.interaction.spherical_utils import relative_3d_distance
from parcels.interaction.base_neighbor import BaseSphericalNeighborSearch


class HashSphericalNeighborSearch(BaseSphericalNeighborSearch):
    '''Neighbor search using a hashtable (similar to octtrees).'''
    name = "hash"

    def __init__(self, interaction_distance, interaction_depth, values=None,
                 max_depth=100000):
        '''Initialize the neighbor data structure.

        :param interaction_distance: maximum horizontal distance of interaction.
        :param interaction_depth: maximum depth of interaction.
        :param values: lat, long, depth values for particles.
        :param max_depth: maximum depth of the ocean.
        '''
        super().__init__(interaction_distance, interaction_depth, values,
                         max_depth)

        self.init_structure()
        if values is not None:
            self.rebuild(values)

    def find_neighbors_by_coor(self, coor):
        '''Get the neighbors around a certain location.

        :param coor: Numpy array with (lat, long, depth).
        :returns List of particle indices.
        '''
        hash_id = self.values_to_hashes(coor.reshape((3, 1)))
        return self._find_neighbors(hash_id, coor)

    def find_neighbors_by_idx(self, particle_idx):
        '''Get the neighbors around a certain particle

        :param particle_idx: index of the particle during tree building.
        :returns List of particle indices
        '''
        hash_id = self._particle_hashes[particle_idx]
        coor = self._values[:, particle_idx]
        return self._find_neighbors(hash_id, coor)

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

        # Loop over neighbor candidates and check their actual distance.
        # TODO: use vectorized computation.
        true_neigh = []
        for neigh in all_neighbor_points:
            distance = relative_3d_distance(
                *self._values[:, neigh], *coor,
                interaction_distance=self.interaction_distance,
                interaction_depth=self.interaction_depth)
            if distance < 1:
                true_neigh.append(neigh)
        return np.array(true_neigh)

    def update_values(self, new_values):
        '''Update the locations of (some) of the particles.

        Particles that stay in the same location are computationally cheap.
        The order and number of the particles is assumed to remain the same.

        :param new_values: new (lat, long, depth) values for particles.
        '''
        old_hashes = self._particle_hashes
        new_hashes = self.values_to_hashes(new_values)

        # See which particles have crossed cell boundaries.
        move_idx = np.where(old_hashes != new_hashes)[0]
        if len(move_idx) == 0:
            return
        remove_split = hash_split(old_hashes[move_idx])
        add_split = hash_split(new_hashes[move_idx])

        # Remove particles from the hash table.
        for cur_hash, remove_idx in remove_split.items():
            if len(remove_idx) == len(self._hashtable[cur_hash]):
                del self._hashtable[cur_hash]
            else:
                rel_remove_idx = self._hash_idx[move_idx[remove_idx]]
                self._hashtable[cur_hash] = np.delete(
                    self._hashtable[cur_hash], rel_remove_idx)
                self._hash_idx[self._hashtable[cur_hash]] = np.arange(
                    len(self._hashtable[cur_hash]))

        # Add particles to the hash table.
        for cur_hash, add_idx in add_split.items():
            if cur_hash not in self._hashtable:
                self._hashtable[cur_hash] = move_idx[add_idx]
                self._hash_idx[move_idx[add_idx]] = np.arange(len(add_idx))
            else:
                self._hash_idx[move_idx[add_idx]] = np.arange(
                    len(self._hashtable[cur_hash]),
                    len(self._hashtable[cur_hash]) + len(add_idx))
                self._hashtable[cur_hash] = np.append(
                    self._hashtable[cur_hash], move_idx[add_idx])
        self._values = new_values
        self._particle_hashes = new_hashes

    def values_to_hashes(self, values):
        '''Convert coordinates to cell ids.

        :param values: positions of particles to convert.
        :returns array of cell ids.
        '''
        lat = values[0, :]
        long = values[1, :]
        depth = values[2, :]

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
        return point_hash

    def rebuild(self, values=None):
        '''Recreate the tree with new values.

        :param values: positions of the particles.
        '''
        if values is None:
            values = self._values
        self._values = values

        # Compute the hash values
        self._particle_hashes = self.values_to_hashes(self._values)

        # Create the hashtable.
        self._hashtable = hash_split(self._particle_hashes)

        # Keep track of the position of a particle index within a cell.
        self._hash_idx = np.empty_like(self._particle_hashes, dtype=int)
        for idx_array in self._hashtable.values():
            self._hash_idx[idx_array] = np.arange(len(idx_array))

    def init_structure(self):
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

    def consistency_check(self):
        '''See if all values are in their proper place.'''
        for idx in range(self._values.shape[1]):
            cur_hash = self._particle_hashes[idx]
            hash_idx = self._hash_idx[idx]
            if self._hashtable[cur_hash][hash_idx] != idx:
                print(cur_hash, hash_idx)
                print(self._hashtable[cur_hash])
            assert self._hashtable[cur_hash][hash_idx] == idx

        n_idx = 0
        for idx_array in self._hashtable.values():
            n_idx += len(idx_array)
        assert n_idx == self._values.shape[1]
        assert np.all(self.values_to_hashes(self._values) == self._particle_hashes)

    @classmethod
    def benchmark(cls, max_n_particles=1000, density=1, interaction_depth=100,
                  update_frac=0.01):
        '''Perform benchmarks to figure out scaling with particles.'''
        np.random.seed(213874)

        def bench_init(values, *args, **kwargs):
            return cls(values, *args, **kwargs)

        def bench_search(neigh_search, n_sample):
            for particle_id in np.random.randint(neigh_search._values.shape[1],
                                                 size=n_sample):
                neigh_search.find_neighbors(particle_id)

        def bench_update(neigh_search, n_change):
            move_values = neigh_search.create_positions(n_change)
            new_values = neigh_search._values.copy()
            move_index = np.random.choice(n_particles, size=n_change, replace=False)
            new_values[:, move_index] = move_values
            neigh_search.update_values(new_values)

        all_dt_init = []
        all_dt_search = []
        all_n_particles = []
        all_dt_update = []
        all_max_dist = []
        n_particles = 30
        n_init = 100
        while n_particles < max_n_particles:
            n_update = int(n_particles*update_frac)
            inter_dist = (density*cls.area*cls.max_depth /
                          (n_particles*interaction_depth))**(1/3)
            kwargs = {"interaction_distance": inter_dist,
                      "interaction_depth": interaction_depth}
            n_sample = min(5000, 10*n_particles)
            n_sample_update = int(n_sample/10)
            if n_particles > 5000:
                n_init = 10
            positions = cls.create_positions(n_particles)
            dt_init = timeit(lambda: bench_init(positions, **kwargs),
                             number=n_init)/n_init
            neigh_search = bench_init(positions, **kwargs)
            dt_search = timeit(lambda: bench_search(neigh_search, n_sample),
                               number=1)/n_sample
            dt_update = timeit(lambda: bench_update(neigh_search, n_update),
                               number=n_sample_update)/n_sample_update
            all_dt_init.append(dt_init)
            all_dt_search.append(dt_search)
            all_n_particles.append(n_particles)
            all_dt_update.append(dt_update)
            all_max_dist.append(inter_dist)
            n_particles *= 2
        return {
            "name": cls.name,
            "n_particles": np.array(all_n_particles),
            "init_time": np.array(all_dt_init),
            "search_time": np.array(all_dt_search),
            "update_time": np.array(all_dt_update),
            "max_dist": np.array(all_max_dist),
        }


def hash_split(hash_ids):
    '''Create a hashtable.

    Multiple particles that are found in the same cell are put in a list
    with that particular hash.
    '''
    sort_idx = np.argsort(hash_ids)
    a_sorted = hash_ids[sort_idx]
    unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    unq_items = a_sorted[unq_first]
    unq_count = np.diff(np.nonzero(unq_first)[0])
    unq_idx = np.split(sort_idx, np.cumsum(unq_count))
    return dict(zip(unq_items, unq_idx))


@njit
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
                neighbors.extend(all_neigh_depth(new_i_lat, new_i_long, new_lat_sign))
        else:
            start_i_long = int(np.floor(coor[1]/d_long))
            for delta_long in [-1, 0, 1]:
                new_i_long = (start_i_long+delta_long+n_new_long) % n_new_long
                neighbors.extend(all_neigh_depth(new_i_lat, new_i_long, new_lat_sign))
    return neighbors

from abc import ABC, abstractmethod

import numpy as np
from timeit import timeit


class BaseHashNeighborSearch(ABC):
    def deactivate_particles(self, particle_idx):
        remove_split = hash_split(self._particle_hashes[particle_idx])
        # Remove particles from the hash table.
        for cur_hash, remove_idx in remove_split.items():
            if len(remove_idx) == len(self._hashtable[cur_hash]):
                del self._hashtable[cur_hash]
            else:
                rel_remove_idx = self._hash_idx[particle_idx[remove_idx]]
                self._hashtable[cur_hash] = np.delete(
                    self._hashtable[cur_hash], rel_remove_idx)
                self._hash_idx[self._hashtable[cur_hash]] = np.arange(
                    len(self._hashtable[cur_hash]))

    def find_neighbors_by_coor(self, coor):
        '''Get the neighbors around a certain location.

        :param coor: Numpy array with (lat, long, depth).
        :returns List of particle indices.
        '''
        coor = coor.reshape(3, 1)
        hash_id = self.values_to_hashes(coor)
        return self._find_neighbors(hash_id, coor)

    def find_neighbors_by_idx(self, particle_idx):
        '''Get the neighbors around a certain particle

        :param particle_idx: index of the particle during tree building.
        :returns List of particle indices
        '''
        hash_id = self._particle_hashes[particle_idx]
        coor = self._values[:, particle_idx].reshape(3, 1)
        return self._find_neighbors(hash_id, coor)

    @abstractmethod
    def _find_neighbors(self, hash_id, coor):
        raise NotImplementedError

    def activate_particles(self, particle_idx):
        add_split = hash_split(self._particle_hashes[particle_idx])

        # Add particles to the hash table.
        for cur_hash, add_idx in add_split.items():
            if cur_hash not in self._hashtable:
                self._hashtable[cur_hash] = particle_idx[add_idx]
                self._hash_idx[particle_idx[add_idx]] = np.arange(len(add_idx))
            else:
                self._hash_idx[particle_idx[add_idx]] = np.arange(
                    len(self._hashtable[cur_hash]),
                    len(self._hashtable[cur_hash]) + len(add_idx))
                self._hashtable[cur_hash] = np.append(
                    self._hashtable[cur_hash], particle_idx[add_idx])

    def consistency_check(self):
        '''See if all values are in their proper place.'''
        active_idx = self.active_idx
        if active_idx is None:
            active_idx = np.arange(self._values.shape[1])

        for idx in active_idx:
            cur_hash = self._particle_hashes[idx]
            hash_idx = self._hash_idx[idx]
            if self._hashtable[cur_hash][hash_idx] != idx:
                print(cur_hash, hash_idx)
                print(self._hashtable[cur_hash])
            assert self._hashtable[cur_hash][hash_idx] == idx

        n_idx = 0
        for idx_array in self._hashtable.values():
            n_idx += len(idx_array)
        assert n_idx == len(active_idx)
        assert np.all(self.values_to_hashes(self._values[:, active_idx]) == self._particle_hashes[active_idx])

    def update_values(self, new_values, new_active_mask=None):
        '''Update the locations of (some) of the particles.

        Particles that stay in the same location are computationally cheap.
        The order and number of the particles is assumed to remain the same.

        :param new_values: new (lat, long, depth) values for particles.
        '''
        if self._values is None:
            self.rebuild(new_values, new_active_mask)
            return

        if new_active_mask is None:
            new_active_mask = np.full(new_values.shape[1], True)

        deactivated_mask = np.logical_and(self._active_mask, np.logical_not(new_active_mask))
        stay_active_mask = np.logical_and(self._active_mask, new_active_mask)
        activated_mask = np.logical_and(np.logical_not(self._active_mask), new_active_mask)

        stay_active_idx = np.where(stay_active_mask)[0]

        old_hashes = self._particle_hashes[stay_active_mask]
        new_hashes = self.values_to_hashes(new_values[:, stay_active_mask])

        # See which particles have crossed cell boundaries.
        move_idx = stay_active_idx[np.where(old_hashes != new_hashes)[0]]
        remove_idx = np.append(move_idx, np.where(deactivated_mask)[0])
        add_idx = np.append(move_idx, np.where(activated_mask)[0])

        self.deactivate_particles(remove_idx)
        self._particle_hashes[stay_active_mask] = new_hashes
        self._particle_hashes[activated_mask] = self.values_to_hashes(
            new_values[:, activated_mask])
        self.activate_particles(add_idx)

        self._active_mask = new_active_mask
        self._values = new_values

    @abstractmethod
    def values_to_hashes(self, values):
        raise NotImplementedError

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


def hash_split(hash_ids, active_idx=None):
    '''Create a hashtable.

    Multiple particles that are found in the same cell are put in a list
    with that particular hash.
    '''
    if len(hash_ids) == 0:
        return {}
    if active_idx is not None:
        sort_idx = active_idx[np.argsort(hash_ids[active_idx])]
    else:
        sort_idx = np.argsort(hash_ids)

    a_sorted = hash_ids[sort_idx]
    unq_first = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    unq_items = a_sorted[unq_first]
    unq_count = np.diff(np.nonzero(unq_first)[0])
    unq_idx = np.split(sort_idx, np.cumsum(unq_count))
    return dict(zip(unq_items, unq_idx))

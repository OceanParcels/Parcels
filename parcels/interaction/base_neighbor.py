from abc import ABC, abstractmethod

import numpy as np
from timeit import timeit


class BaseNeighborSearch(ABC):
    name = "unknown"

    def __init__(self, interaction_distance, values=None):
        self._values = values
        self.interaction_distance = interaction_distance

    @abstractmethod
    def find_neighbors_by_idx(self, particle_id):
        '''Find neighbors with particle_id.'''
        pass

    def update_values(self, new_values):
        self._values = new_values

    @classmethod
    def benchmark(cls, max_n_particles=1000, density=1):
        '''Perform benchmarks to figure out scaling with particles.'''
        np.random.seed(213874)

        def bench_init(values, interaction_distance):
            return cls(values, max_dist)

        def bench_search(neigh_search, n_sample):
            for particle_id in np.random.randint(neigh_search._values.shape[1],
                                                 size=n_sample):
                neigh_search.find_neighbors(particle_id)

        all_dt_init = []
        all_dt_search = []
        all_n_particles = []
        n_particles = 30
        n_init = 100
        while n_particles < max_n_particles:
            max_dist = np.sqrt(density*cls.area/(n_particles))
            n_sample = min(5000, 10*n_particles)
            if n_particles > 5000:
                n_init = 10
            positions = cls.create_positions(n_particles)
            dt_init = timeit(lambda: bench_init(positions, max_dist),
                             number=n_init)/n_init
            neigh_search = bench_init(positions, max_dist)
            dt_search = timeit(lambda: bench_search(neigh_search, n_sample),
                               number=1)/n_sample
            all_dt_init.append(dt_init)
            all_dt_search.append(dt_search)
            all_n_particles.append(n_particles)
            n_particles *= 2
        return {
            "name": cls.name,
            "n_particles": np.array(all_n_particles),
            "init_time": np.array(all_dt_init),
            "search_time": np.array(all_dt_search)}

    @classmethod
    @abstractmethod
    def create_positions(cls, n_particles):
        pass


class BaseNeighborSearchGeo3D(BaseNeighborSearch):
    # TODO: remove/change max_depth stuff. Should only affect benchmarks.
    def __init__(self, interaction_distance, interaction_depth, values=None):
        self.interaction_depth = interaction_depth
        super(BaseNeighborSearchGeo3D, self).__init__(interaction_distance, values=values)

    @classmethod
    def create_positions(cls, n_particles):
        yrange = 2*np.random.rand(n_particles)
        lat = (np.arccos(1-yrange)-0.5*np.pi)/np.pi
        long = 360*np.random.rand(n_particles)
        depth = cls.max_depth*np.random.rand(n_particles)
        return np.array((lat, long, depth))

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


class BaseNeighborSearchGeo(BaseNeighborSearch):
    area = 4*np.pi

    @classmethod
    def create_positions(cls, n_particles):
        yrange = 2*np.random.rand(n_particles)
        lat = np.arccos(1-yrange)-0.5*np.pi
        long = 2*np.pi*np.random.rand(n_particles)
        return np.array((lat, long))


class BaseNeighborSearchCart(BaseNeighborSearch):
    area = 1

    @classmethod
    def create_positions(cls, n_particle):
        return np.random.rand(n_particle*2).reshape(2, -1)

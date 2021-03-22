from parcels.interaction.scipy_flat import ScipyFlatNeighborSearch
from parcels.interaction.brute_force import BruteFlatNeighborSearch
from parcels.interaction.hash_flat import HashFlatNeighborSearch
import numpy as np


def test_flat_neighbors():
    np.random.seed(129873)
    neighbor_classes = [
        ScipyFlatNeighborSearch, BruteFlatNeighborSearch, HashFlatNeighborSearch
    ]

    instances = []
    positions = ScipyFlatNeighborSearch.create_positions(1000)
    for cur_class in neighbor_classes:
        cur_instance = cur_class(interaction_distance=0.3, interaction_depth=0.3)
        cur_instance.rebuild(positions)
        instances.append(cur_instance)

    for particle_idx in np.random.choice(positions.shape[1], 100, replace=False):
        res = {}
        for instance in instances:
            cur_neigh = instance.find_neighbors_by_idx(particle_idx)
            assert instance.name != "unknown"
            res[instance.name] = cur_neigh

        assert len(res) == len(instances)
        instance_zero = instances[0]
        result_zero = res[instance_zero.name]
        assert len(result_zero) == len(set(result_zero))
        assert isinstance(result_zero, np.ndarray), f"type: {type(result_zero)}"
        for instance in instances[1:]:
            cur_result = res[instance.name]
            assert isinstance(cur_result, np.ndarray), f"Failed on {instance.name}"
            assert set(result_zero) == set(cur_result), f"Failed on {instance.name}"
            assert len(cur_result) == len(set(cur_result))

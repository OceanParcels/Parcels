import numpy as np
from parcels.interaction.base_neighbor import BaseFlatNeighborSearch
from numba import njit
import numba as nb
from parcels.interaction.base_hash import BaseHashNeighborSearch, hash_split


class HashFlatNeighborSearch(BaseHashNeighborSearch, BaseFlatNeighborSearch):
    '''Neighbor search using a hashtable (similar to octtrees).'''
    name = "hash"

    def _find_neighbors(self, hash_id, coor):
        neighbor_blocks = hash_to_neighbors(hash_id, self._bits)
        all_neighbor_points = []
        for block in neighbor_blocks:
            try:
                all_neighbor_points.extend(self._hash_table[block])
            except KeyError:
                pass

        pot_neighbors = np.array(all_neighbor_points)
        distances = np.linalg.norm((self._values[:, pot_neighbors]-coor)/self.inter_dist, axis=0)
        neighbors = pot_neighbors[np.where(distances < 1)]
        return neighbors

    def rebuild(self, values, active_mask=-1):
        super().rebuild(values, active_mask)
        active_values = self._values[self._active_mask]

        self._box = np.array([[active_values[i, :].min(), active_values[i, :].max()]
                              for i in range(active_values.shape[0])])

        epsilon = 1e-8

        n_bits = ((self._box[:, 1] - self._box[:, 0])/self.inter_dist.reshape(-1) + epsilon)/np.log(2)
        self._bits = np.ceil(n_bits).astype(int)
        self._min_box = self._box[:, 0]
        self._min_box = self._min_box.reshape(-1, 1)
        particle_hashes = self.values_to_hashes(values, self.active_idx)
        self._hash_table = hash_split(particle_hashes)
        self._particle_hashes = particle_hashes

    def values_to_hashes(self, values, active_idx=None):
        if active_idx is None:
            active_values = values
        else:
            active_values = values[active_idx]
        box_i = ((active_values-self._min_box)/self.inter_dist).astype(int)
        particle_hashes = np.bitwise_or(
            box_i[0, :], np.left_shift(box_i[1, :], self._bits[0]))

        if active_values is None:
            return particle_hashes

        all_hashes = np.empty(values.shape[1], dtype=int)
        all_hashes[active_idx] = particle_hashes
        return all_hashes


@njit
def hash_to_neighbors(hash_id, bits):
#   coor = np.zeros((len(bits),), dtype=np.int32)
#   new_coor = np.zeros((len(bits),), dtype=np.int32)
    coor = np.zeros((len(bits),), dtype=nb.int32)
    new_coor = np.zeros((len(bits),), dtype=nb.int32)
    tot_bits = 0
    for dim in range(len(bits)):
        coor[dim] = (hash_id >> tot_bits) & ((1 << bits[dim])-1)
        tot_bits += bits[dim]

    coor_max = np.left_shift(1, bits)

    neighbors = []

    for offset in range(pow(3, len(bits))):
        divider = 1
        for dim in range(len(bits)):
            new_coor[dim] = coor[dim] + (1-((offset//divider) % 3))
            divider *= 3
        if np.any(new_coor > coor_max) or np.any(new_coor < 0):
            continue
        new_hash = 0
        tot_bits = 0
        for dim in range(len(bits)):
            new_hash |= (new_coor[dim] << tot_bits)
            tot_bits += bits[dim]
        neighbors.append(new_hash)
    return neighbors

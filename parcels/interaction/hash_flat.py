from math import ceil

import numpy as np
from parcels.interaction.base_neighbor import BaseFlatNeighborSearch
from numba import njit
import numba as nb


class HashFlatNeighborSearch(BaseFlatNeighborSearch):
    '''Neighbor search using a hashtable (similar to octtrees).'''
    name = "hash"

    def __init__(self, interaction_distance, interaction_depth,
                 values=None):
        super().__init__(interaction_distance, interaction_depth, values)
#         self.inter_dist = np.array(
#             [self.interaction_distance, self.interaction_distance,
#              self.interaction_depth])
        if values is not None:
            self.rebuild(values)
#         self._box = [[self._values[i, :].min(), self._values[i, :].max()]
#                      for i in range(self._values.shape[0])]
#         self.build_tree()

    def find_neighbors_by_coor(self, coor):
        hash_id = self.values_to_hashes(coor.reshape((3, 1)))
        return self._find_neighbors(hash_id, coor)

    def find_neighbors_by_idx(self, particle_idx):
        hash_id = self._particle_hashes[particle_idx]
        coor = self._values[:, particle_idx].reshape(3, 1)
        return self._find_neighbors(hash_id, coor)

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

    def rebuild(self, values=None):
        if values is None:
            values = self._values
        self._values = values
        self._box = np.array([[self._values[i, :].min(), self._values[i, :].max()]
                              for i in range(self._values.shape[0])])

        epsilon = 1e-8

        n_bits = ((self._box[:, 1] - self._box[:, 0])/self.inter_dist.reshape(-1) + epsilon)/np.log(2)
        bits = np.ceil(n_bits).astype(int)
        self._min_box = self._box[:, 0]
        self._min_box = self._min_box.reshape(-1, 1)
        box_i = ((values-self._min_box)/self.inter_dist).astype(int)
        print(bits.shape)
        particle_hashes = np.bitwise_or(
            box_i[0, :], np.left_shift(box_i[1, :], bits[0]))
        self._hash_table = hash_split(particle_hashes)
        self._particle_hashes = particle_hashes
        self._bits = bits

    def values_to_hashes(self, values):
        box_i = ((values-self._min_box)/self.inter_dist).astype(int)
        particle_hashes = np.bitwise_or(
            box_i[0, :], np.left_shift(box_i[1, :], self._bits[0]))
        return particle_hashes


# def build_tree(values, max_dist, box):
#     bits = []
#     min_box = []
#     for interval in box:
#         epsilon = 1e-8
#         n_bits = np.log((interval[1] - interval[0])/max_dist+epsilon)/np.log(2)
#         bits.append(ceil(n_bits))
#         min_box.append(interval[0])
# 
#     #min_box = np.array(min_box)
#     #particle_hashes = np.empty(values.shape[1], dtype=int)
#     #for particle_id in range(values.shape[1]):
#     #    box_f = (values[:, particle_id] - min_box)/max_dist
#     #    particle_hashes[particle_id] = np.bitwise_or(
#     #        int(box_f[0]), np.left_shift(int(box_f[1]), bits[0]))
#     min_box = np.array(min_box).reshape(-1, 1)
#     box_i = ((values-min_box)/max_dist).astype(int)
#     particle_hashes = np.bitwise_or(box_i[0, :],
#                                     np.left_shift(box_i[1, :],
#                                                   bits[0]))
#     oct_dict = hash_split(particle_hashes)
#     return oct_dict, particle_hashes, np.array(bits, dtype=int)


def hash_split(hash_ids):
    sort_idx = np.argsort(hash_ids)
    a_sorted = hash_ids[sort_idx]
    unq_first = np.concatenate((np.array([True]), a_sorted[1:] != a_sorted[:-1]))
    unq_items = a_sorted[unq_first]
    unq_count = np.diff(np.nonzero(unq_first)[0])
    unq_idx = np.split(sort_idx, np.cumsum(unq_count))
    return dict(zip(unq_items, unq_idx))


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

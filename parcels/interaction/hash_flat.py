import numpy as np
from parcels.interaction.base_neighbor import BaseFlatNeighborSearch
# from numba import njit
# import numba as nb
from parcels.interaction.base_hash import BaseHashNeighborSearch, hash_split


class HashFlatNeighborSearch(BaseHashNeighborSearch, BaseFlatNeighborSearch):
    '''Neighbor search using a hashtable (similar to octtrees).'''
    name = "hash"
    _box = None

    def _find_neighbors(self, hash_id, coor):
        neighbor_blocks = hash_to_neighbors(hash_id, self._bits)
        all_neighbor_points = []
        for block in neighbor_blocks:
            try:
                all_neighbor_points.extend(self._hashtable[block])
            except KeyError:
                pass

        pot_neighbors = np.array(all_neighbor_points)
        return self._get_close_neighbor_dist(coor, pot_neighbors)

    def update_values(self, new_values, new_active_mask=None):
        if not self.check_box(new_values, new_active_mask):
            self.rebuild(new_values, new_active_mask)
        else:
            super().update_values(new_values, new_active_mask=new_active_mask)

    def check_box(self, new_values, new_active_mask):
        if self._box is None:
            return False
        if new_active_mask is None:
            active_values = new_values
        else:
            active_values = new_values[:, new_active_mask]
        for i_dim in range(3):
            if np.any(active_values[i_dim, :] - self._box[i_dim][0] < 0):
                return False
            if np.any(active_values[i_dim, :] - self._box[i_dim][1] > 0):
                return False
        return True

    def rebuild(self, values, active_mask=-1):
        super().rebuild(values, active_mask)
        active_values = self._values[:, self._active_mask]

        self._box = []
        for i_dim in range(3):
            val_min = active_values[i_dim, :].min()
            val_max = active_values[i_dim, :].max()
            margin = (val_max-val_min)*0.3
            self._box.append([val_min-margin, val_max+margin])

        self._box = np.array(self._box)

        epsilon = 1e-8

        n_bits = ((self._box[:, 1] - self._box[:, 0]
                   )/self.inter_dist.reshape(-1) + epsilon)/np.log(2)
        self._bits = np.ceil(n_bits).astype(int)
        self._min_box = self._box[:, 0]
        self._min_box = self._min_box.reshape(-1, 1)
        particle_hashes = self.values_to_hashes(values, self.active_idx)
        self._hashtable = hash_split(particle_hashes,
                                     active_idx=self.active_idx)
        self._particle_hashes = particle_hashes

        # Keep track of the position of a particle index within a cell.
        self._hash_idx = np.empty_like(self._particle_hashes, dtype=int)
        for idx_array in self._hashtable.values():
            self._hash_idx[idx_array] = np.arange(len(idx_array))

    def values_to_hashes(self, values, active_idx=None):
        if active_idx is None:
            active_values = values
        else:
            active_values = values[:, active_idx]
        box_i = ((active_values-self._min_box)/self.inter_dist).astype(int)
        particle_hashes = np.bitwise_or(
            box_i[0, :], np.left_shift(box_i[1, :], self._bits[0]))

        if active_values is None:
            return particle_hashes

        all_hashes = np.empty(values.shape[1], dtype=int)
        all_hashes[active_idx] = particle_hashes
        return all_hashes


# @njit
def hash_to_neighbors(hash_id, bits):
    coor = np.zeros((len(bits),), dtype=np.int32)
    new_coor = np.zeros((len(bits),), dtype=np.int32)
#     coor = np.zeros((len(bits),), dtype=nb.int32)
#     new_coor = np.zeros((len(bits),), dtype=nb.int32)
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

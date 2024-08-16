import numpy as np

from parcels.interaction.neighborsearch.base import BaseFlatNeighborSearch
from parcels.interaction.neighborsearch.basehash import (
    BaseHashNeighborSearch,
    hash_split,
)


class HashFlatNeighborSearch(BaseHashNeighborSearch, BaseFlatNeighborSearch):
    """Neighbor search using a hashtable (similar to octtrees)."""

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
        if not self._check_box(new_values, new_active_mask):
            self.rebuild(new_values, new_active_mask)
        else:
            super().update_values(new_values, new_active_mask=new_active_mask)

    def _check_box(self, new_values, new_active_mask):
        """Check whether particles have moved out of the overall box.

        Parameters
        ----------
        new_values :
            New particle coordinates (depth, lat, lon) to be checked.
        new_active_mask :
            New active mask for the particles.

        Returns
        -------
        type
            True if box is still big enough, False if not.

        """
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

        # Compute the dimensions of the box with a margin.
        self._box = []
        for i_dim in range(3):
            val_min = active_values[i_dim, :].min()
            val_max = active_values[i_dim, :].max()
            margin = (val_max - val_min) * 0.3
            self._box.append([val_min - margin, val_max + margin])

        self._box = np.array(self._box)

        epsilon = 1e-8

        # Compute the number of bits in each of the three dimensions
        # E.g. if we have 3 bits (depth), we must have less than 2^3 cells in
        # that direction.
        n_bits = ((self._box[:, 1] - self._box[:, 0]) / self.inter_dist.reshape(-1) + epsilon) / np.log(2)
        self._bits = np.ceil(n_bits).astype(int)

        # Compute the starting point of the cell (0, 0, 0).
        self._min_box = self._box[:, 0]
        self._min_box = self._min_box.reshape(-1, 1)

        # Compute the hash table.
        particle_hashes = self._values_to_hashes(values, self.active_idx)
        self._hashtable = hash_split(particle_hashes, active_idx=self.active_idx)
        self._particle_hashes = particle_hashes

        # Keep track of the position of a particle index within a cell.
        self._hash_idx = np.empty_like(self._particle_hashes, dtype=int)
        for idx_array in self._hashtable.values():
            self._hash_idx[idx_array] = np.arange(len(idx_array))

    def _values_to_hashes(self, values, active_idx=None):
        if active_idx is None:
            active_values = values
        else:
            active_values = values[:, active_idx]

        # Compute the box_id/hashes.
        box_i = ((active_values - self._min_box) / self.inter_dist).astype(int)
        particle_hashes = np.bitwise_or(box_i[0, :], np.left_shift(box_i[1, :], self._bits[0]))

        if active_values is None:
            return particle_hashes

        # Put the hashes back
        all_hashes = np.empty(values.shape[1], dtype=int)
        all_hashes[active_idx] = particle_hashes
        return all_hashes


def hash_to_neighbors(hash_id, bits):
    """Compute neighboring cells from a hash.

    Parameters
    ----------
    hash_id :
        hash value of the current cell.
    bits :
        key to compute the hashes.

    Returns
    -------
    type
        neighbors: List of cells neighboring hash_id.

    """
    coor = np.zeros((len(bits),), dtype=np.int32)
    new_coor = np.zeros((len(bits),), dtype=np.int32)

    # Compute the (ix, iy, iz) coordinates of the hash.
    tot_bits = 0
    for dim in range(len(bits)):
        coor[dim] = (hash_id >> tot_bits) & ((1 << bits[dim]) - 1)
        tot_bits += bits[dim]

    coor_max = np.left_shift(1, bits)

    neighbors = []

    # Loop over all 3^3 neighboring cells.
    for offset in range(pow(3, len(bits))):
        # Compute the integer coordinates of the neighboring cell.
        divider = 1
        for dim in range(len(bits)):
            new_coor[dim] = coor[dim] + (1 - ((offset // divider) % 3))
            divider *= 3

        # Cell is outside the box/doesn't exist.
        if np.any(new_coor > coor_max) or np.any(new_coor < 0):
            continue

        # Compute the hash of the neighboring cell
        new_hash = 0
        tot_bits = 0
        for dim in range(len(bits)):
            new_hash |= new_coor[dim] << tot_bits
            tot_bits += bits[dim]
        neighbors.append(new_hash)
    return neighbors

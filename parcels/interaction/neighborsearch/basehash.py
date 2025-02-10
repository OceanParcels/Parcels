from abc import ABC, abstractmethod

import numpy as np


class BaseHashNeighborSearch(ABC):
    def find_neighbors_by_coor(self, coor):
        """Get the neighbors around a certain location.

        Parameters
        ----------
        coor :
            Numpy array with [depth, lat, lon].

        Returns
        -------
        type
            List of particle indices.

        """
        coor = coor.reshape(3, 1)
        hash_id = self._values_to_hashes(coor)[0]
        return self._find_neighbors(hash_id, coor)

    def find_neighbors_by_idx(self, particle_idx):
        """Get the neighbors around a certain particle.

        Mainly useful for Structure of Array (SoA) datastructure

        Parameters
        ----------
        particle_idx :
            index of the particle (SoA).

        Returns
        -------
        type
            List of particle indices

        """
        hash_id = self._particle_hashes[particle_idx]
        coor = self._values[:, particle_idx].reshape(3, 1)
        return self._find_neighbors(hash_id, coor)

    @abstractmethod
    def _find_neighbors(self, hash_id, coor):
        raise NotImplementedError

    def consistency_check(self):
        """See if all values are in their proper place.

        Only used for debugging purposes.
        """
        active_idx = self.active_idx
        if active_idx is None:
            active_idx = np.arange(self._values.shape[1])

        for idx in active_idx:
            cur_hash = self._particle_hashes[idx]
            hash_idx = self._hash_idx[idx]
            assert self._hashtable[cur_hash][hash_idx] == idx

        n_idx = 0
        for idx_array in self._hashtable.values():
            for idx in idx_array:
                assert idx in active_idx
            n_idx += len(idx_array)
        assert n_idx == len(active_idx)
        cur_hashes = self._values_to_hashes(self._values[:, active_idx])
        assert np.all(cur_hashes == self._particle_hashes[active_idx])

    def update_values(self, new_values, new_active_mask=None):
        """Update the locations of (some) of the particles.

        Particles that stay in the same location are computationally cheap.
        The order and number of the particles is assumed to remain the same.

        Parameters
        ----------
        new_values :
            new (depth, lat, lon) values for particles.
        new_active_mask :
             (Default value = None)
        """
        if self._values is None:
            self.rebuild(new_values, new_active_mask)
            return

        if new_active_mask is None:
            new_active_mask = np.full(new_values.shape[1], True)

        # Figure out the changes in the active mask.
        deactivated_mask = np.logical_and(self._active_mask, np.logical_not(new_active_mask))
        stay_active_mask = np.logical_and(self._active_mask, new_active_mask)
        activated_mask = np.logical_and(np.logical_not(self._active_mask), new_active_mask)

        stay_active_idx = np.where(stay_active_mask)[0]

        # Find the old and new hashes of particles that stayed active.
        old_hashes = self._particle_hashes[stay_active_mask]
        new_hashes = self._values_to_hashes(new_values[:, stay_active_mask])

        # See which particles have crossed cell boundaries.
        move_idx = stay_active_idx[np.where(old_hashes != new_hashes)[0]]
        remove_idx = np.append(move_idx, np.where(deactivated_mask)[0])
        add_idx = np.append(move_idx, np.where(activated_mask)[0])

        # Remove/add/modify particles.
        self._deactivate_particles(remove_idx)
        self._particle_hashes[stay_active_mask] = new_hashes
        self._particle_hashes[activated_mask] = self._values_to_hashes(new_values[:, activated_mask])
        self._activate_particles(add_idx)

        # Set the state to the new values.
        self._active_mask = new_active_mask
        self._values = new_values

    @abstractmethod
    def _values_to_hashes(self, values, active_idx=None):
        """Convert (particle) coordinates to hashes.

        The hashes correspond to the cells that particles reside in.

        Parameters
        ----------
        values :
            3D coordinates to be hashed.
        active_idx :
            Active particle indices (relative to values). (Default value = None)

        Returns
        -------
        type
            all_hashes: An array of length len(values) with hashes.

        """
        raise NotImplementedError

    def _deactivate_particles(self, particle_idx):
        """Remove particles from the hashtable."""
        # Get the hashes of the particles to be removed.
        remove_split = hash_split(self._particle_hashes[particle_idx])
        for cur_hash, remove_idx in remove_split.items():
            # If the number of items to removed from cur_hash is equal
            # to the number of hashes stored under cur_hash, remove the entry.
            if len(remove_idx) == len(self._hashtable[cur_hash]):
                del self._hashtable[cur_hash]
            # Else create a new array that doesn't include remove_idx.
            else:
                rel_remove_idx = self._hash_idx[particle_idx[remove_idx]]
                self._hashtable[cur_hash] = np.delete(self._hashtable[cur_hash], rel_remove_idx)
                self._hash_idx[self._hashtable[cur_hash]] = np.arange(len(self._hashtable[cur_hash]))

    def _activate_particles(self, particle_idx):
        """Add particles to the hashtable"""
        # See _deactivate_particles.
        add_split = hash_split(self._particle_hashes[particle_idx])
        for cur_hash, add_idx in add_split.items():
            if cur_hash not in self._hashtable:
                self._hashtable[cur_hash] = particle_idx[add_idx]
                self._hash_idx[particle_idx[add_idx]] = np.arange(len(add_idx))
            else:
                self._hash_idx[particle_idx[add_idx]] = np.arange(
                    len(self._hashtable[cur_hash]), len(self._hashtable[cur_hash]) + len(add_idx)
                )
                self._hashtable[cur_hash] = np.append(self._hashtable[cur_hash], particle_idx[add_idx])


def hash_split(hash_ids, active_idx=None):
    """Create a hashtable.

    Multiple particles that are found in the same cell are put in a list
    with that particular hash.

    Parameters
    ----------
    hash_ids :
        Hash values for the particles.
    active_idx :
        Subset on which to compute the hash split. (Default value = None)

    Returns
    -------
    type
        hash_split: Dictionary with {hash: [idx_1, idx_2, ..], ..}

    """
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
    return dict(zip(unq_items, unq_idx, strict=True))

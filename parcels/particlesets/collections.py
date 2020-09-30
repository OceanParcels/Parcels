import numpy as np
from abc import ABC
from abc import abstractmethod

from .baseparticleset import BaseParticleAccessor
from parcels.particle import ScipyParticle, JITParticle

class ParticleCollection(ABC):
    _ncount = -1

    def __init__(self):
        """
        Initializes a particle collection by pre-allocating memory (where needed), initialising indexing structures
        (where needed), initialising iterators and preparing the C-JIT-glue.
        """
        pass

    def __del__(self):
        pass

    def __add__(self, other):
        pass

    def add_collection(self, pcollection):
        pass

    def add_single(self, particle_obj):
        assert (isinstance(particle_obj, BaseParticleAccessor) or isinstance(particle_obj, ScipyParticle))

    def add_same(self, same_class):
        pass

    def __iadd__(self, same_class):
        assert same_class is not None
        assert type(same_class) is type(self), "Trying to increment-add collection of type {} into collection of type {} - invalid operation.".format(type(same_class), type(self))

    def insert(self, obj):
        pass

    def push(self, particle_obj):
        return -1

    def __sub__(self, other):
        self.remove(other)

    def remove(self, other):
        pass

    def remove_single_by_index(self, index):
        assert type(index) in [int, np.int32], "Trying to remove a particle by index, but index {} is not a 32-bit integer - invalid operation.".format(index)

    def remove_single_by_object(self, particle_obj):
        assert (isinstance(particle_obj, BaseParticleAccessor) or isinstance(particle_obj, ScipyParticle))

    def remove_single_by_ID(self, id):
        assert type(id) in [np.int64, np.uint64], "Trying to remove a particle by ID, but ID {} is not a 64-bit (signed or unsigned) iteger - invalid operation.".format(id)

    def remove_multi_by_PyCollection_Particles(self, pycollectionp):
        assert type(pycollectionp) in [list, dict, np.ndarray], "Trying to remove a collection of Particles, but their container is not a valid Python-collection - invalid operation."

    def remove_multi_by_IDs(self, ids):
        assert ids is not None, "Trying to remove particles by their IDs, but the ID list is None - invalid operation."
        assert type(ids) in [list, dict, np.ndarray], "Trying to remove particles by their IDs, but the ID container is not a valid Python-collection - invalid operation."
        if type(ids) is not dict:
            assert ids[0] in [np.int64, np.uint64], "Trying to remove particles by their IDs, but the ID type in the Python collection ins not a 64-bit (signed or unsigned) integer - invalid operation."
        else:
            assert ids.values()[0] in [np.int64, np.uint64], "Trying to remove particles by their IDs, but the ID type in the Python collection ins not a 64-bit (signed or unsigned) integer - invalid operation."

    def remove_multi_collection(self, pcollection):
        assert isinstance(pcollection, ParticleCollection), "Trying to remove particles via another ParticleCollection, but the other particles are not part of a ParticleCollection - invalid operation."

    def pop_single_by_index(self, index):
        """
        Searches for Particle at index 'index', removes that Particle from the Collection and returns that Particle (or: ParticleAccessor).
        If index is out of bounds, throws and OutOfRangeException
        If Particle cannot be retrieved, returns None.
        """
        return None

    def pop_single_by_ID(self, id):
        """
        Searches for Particle with ID 'id', removes that Particle from the Collection and returns that Particle (or: ParticleAccessor).
        If Particle cannot be retrieved (e.g. because the ID is not available), returns None.
        """
        return None

    def pop_multi_by_indices(self, indices):
        """
        Searches for Particles with the indices registered in 'indices', removes the Particles from the Collection and returns the Particles (or: their ParticleAccessors).
        If index in 'indices' is out of bounds, throws and OutOfRangeException
        If Particles cannot be retrieved, returns None.
        """
        return None

    def pop_multi_by_IDs(self, ids):
        """
        Searches for Particles with the IDs registered in 'ids', removes the Particles from the Collection and returns the Particles (or: their ParticleAccessors).
        If Particles cannot be retrieved (e.g. because the IDs are not available), returns None.
        """
        return None

    def __delitem__(self, key):
        pass

    def delete_by_index(self, index):
        pass

    def delete_by_ID(self, id):
        pass

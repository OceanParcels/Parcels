import numpy as np
from abc import ABC
from abc import abstractmethod

from .baseparticleset import BaseParticleAccessor
from parcels.particle import ScipyParticle, JITParticle

class ParticleCollection(ABC):
    _ncount = -1

    @abstractmethod
    def __init__(self):
        """
        ParticleCollection - Constructor
        Initializes a particle collection by pre-allocating memory (where needed), initialising indexing structures
        (where needed), initialising iterators and preparing the C-JIT-glue.
        """
        pass

    @abstractmethod
    def __del__(self):
        """
        ParticleCollection - Destructor
        """
        pass

    @abstractmethod
    def __add__(self, other):
        """
        This is a generic super-method to add one- or multiple objects to this collection. Ideally, it just discerns
        between the types of the 'other' parameter, and then forwards the call to the related specific function.
        """
        pass

    def add_collection(self, pcollection):
        """
        Adds another, differently structured ParticleCollection to this collection. This is done by, for example,
        appending/adding the items of the other collection to this collection.
        """
        assert pcollection is not None, "Trying to add another particle collection to this one, but the other one is None - invalid operation."
        assert isinstance(pcollection, ParticleCollection), "Trying to add another particle collection to this one, but the other is not of the type of 'ParticleCollection' - invalid operation."
        assert type(pcollection) is not type(self)


    def add_single(self, particle_obj):
        """
        Adding a single Particle to the collection - either as a 'Particle; object in parcels itself, or
        via its ParticleAccessor.
        """
        assert (isinstance(particle_obj, BaseParticleAccessor) or isinstance(particle_obj, ScipyParticle))

    def add_same(self, same_class):
        """
        Adds another, equi-structured ParticleCollection to this collection. This is done by concatenating
        both collections. The fact that they are of the same ParticleCollection's derivative simplifies
        parsing and concatenation.
        """
        pass

    def __iadd__(self, same_class):
        """
        Performs an incremental addition of the equi-structured ParticleCollections, such to allow

        a += b,

        with 'a' and 'b' begin the two equi-structured objects.
        """
        assert same_class is not None
        assert type(same_class) is type(self), "Trying to increment-add collection of type {} into collection of type {} - invalid operation.".format(type(same_class), type(self))

    @abstractmethod
    def insert(self, obj, index=None):
        """
        This function allows to 'insert' a Particle (as object or via its accessor) into this collection. This method
        needs to be specified to each collection individually. Some collections (e.g. unordered list) allow to define
        the index where the object is to be inserted. Some collections can optionally insert an object at a specific
        position - at a significant speed- and memory malus cost (e.g. vectors, arrays, dense matrices). Some
        collections that manage a specified indexing order internally (e.g. ordered lists, sets, trees), and thus
        have no use for an 'index' parameter. For those collections with an internally-enforced order, the function
        mapping equates to:

        insert(obj) -> add_single(obj)
        """
        pass

    @abstractmethod
    def push(self, particle_obj):
        """
        This function pushes a Particle (as object or via its accessor) to the end of a collection ('end' definition
        depends on the specific collection itself). For collections with an inherent indexing order (e.g. ordered lists,
        sets, trees), the function just includes the object at its pre-defined position (i.e. not necessarily at the
        end). For the collections, the function mapping equates to:

        int32 push(particle_obj) -> add_single(particle_obj); return -1;

        This function further returns the index, at which position the Particle has been inserted. By definition,
        the index is positive, thus: a return of '-1' indicates push failure, NOT the last position in the collection.
        Furthermore, collections that do not work on an index-preserving manner also return '-1'.
        """
        pass

    @abstractmethod
    def append(self, particle_obj):
        """
        This function appends a Particle (as object or via its accessor) to the end of a collection ('end' definition
        depends on the specific collection itself). For collections with an inherent indexing order (e.g. ordered lists,
        sets, trees), the function just includes the object at its pre-defined position (i.e. not necessarily at the
        end). For the collections, the function mapping equates to:

        append(particle_obj) -> add_single(particle_obj)

        The function - in contrast to 'push' - does not return the index of the inserted object.
        """
        pass

    def __sub__(self, other):
        """
        This is a generic super-method to remove one- or multiple Particles (via their object, their ParticleAccessor,
        their ID or their index) from the collection. As the function applies to collections itself, it maps directly
        to:

        a-b -> a.remove(b)
        """
        self.remove(other)

    @abstractmethod
    def remove(self, other):
        """
        This is a generic super-method to remove one- or multiple Particles (via their object, their ParticleAccessor,
        their ID or their index) from the collection. Ideally, it just discerns between the types of the 'other'
        parameter, and then forwards the call to the related specific function.
        """
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

    @abstractmethod
    def __isub__(self, other):
        pass

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

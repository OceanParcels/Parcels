import numpy as np
from abc import ABC
from abc import abstractmethod

from .baseparticleset import BaseParticleAccessor
from parcels.particle import ScipyParticle, JITParticle

"""
Author: Dr. Christian Kehl
github relation: #913 (particleset_class_hierarchy)
"""

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

    def __add__(self, other):
        """
        This is a generic super-method to add one- or multiple objects to this collection. Ideally, it just discerns
        between the types of the 'other' parameter, and then forwards the call to the related specific function.

        Comment/Annotation:
        Functions for adding multiple objects are more specialised than just a for-each loop of single-item addition,
        because certain data structures can add multiple objects in-bulk faster with specialised function than making a
        roundtrip per-item add operation. Because of the sheer size of those containers and the resulting
        performance demands, we need to make use of those specialised 'add' functions, where available.
        """
        if other is None:
            return
        if type(other) is type(self):
            self.add_same(other)
        elif isinstance(other, ParticleCollection):
            self.add_collection(other)
        else:
            self.add_single(other)

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
        assert same_class is not None, "Trying to add another {}} to this one, but the other one is None - invalid operation.".format(type(self))
        assert type(same_class) is type(self)

    def __iadd__(self, same_class):
        """
        Performs an incremental addition of the equi-structured ParticleCollections, such to allow

        a += b,

        with 'a' and 'b' begin the two equi-structured objects (or: 'b' being and individual object).
        This operation is equal to an in-place addition of (an) element(s).
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

    def remove(self, other):
        """
        This is a generic super-method to remove one- or multiple Particles (via their object, their ParticleAccessor,
        their ID or their index) from the collection. Ideally, it just discerns between the types of the 'other'
        parameter, and then forwards the call to the related specific function.

        Comment/Annotation:
        Functions for removing multiple objects are more specialised than just a for-each loop of single-item removal,
        because certain data structures can remove multiple objects faster with specialised function than making a
        roundtrip per-item check-and-remove operation. Because of the sheer size of those containers and the resulting
        performance demands, we need to make use of those specialised 'remove' functions, where available.
        """
        if other is None:
            return
        if type(other) is type(self):
            self.remove_same(other)
        elif isinstance(other, ParticleCollection):
            self.remove_collection(other)
        elif type(other) in [list, dict, np.ndarray]:
            # multi-removal routines - hard to discern at this point
            if type(other) is not dict:
                if other[0] in [int, np.int32]:
                    self.remove_multi_by_indices(other)
                elif other[0] in [np.int64, np.uint64]:
                    self.remove_multi_by_IDs(other)
                else:
                    self.remove_multi_by_PyCollection_Particles(other)
            else:
                if other.values()[0] in [int, np.int32]:
                    self.remove_multi_by_indices(other)
                elif other.values()[0] in [np.int64, np.uint64]:
                    self.remove_multi_by_IDs(other)
                else:
                    self.remove_multi_by_PyCollection_Particles(other)
        elif type(other) in [int, np.int32]:
            self.remove_single_by_index(other)
        elif type(other) in [np.int64, np.uint64]:
            self.remove_single_by_ID(other)
        else:
            self.remove_single_by_object(other)

    def remove_single_by_index(self, index):
        """
        This function removes a (particle) object from the collection based on its index within the collection. For
        collections that are not based on random access (e.g. ordered lists, sets, trees), this function involves a
        translation of the index into the specific object reference in the collection - or (if unavoidable) the
        translation of the collection from a none-indexable, none-random-access structure into an indexable structure,
        and then perform the removal.
        In cases where a removal-by-index would result in a performance malus, it is highly-advisable to use a different
        removal functions, e.g. remove-by-object or remove-by-ID.
        """
        assert type(index) in [int, np.int32], "Trying to remove a particle by index, but index {} is not a 32-bit integer - invalid operation.".format(index)

    def remove_single_by_object(self, particle_obj):
        """
        This function removes a (particle) object from the collection based on its actual object. For collections that
        are random-access and based on indices (e.g. unordered list, vectors, arrays and dense matrices), this function
        would involves a parsing of the whole list and translation of the object into an index in the collection to
        perform the removal - which results in a significant performance malus.
        In cases where a removal-by-object would result in a performance malus, it is highly-advisable to use a different
        removal functions, e.g. remove-by-index or remove-by-ID.
        """
        assert (isinstance(particle_obj, BaseParticleAccessor) or isinstance(particle_obj, ScipyParticle))

    def remove_single_by_ID(self, id):
        """
        This function removes a (particle) object from the collection based on the object's ID. For some collections,
        this operation may involve a parsing of the whole list and translation of the object's ID into an index  or an
        object reference in the collection in order to perform the removal - which results in a significant performance
        malus.
        In cases where a removal-by-ID would result in a performance malus, it is highly-advisable to use a different
        removal functions, e.g. remove-by-object or remove-by-index.
        """
        assert type(id) in [np.int64, np.uint64], "Trying to remove a particle by ID, but ID {} is not a 64-bit (signed or unsigned) iteger - invalid operation.".format(id)

    def remove_same(self, same_class):
        assert same_class is not None, "Trying to remove another {} from this one, but the other one is None - invalid operation.".format(type(self))
        assert type(same_class) is type(self)

    def remove_collection(self, pcollection):
        assert pcollection is not None, "Trying to remove another particle collection from this one, but the other one is None - invalid operation."
        assert isinstance(pcollection, ParticleCollection), "Trying to remove another particle collection from this one, but the other is not of the type of 'ParticleCollection' - invalid operation."
        assert type(pcollection) is not type(self)

    def remove_multi_by_PyCollection_Particles(self, pycollectionp):
        assert type(pycollectionp) in [list, dict, np.ndarray], "Trying to remove a collection of Particles, but their container is not a valid Python-collection - invalid operation."

    def remove_multi_by_indices(self, indices):
        assert indices is not None, "Trying to remove particles by their collection indices, but the index list is None - invalid operation."
        assert type(indices) in [list, dict, np.ndarray], "Trying to remove particles by their IDs, but the ID container is not a valid Python-collection - invalid operation."
        if type(indices) is not dict:
            assert indices[0] in [int, np.int32], "Trying to remove particles by their index, but the index type in the Python collection is not a 32-bit integer - invalid operation."
        else:
            assert indices.values()[0] in [int, np.int32], "Trying to remove particles by their index, but the index type in the Python collection is not a 32-bit integer - invalid operation."

    def remove_multi_by_IDs(self, ids):
        assert ids is not None, "Trying to remove particles by their IDs, but the ID list is None - invalid operation."
        assert type(ids) in [list, dict, np.ndarray], "Trying to remove particles by their IDs, but the ID container is not a valid Python-collection - invalid operation."
        if type(ids) is not dict:
            assert ids[0] in [np.int64, np.uint64], "Trying to remove particles by their IDs, but the ID type in the Python collection is not a 64-bit (signed or unsigned) integer - invalid operation."
        else:
            assert ids.values()[0] in [np.int64, np.uint64], "Trying to remove particles by their IDs, but the ID type in the Python collection is not a 64-bit (signed or unsigned) integer - invalid operation."

    @abstractmethod
    def __isub__(self, other):
        """
        This method performs an incremental removal of the equi-structured ParticleCollections, such to allow

        a -= b,

        with 'a' and 'b' begin the two equi-structured objects (or: 'b' being and individual object).
        This operation is equal to an in-place removal of (an) element(s).
        """
        pass

    def pop(self, other):
        """
        This function pushes a Particle (as object or via its accessor) to the end of a collection ('end' definition
        depends on the specific collection itself). For collections with an inherent indexing order (e.g. ordered lists,
        sets, trees), the function just includes the object at its pre-defined position (i.e. not necessarily at the
        end). For the collections, the function mapping equates to:

        int32 push(particle_obj) -> add_single(particle_obj); return -1;

        This function further returns the index, at which position the Particle has been inserted. By definition,
        the index is positive, thus: a return of '-1' indicates push failure, NOT the last position in the collection.
        Furthermore, collections that do not work on an index-preserving manner also return '-1'.

        Comment/Annotation:
        Functions for popping multiple objects are more specialised than just a for-each loop of single-item pop,
        because certain data structures can pop multiple objects faster with specialised function than making a
        roundtrip per-item check-and-pop operation. Because of the sheer size of those containers and the resulting
        performance demands, we need to make use of those specialised 'pop' functions, where available.
        """
        if other is None:
            return
        if type(other) in [int, np.int32]:
            self.pop_single_by_index(other)
        elif type(other) in [np.int64, np.uint64]:
            self.pop_single_by_ID(other)
        elif type(other) in [list, dict, np.ndarray]:
            # multi-removal routines - hard to discern at this point
            if type(other) is not dict:
                if other[0] in [int, np.int32]:
                    self.pop_multi_by_indices(other)
                elif other[0] in [np.int64, np.uint64]:
                    self.pop_multi_by_IDs(other)
            else:
                if other.values()[0] in [int, np.int32]:
                    self.pop_multi_by_indices(other)
                elif other.values()[0] in [np.int64, np.uint64]:
                    self.pop_multi_by_IDs(other)

    def pop_single_by_index(self, index):
        """
        Searches for Particle at index 'index', removes that Particle from the Collection and returns that Particle (or: ParticleAccessor).
        If index is None, return last item (-1);
        If index < 0: return from 'end' of collection.
        If index is out of bounds, throws and OutOfRangeException.
        If Particle cannot be retrieved, returns None.
        """
        assert type(index) in [int, np.int32], "Trying to pop a particle by index, but index {} is not a 32-bit integer - invalid operation.".format(index)
        return None

    def pop_single_by_ID(self, id):
        """
        Searches for Particle with ID 'id', removes that Particle from the Collection and returns that Particle (or: ParticleAccessor).
        If Particle cannot be retrieved (e.g. because the ID is not available), returns None.
        """
        assert type(id) in [np.int64, np.uint64], "Trying to pop a particle by ID, but ID {} is not a 64-bit (signed or unsigned) iteger - invalid operation.".format(id)
        return None

    def pop_multi_by_indices(self, indices):
        """
        Searches for Particles with the indices registered in 'indices', removes the Particles from the Collection and returns the Particles (or: their ParticleAccessors).
        If indices is None -> Particle cannot be retrieved -> Assert-Error and return None
        If index is None, return last item (-1);
        If index < 0: return from 'end' of collection.
        If index in 'indices' is out of bounds, throws and OutOfRangeException.
        If Particles cannot be retrieved, returns None.
        """
        assert indices is not None, "Trying to pop particles by their collection indices, but the index list is None - invalid operation."
        assert type(indices) in [list, dict, np.ndarray], "Trying to pop particles by their IDs, but the ID container is not a valid Python-collection - invalid operation."
        if type(indices) is not dict:
            assert indices[0] in [int, np.int32], "Trying to pop particles by their index, but the index type in the Python collection is not a 32-bit integer - invalid operation."
        else:
            assert indices.values()[0] in [int, np.int32], "Trying to pop particles by their index, but the index type in the Python collection is not a 32-bit integer - invalid operation."
        return None

    def pop_multi_by_IDs(self, ids):
        """
        Searches for Particles with the IDs registered in 'ids', removes the Particles from the Collection and returns the Particles (or: their ParticleAccessors).
        If Particles cannot be retrieved (e.g. because the IDs are not available), returns None.
        """
        assert ids is not None, "Trying to pop particles by their IDs, but the ID list is None - invalid operation."
        assert type(ids) in [list, dict, np.ndarray], "Trying to pop particles by their IDs, but the ID container is not a valid Python-collection - invalid operation."
        if type(ids) is not dict:
            assert ids[0] in [np.int64, np.uint64], "Trying to pop particles by their IDs, but the ID type in the Python collection is not a 64-bit (signed or unsigned) integer - invalid operation."
        else:
            assert ids.values()[0] in [np.int64, np.uint64], "Trying to pop particles by their IDs, but the ID type in the Python collection is not a 64-bit (signed or unsigned) integer - invalid operation."
        return None

    def __delitem__(self, key):
        """
        This is a generic super-method to delete a specific object from this collection. Ideally, it just discerns
        between the types of the 'key' parameter, and then forwards the call to the related specific function.

        Comment/Annotation:
        Functions for deleting multiple objects are more specialised than just a for-each loop of single-item deletion,
        because certain data structures can delete multiple objects in-bulk faster with specialised function than making a
        roundtrip per-item delete operation. Because of the sheer size of those containers and the resulting
        performance demands, we need to make use of those specialised 'del' functions, where available.
        """
        if key is None:
            return
        if type(key) in [int, np.int32]:
            self.delete_by_index(key)
        elif type(key) in [np.int64, np.uint64]:
            self.delete_by_ID(key)

    @abstractmethod
    def delete_by_index(self, index):
        """
        This method deletes a particle from the  the collection based on its index. It does not return the deleted item.
        Semantically, the function appears similar to the 'remove' operation. That said, the function in OceanParcels -
        instead of directly deleting the particle - just raises the 'deleted' status flag for the indexed particle.
        In result, the particle still remains in the collection. The functional interpretation of the 'deleted' status
        is handled by 'recovery' dictionary during simulation execution.
        """
        pass

    @abstractmethod
    def delete_by_ID(self, id):
        """
        This method deletes a particle from the  the collection based on its ID. It does not return the deleted item.
        Semantically, the function appears similar to the 'remove' operation. That said, the function in OceanParcels -
        instead of directly deleting the particle - just raises the 'deleted' status flag for the indexed particle.
        In result, the particle still remains in the collection. The functional interpretation of the 'deleted' status
        is handled by 'recovery' dictionary during simulation execution.
        """
        pass

    @abstractmethod
    def _clear_deleted_(self):
        """
        This (protected) function physically removes particles from the collection whose status is set to 'DELETE'.
        It is the logical finalisation method of physically deleting particles that have been marked for deletion and
        that have not otherwise been recovered.
        This methods in heavily dependent on the actual collection type and should be implemented very specific
        to the actual data structure, to remove objects 'the fastest way possible'.
        """
        pass
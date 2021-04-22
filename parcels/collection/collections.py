import numpy as np
from abc import ABC
from abc import abstractmethod

from .iterators import BaseParticleAccessor
from parcels.particle import ScipyParticle

"""
Author: Dr. Christian Kehl
github relation: #913 (particleset_class_hierarchy)
"""


class Collection(ABC):
    _ncount = -1
    _iterator = None
    _riterator = None

    @abstractmethod
    def __init__(self):
        """
        Collection - Constructor
        Initializes a collection by pre-allocating memory (where needed), initialising indexing structures
        (where needed), initialising iterators and preparing the C-JIT-glue.
        """
        pass

    @abstractmethod
    def __del__(self):
        """
        Collection - Destructor
        """
        pass

    @property
    def ncount(self):
        return self._ncount

    @abstractmethod
    def __iter__(self):
        """
        This function represents a forward-iteration over the collection, the the sense it is called

        for item in collection_object:
            <do something>

        In this sense, this function (if actually being implemented as such - which is not necessarily the sensible case
        for every collection) would here return an iterator (either re-created or persistent; preferably created anew).

        For iterating over a collection, it is possible to 'iterate' over an iterator, such that:

        collection_iterator = collection_object.iterator()
        for item in collection_iterator:
            <do something>
        """
        pass

    @abstractmethod
    def iterator(self):
        """
        This function is an explicit object-return of a forward-iterator over this collection. If this iterator is
        persistent or re-created upon call depends on the specific implementation of the '__iter__' function.

        This function is an explicit forward to the Collection::__iter__() member function.
        """
        pass

    @abstractmethod
    def __reversed__(self):
        """
        This function represents a backward-iteration over the collection, the the sense it is called

        for item in reversed(collection_object):
            <do something>

        In this sense, this function (if actually being implemented as such - which is not necessarily the sensible case
        for every collection) would here return a reverse-iterator (either re-created or persistent; preferably created
        anew).

        For iterating over a collection, it is possible to 'iterate' over an iterator, such that:

        collection_iterator = collection_object.reverse_iterator()
        for item in collection_iterator:
            <do something>
        """
        pass

    @abstractmethod
    def reverse_iterator(self):
        """
        This function is an explicit object-return of a backward-iterator over this collection. If this iterator is
        persistent or re-created upon call depends on the specific implementation of the '__reversed__' function.

        This function is an explicit forward to the Collection::__reversed__() member function.
        """
        pass

    @abstractmethod
    def __getitem__(self, item):
        """
        This function should return an item [accessor, as reference] at a certain 'position', in the sense of

        item_ref = collection_object[<some_indexing_concept>]

        The actual return and implementation of this will depend on the specific collection at hand. Ideally, this
        function demands as 'key' an object that matches the most-performant access method, i.e.

        ordered lists, sets, trees -> getitem(key = [int64, uint64] id) -> operates on a list-iterator
        arrays, vectors, dense arrays & matrices -> getitem(key = [int, int32] index) -> operates on an index-managed iterator

        Out of performance reasons, there should be no alternative getitem with other positional arguments than the
        optimal one.
        """
        pass

    def get(self, other):
        """
        This is a generic super-method to get one- or multiple Particles (via their object, their ParticleAccessor,
        their ID or their index) from the collection. Ideally, it just discerns between the types of the 'other'
        parameter, and then forwards the call to the related specific function.

        Comment/Annotation:
        Not all arguments have a sensible use-case in every datastructure, so some concrete classes may not implementat
        all of them.
        """
        if other is None:
            return
        if type(other) is type(self):
            return self.get_same(other)
        elif isinstance(other, ParticleCollection):
            return self.get_collection(other)
        elif type(other) in [list, dict, np.ndarray]:
            # multi-get routines - hard to discern at this point
            if type(other) is not dict:
                if len(other) == 0:
                    return None
                elif type(other[0]) in [int, np.int32, np.intp]:
                    return self.get_multi_by_indices(other)
                elif type(other[0]) in [np.int64, np.uint64]:
                    return self.get_multi_by_IDs(other)
                else:
                    return self.get_multi_by_PyCollection_Particles(other)
            else:
                if len(list(other.values())) == 0:
                    return None
                if type(list(other.values())[0]) in [int, np.int32, np.intp]:
                    return self.get_multi_by_indices(other)
                elif type(list(other.values())[0]) in [np.int64, np.uint64]:
                    return self.get_multi_by_IDs(other)
                else:
                    return self.get_multi_by_PyCollection_Particles(other)
        elif type(other) in [int, np.int32, np.intp]:
            return self.get_single_by_index(other)
        elif type(other) in [np.int64, np.uint64]:
            return self.get_single_by_ID(other)
        else:
            return self.get_single_by_object(other)

    @abstractmethod
    def get_single_by_index(self, index):
        """
        This function gets a (particle) object from the collection based on its index within the collection. For
        collections that are not based on random access (e.g. ordered lists, sets, trees), this function involves a
        translation of the index into the specific object reference in the collection - or (if unavoidable) the
        translation of the collection from a none-indexable, none-random-access structure into an indexable structure.
        In cases where a get-by-index would result in a performance malus, it is highly-advisable to use a different
        get function, e.g. get-by-ID.
        """
        assert type(index) in [int, np.int32, np.intp], "Trying to get a particle by index, but index {} is not a 32-bit integer - invalid operation.".format(index)

    @abstractmethod
    def get_single_by_object(self, particle_obj):
        """
        This function gets a (particle) object from the collection based on its actual object. For collections that
        are random-access and based on indices (e.g. unordered list, vectors, arrays and dense matrices), this function
        would involve a parsing of the whole list and translation of the object into an index in the collection - which
        results in a significant performance malus.
        In cases where a get-by-object would result in a performance malus, it is highly-advisable to use a different
        get function, e.g. get-by-index or get-by-ID.
        """
        assert (isinstance(particle_obj, BaseParticleAccessor) or isinstance(particle_obj, ScipyParticle))

    @abstractmethod
    def get_single_by_ID(self, id):
        """
        This function gets a (particle) object from the collection based on the object's ID. For some collections,
        this operation may involve a parsing of the whole list and translation of the object's ID into an index  or an
        object reference in the collection - which results in a significant performance malus.
        In cases where a get-by-ID would result in a performance malus, it is highly-advisable to use a different
        get function, e.g. get-by-index.
        """
        assert type(id) in [np.int64, np.uint64], "Trying to get a particle by ID, but ID {} is not a 64-bit (signed or unsigned) iteger - invalid operation.".format(id)

    @abstractmethod
    def get_same(self, same_class):
        """
        This function gets particles from this collection that are themselves stored in another object of an equi-
        structured ParticleCollection.
        """
        assert same_class is not None, "Trying to get another {} from this one, but the other one is None - invalid operation.".format(type(self))
        assert type(same_class) is type(self)

    @abstractmethod
    def get_collection(self, pcollection):
        """
        This function gets particles from this collection that are themselves stored in a ParticleCollection, which
        is differently structured than this one. That means the other-collection has to be re-formatted first in an
        intermediary format.
        """
        assert pcollection is not None, "Trying to get another particle collection from this one, but the other one is None - invalid operation."
        assert isinstance(pcollection, ParticleCollection), "Trying to get another particle collection from this one, but the other is not of the type of 'ParticleCollection' - invalid operation."
        assert type(pcollection) is not type(self)

    @abstractmethod
    def get_multi_by_PyCollection_Particles(self, pycollectionp):
        """
        This function gets particles from this collection, which are themselves in common Python collections, such as
        lists, dicts and numpy structures. We can either directly get the referred Particle instances (for internally-
        ordered collections, e.g. ordered lists, sets, trees) or we may need to parse each instance for its index (for
        random-access structures), which results in a considerable performance malus.

        For collections where get-by-object incurs a performance malus, it is advisable to multi-get particles
        by indices or IDs.
        """
        assert type(pycollectionp) in [list, dict, np.ndarray], "Trying to get a collection of Particles, but their container is not a valid Python-collection - invalid operation."

    @abstractmethod
    def get_multi_by_indices(self, indices):
        """
        This function gets particles from this collection based on their indices. This works best for random-access
        collections (e.g. numpy's ndarrays, dense matrices and dense arrays), whereas internally ordered collections
        shall rather use a get-via-object-reference strategy.
        """
        assert indices is not None, "Trying to get particles by their collection indices, but the index list is None - invalid operation."
        assert type(indices) in [list, dict, np.ndarray], "Trying to get particles by their IDs, but the ID container is not a valid Python-collection - invalid operation."
        if type(indices) is not dict:
            assert len(indices) == 0 or type(indices[0]) in [int, np.int32, np.intp], "Trying to get particles by their index, but the index type in the Python collection is not a 32-bit integer - invalid operation."
        else:
            assert len(list(indices.values())) == 0 or type(list(indices.values())[0]) in [int, np.int32, np.intp], "Trying to get particles by their index, but the index type in the Python collection is not a 32-bit integer - invalid operation."

    @abstractmethod
    def get_multi_by_IDs(self, ids):
        """
        This function gets particles from this collection based on their IDs. For collections where this removal
        strategy would require a collection transformation or by-ID parsing, it is advisable to rather apply a get-
        by-objects or get-by-indices scheme.
        """
        assert ids is not None, "Trying to get particles by their IDs, but the ID list is None - invalid operation."
        assert type(ids) in [list, dict, np.ndarray], "Trying to get particles by their IDs, but the ID container is not a valid Python-collection - invalid operation."
        if type(ids) is not dict:
            assert len(ids) == 0 or type(ids[0]) in [np.int64, np.uint64], "Trying to get particles by their IDs, but the ID type in the Python collection is not a 64-bit (signed or unsigned) integer - invalid operation."
        else:
            assert len(list(ids.values())) == 0 or type(list(ids.values())[0]) in [np.int64, np.uint64], "Trying to get particles by their IDs, but the ID type in the Python collection is not a 64-bit (signed or unsigned) integer - invalid operation."

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

    @abstractmethod
    def add_collection(self, pcollection):
        """
        Adds another, differently structured ParticleCollection to this collection. This is done by, for example,
        appending/adding the items of the other collection to this collection.
        """
        assert pcollection is not None, "Trying to add another particle collection to this one, but the other one is None - invalid operation."
        assert isinstance(pcollection, ParticleCollection), "Trying to add another particle collection to this one, but the other is not of the type of 'ParticleCollection' - invalid operation."
        assert type(pcollection) is not type(self)

    @abstractmethod
    def add_single(self, particle_obj):
        """
        Adding a single Particle to the collection - either as a 'Particle; object in parcels itself, or
        via its ParticleAccessor.
        """
        assert (isinstance(particle_obj, BaseParticleAccessor) or isinstance(particle_obj, ScipyParticle))

    @abstractmethod
    def add_same(self, same_class):
        """
        Adds another, equi-structured ParticleCollection to this collection. This is done by concatenating
        both collections. The fact that they are of the same ParticleCollection's derivative simplifies
        parsing and concatenation.
        """
        assert same_class is not None, "Trying to add another {} to this one, but the other one is None - invalid operation.".format(type(self))
        assert type(same_class) is type(self)

    @abstractmethod
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

    @abstractmethod
    def __delitem__(self, key):
        """
        This is the high-performance method to delete a specific object from this collection.
        As the most-performant way depends on the specific collection in question, the function is abstract.

        Highlight for the specific implementation:
        The 'key' parameter should still be evaluated for being a single or a multi-entry delete, and needs to check
        that it received the correct type of 'indexing' argument (i.e. index, id or iterator).
        """
        pass

    def delete(self, key):
        """
        This is the generic super-method to indicate obejct deletion of a specific object from this collection.

        Comment/Annotation:
        Functions for deleting multiple objects are more specialised than just a for-each loop of single-item deletion,
        because certain data structures can delete multiple objects in-bulk faster with specialised function than making a
        roundtrip per-item delete operation. Because of the sheer size of those containers and the resulting
        performance demands, we need to make use of those specialised 'del' functions, where available.
        """
        if key is None:
            return
        if type(key) in [int, np.int32, np.intp]:
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
        assert type(index) in [int, np.int32, np.intp], "Trying to delete a particle by index, but index {} is not a 32-bit integer - invalid operation.".format(index)

    @abstractmethod
    def delete_by_ID(self, id):
        """
        This method deletes a particle from the  the collection based on its ID. It does not return the deleted item.
        Semantically, the function appears similar to the 'remove' operation. That said, the function in OceanParcels -
        instead of directly deleting the particle - just raises the 'deleted' status flag for the indexed particle.
        In result, the particle still remains in the collection. The functional interpretation of the 'deleted' status
        is handled by 'recovery' dictionary during simulation execution.
        """
        assert type(id) in [np.int64, np.uint64], "Trying to delete a particle by ID, but ID {} is not a 64-bit (signed or unsigned) integer - invalid operation.".format(id)

    def __sub__(self, other):
        """
        This is a high-performance method to remove an object (via their object, their ParticleAccessor,
        their ID or their index) from the collection. As the function applies to collections itself, it maps directly
        to:

        a-b -> a.remove(b)
        """
        if other is None:
            return
        if type(other) is type(self):
            self.remove_same(other)
        elif isinstance(other, ParticleCollection):
            self.remove_collection(other)

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
                if len(other) == 0:
                    return
                elif type(other[0]) in [int, np.int32, np.intp]:
                    self.remove_multi_by_indices(other)
                elif type(other[0]) in [np.int64, np.uint64]:
                    self.remove_multi_by_IDs(other)
                else:
                    self.remove_multi_by_PyCollection_Particles(other)
            else:
                if len(list(other.values())) == 0:
                    return
                elif type(list(other.values())[0]) in [int, np.int32, np.intp]:
                    self.remove_multi_by_indices(other)
                elif type(list(other.values())[0]) in [np.int64, np.uint64]:
                    self.remove_multi_by_IDs(other)
                else:
                    self.remove_multi_by_PyCollection_Particles(other)
        elif type(other) in [int, np.int32, np.intp]:
            self.remove_single_by_index(other)
        elif type(other) in [np.int64, np.uint64]:
            self.remove_single_by_ID(other)
        else:
            self.remove_single_by_object(other)

    @abstractmethod
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
        assert type(index) in [int, np.int32, np.intp], "Trying to remove a particle by index, but index {} is not a 32-bit integer - invalid operation.".format(index)

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def remove_same(self, same_class):
        """
        This function removes particles from this collection that are themselves stored in another object of an equi-
        structured ParticleCollection. As the structures of both collections are the same, a more efficient M-in-N
        removal can be applied without an in-between reformatting.
        """
        assert same_class is not None, "Trying to remove another {} from this one, but the other one is None - invalid operation.".format(type(self))
        assert type(same_class) is type(self)

    @abstractmethod
    def remove_collection(self, pcollection):
        """
        This function removes particles from this collection that are themselves stored in a ParticleCollection, which
        is differently structured than this one. Tht means the removal first requires the removal-collection to be re-
        formatted in an intermediary format, before executing the removal.
        That said, this method should still be at least as efficient as a removal via common Python collections (i.e.
        lists, dicts, numpy's nD arrays & dense arrays). Despite this, due to the reformatting, in some cases it may
        be more efficient to remove items then rather by IDs oder indices.
        """
        assert pcollection is not None, "Trying to remove another particle collection from this one, but the other one is None - invalid operation."
        assert isinstance(pcollection, ParticleCollection), "Trying to remove another particle collection from this one, but the other is not of the type of 'ParticleCollection' - invalid operation."
        assert type(pcollection) is not type(self)

    @abstractmethod
    def remove_multi_by_PyCollection_Particles(self, pycollectionp):
        """
        This function removes particles from this collection, which are themselves in common Python collections, such as
        lists, dicts and numpy structures. In order to perform the removal, we can either directly remove the referred
        Particle instances (for internally-ordered collections, e.g. ordered lists, sets, trees) or we may need to parse
        each instance for its index (for random-access structures), which results in a considerable performance malus.

        For collections where removal-by-object incurs a performance malus, it is advisable to multi-remove particles
        by indices or IDs.
        """
        assert type(pycollectionp) in [list, dict, np.ndarray], "Trying to remove a collection of Particles, but their container is not a valid Python-collection - invalid operation."

    @abstractmethod
    def remove_multi_by_indices(self, indices):
        """
        This function removes particles from this collection based on their indices. This works best for random-access
        collections (e.g. numpy's ndarrays, dense matrices and dense arrays), whereas internally ordered collections
        shall rather use a removal-via-object-reference strategy.
        """
        assert indices is not None, "Trying to remove particles by their collection indices, but the index list is None - invalid operation."
        assert type(indices) in [list, dict, np.ndarray], "Trying to remove particles by their indices, but the index container is not a valid Python-collection - invalid operation."
        if type(indices) is not dict:
            assert len(indices) == 0 or type(indices[0]) in [int, np.int32, np.intp], "Trying to remove particles by their index, but the index type in the Python collection is not a 32-bit integer - invalid operation."
        else:
            assert len(list(indices.values())) == 0 or type(list(indices.values())[0]) in [int, np.int32, np.intp], "Trying to remove particles by their index, but the index type in the Python collection is not a 32-bit integer - invalid operation."

    @abstractmethod
    def remove_multi_by_IDs(self, ids):
        """
        This function removes particles from this collection based on their IDs. For collections where this removal
        strategy would require a collection transformation or by-ID parsing, it is advisable to rather apply a removal-
        by-objects or removal-by-indices scheme.
        """
        assert ids is not None, "Trying to remove particles by their IDs, but the ID list is None - invalid operation."
        assert type(ids) in [list, dict, np.ndarray], "Trying to remove particles by their IDs, but the ID container is not a valid Python-collection - invalid operation."
        if type(ids) is not dict:
            assert len(ids) == 0 or type(ids[0]) in [np.int64, np.uint64], "Trying to remove particles by their IDs, but the ID type in the Python collection is not a 64-bit (signed or unsigned) integer - invalid operation."
        else:
            assert len(list(ids.values())) == 0 or type(list(ids.values())[0]) in [np.int64, np.uint64], "Trying to remove particles by their IDs, but the ID type in the Python collection is not a 64-bit (signed or unsigned) integer - invalid operation."

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
        This function pops a Particle (as object or via its accessor) from a collection.

        This function removes the particle and then returns it.

        Comment/Annotation:
        Functions for popping multiple objects are more specialised than just a for-each loop of single-item pop,
        because certain data structures can pop multiple objects faster with specialised function than making a
        roundtrip per-item check-and-pop operation. Because of the sheer size of those containers and the resulting
        performance demands, we need to make use of those specialised 'pop' functions, where available.
        """
        if other is None:
            return
        if type(other) in [int, np.int32, np.intp]:
            return self.pop_single_by_index(other)
        elif type(other) in [np.int64, np.uint64]:
            return self.pop_single_by_ID(other)
        elif type(other) in [list, dict, np.ndarray]:
            # multi-removal routines - hard to discern at this point
            if type(other) is not dict:
                if len(other) == 0:
                    return None
                elif type(other[0]) in [int, np.int32, np.intp]:
                    return self.pop_multi_by_indices(other)
                elif type(other[0]) in [np.int64, np.uint64]:
                    return self.pop_multi_by_IDs(other)
            else:
                if len(list(other.values())) == 0:
                    return None
                elif type(list(other.values())[0]) in [int, np.int32, np.intp]:
                    return self.pop_multi_by_indices(other)
                elif type(list(other.values())[0]) in [np.int64, np.uint64]:
                    return self.pop_multi_by_IDs(other)

    @abstractmethod
    def pop_single_by_index(self, index):
        """
        Searches for Particle at index 'index', removes that Particle from the Collection and returns that Particle (or: ParticleAccessor).
        If index is None, return last item (-1);
        If index < 0: return from 'end' of collection.
        If index is out of bounds, throws and OutOfRangeException.
        If Particle cannot be retrieved, returns None.
        """
        assert type(index) in [int, np.int32, np.intp], "Trying to pop a particle by index, but index {} is not a 32-bit integer - invalid operation.".format(index)
        return None

    @abstractmethod
    def pop_single_by_ID(self, id):
        """
        Searches for Particle with ID 'id', removes that Particle from the Collection and returns that Particle (or: ParticleAccessor).
        If Particle cannot be retrieved (e.g. because the ID is not available), returns None.
        """
        assert type(id) in [np.int64, np.uint64], "Trying to pop a particle by ID, but ID {} is not a 64-bit (signed or unsigned) iteger - invalid operation.".format(id)
        return None

    @abstractmethod
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
            assert len(indices) == 0 or type(indices[0]) in [int, np.int32, np.intp], "Trying to pop particles by their index, but the index type in the Python collection is not a 32-bit integer - invalid operation."
        else:
            assert len(list(indices.values())) == 0 or type(list(indices.values())[0]) in [int, np.int32, np.intp], "Trying to pop particles by their index, but the index type in the Python collection is not a 32-bit integer - invalid operation."
        return None

    @abstractmethod
    def pop_multi_by_IDs(self, ids):
        """
        Searches for Particles with the IDs registered in 'ids', removes the Particles from the Collection and returns the Particles (or: their ParticleAccessors).
        If Particles cannot be retrieved (e.g. because the IDs are not available), returns None.
        """
        assert ids is not None, "Trying to pop particles by their IDs, but the ID list is None - invalid operation."
        assert type(ids) in [list, dict, np.ndarray], "Trying to pop particles by their IDs, but the ID container is not a valid Python-collection - invalid operation."
        if type(ids) is not dict:
            assert len(ids) == 0 or type(ids[0]) in [np.int64, np.uint64], "Trying to pop particles by their IDs, but the ID type in the Python collection is not a 64-bit (signed or unsigned) integer - invalid operation."
        else:
            assert len(list(ids.values())) == 0 or type(list(ids.values())[0]) in [np.int64, np.uint64], "Trying to pop particles by their IDs, but the ID type in the Python collection is not a 64-bit (signed or unsigned) integer - invalid operation."
        return None

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

    @abstractmethod
    def merge(self, same_class=None):
        """
        This function merge two strictly equally-structured ParticleCollections into one. This can be, for example,
        quite handy to merge two particle subsets that - due to continuous removal - become too small to be effective.

        On the other hand, this function can also internally merge individual particles that are tagged by status as
        being 'merged' (see the particle status for information on that).

        In order to distinguish both use cases, we can evaluate the 'same_class' parameter. In cases where this is
        'None', the merge operation semantically refers to an internal merge of individual particles - otherwise,
        it performs a 2-collection merge.

        Comment: the function can be simplified later by pre-evaluating the function parameter and then reference
        the individual, specific functions for internal- or external merge.

        The function shall return the merged ParticleCollection.
        """
        return None

    @abstractmethod
    def split(self, indices=None):
        """
        This function splits this collection into two disect equi-structured collections. The reason for it can, for
        example, be that the set exceeds a pre-defined maximum number of elements, which for performance reasons
        mandates a split.

        On the other hand, this function can also internally split individual particles that are tagged byt status as
        to be 'split' (see the particle status for information on that).

        In order to distinguish both use cases, we can evaluate the 'indices' parameter. In cases where this is
        'None', the split operation semantically refers to an internal split of individual particles - otherwise,
        it performs a collection-split.

        Comment: the function can be simplified later by pre-evaluating the function parameter and then reference
        the individual, specific functions for element- or collection split.

        The function shall return the newly created or extended Particle collection, i.e. either the collection that
        results from a collection split or this very collection, containing the newly-split particles.
        """
        return None

    def __str__(self):
        """
        This function returns and informative string about the collection (i.e. the type of collection) and a summary
        of its internal, distinct values.
        """
        return "ParticleCollection - N: {}".format(self._ncount)

    @abstractmethod
    def toArray(self):
        """
        This function converts (or: transforms; reformats; translates) this collection into an array-like structure
        (e.g. Python list or numpy nD array) that can be addressed by index. In the common case of 'no ID recovery',
        the global ID and the index match exactly.

        While this function may be very convenient for may users, it is STRONGLY DISADVISED to use the function to
        often, and the performance- and memory overhead malus may be exceed any speed-up one could get from optimised
        data structures - in fact, for large collections with an implicit-order structure (i.e. ordered lists, sets,
        trees, etc.), this may be 'the most constly' function in any kind of simulation.

        It can be - though - useful at the final stage of a simulation to dump the results to disk.
        """
        pass

    def __len__(self):
        """
        This function returns the length, in terms of 'number of elements, of a collection.
        """
        return self._ncount

    @abstractmethod
    def __sizeof__(self):
        """
        This function returns the size in actual bytes required in memory to hold the collection. Ideally and simply,
        the size is computed as follows:

        sizeof(self) = len(self) * sizeof(pclass)
        """
        pass

    def empty(self):
        """
        This function retuns a boolean value, expressing if a collection is emoty (i.e. does not [anymore] contain any
        elements) or not.
        """
        return (self._ncount < 1)

    @abstractmethod
    def clear(self):
        """
        This function physically removes all elements of the collection, yielding an empty collection as result of the
        operation.
        """
        pass


class ParticleCollection(Collection):
    _pu_indicators = None  # formerly: partitions
    _pu_centers = None
    _offset = 0
    _pclass = None
    _ptype = None
    _latlondepth_dtype = np.float32
    _data = None  # formerly: particle_data

    def __init__(self):
        """
        ParticleCollection - Constructor
        Initializes a particle collection by pre-allocating memory (where needed), initialising indexing structures
        (where needed), initialising iterators (if maintaining a persistent iterator) and preparing the C-JIT-glue.

        Furthermore, initialising the PU distribution and its distribution metrics.
        """
        self._ncount = -1
        self._pu_indicators = None  # formerly: partitions
        self._pu_centers = None
        self._offset = 0
        self._pclass = None
        self._ptype = None
        self._latlondepth_dtype = np.float32
        self._data = None  # formerly: particle_data
        super(ParticleCollection, self).__init__()

    def __del__(self):
        """
        ParticleCollection - Destructor
        """
        pass

    @property
    def pu_indicators(self):
        """
        The 'pu_indicator' is an [array or dictionary]-of-indicators, where each indicator entry tells per item
        (i.e. particle) in the collection to which processing unit (PU) in a parallelised setup it belongs to.
        """
        return self._pu_indicators

    @property
    def pu_centers(self):
        """
        The 'pu_centers" is an array of 2D/3D vectors storing the center of each cluster-of-particle partion that
        is handled by the respective PU. Storing the centers allows us to only run the initial kMeans segmentation
        once and then, on later particle additions, just (i) makes a closest-distance calculation, (ii) attaches the
        new particle to the closest cluster and (iii) updates the new cluster center. The last part may require at some
        point to merge overlaying clusters and them split them again in equi-sized partions.
        """
        return self._pu_centers

    @property
    def pclass(self):
        """
        'pclass' stores the actual class type of the particles allocated and managed in this collection
        """
        return self._pclass

    @property
    def ptype(self):
        """
        'ptype' returns an instance of the particular type of class 'ParticleType' of the particle class of the particles
        in this collection.

        basically:
        pytpe -> pclass().getPType()
        """
        return self._ptype

    @property
    def lonlatdepth_dtype(self):
        """
        'lonlatdepth_dtype' stores the numeric data type that is used to represent the lon, lat and depth of a particle.
        This can be either 'float32' (default) or 'float64'
        """
        return self._lonlatdepth_dtype

    @property
    def data(self):
        """
        'data' is a reference to the actual barebone-storage of the particle data, and thus depends directly on the
        specific collection in question.
        """
        return self._data

    @property
    def particle_data(self):
        """
        'particle_data' is a reference to the actual barebone-storage of the particle data, and thus depends directly on the
        specific collection in question. This property is just available for convenience and backward-compatibility, and
        this returns the same as 'data'.
        """
        return self._data

    @abstractmethod
    def cstruct(self):
        """
        'cstruct' returns the ctypes mapping of the particle data. This depends on the specific structure in question.
        """
        pass

    @abstractmethod
    def __getattr__(self, name):
        """
        Access a single property of all particles.
        NOTE: This is a fallback implementation, and it is NOT efficient.
        Specific datastructures may implement a more efficient variant.

        :param name: name of the property
        """
        for v in self.ptype.variables:
            if v.name == name:
                return np.array([getattr(p, name) for p in self], dtype=v.dtype)
        if name in self.__dict__ and name[0] != '_':
            return self.__dict__[name]
        else:
            return False

    @abstractmethod
    def toDictionary(self):
        """
        Convert all Particle data from one time step to a python dictionary.
        :param pfile: ParticleFile object requesting the conversion
        :param time: Time at which to write ParticleSet
        :param deleted_only: Flag to write only the deleted Particles
        returns two dictionaries: one for all variables to be written each outputdt,
         and one for all variables to be written once

         This function depends on the specific collection in question and thus needs to be specified in specific
         derivatives classes.
        """
        pass

    @abstractmethod
    def set_variable_write_status(self, var, write_status):
        """
        Method to set the write status of a Variable
        :param var: Name of the variable (string)
        :param status: Write status of the variable (True, False or 'once')

         This function depends on the specific collection in question and thus needs to be specified in specific
         derivatives classes.
        """
        pass

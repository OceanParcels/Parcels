from abc import ABC
from bisect import bisect_left
from ctypes import POINTER, Structure
from operator import attrgetter

import numpy as np

from parcels.particle import ScipyParticle
from parcels.tools.converters import convert_to_flat_array
from parcels.tools.loggers import logger
from parcels.tools.statuscodes import NotTestedError, StatusCode

try:
    from mpi4py import MPI
except:
    MPI = None
if MPI:
    try:
        from sklearn.cluster import KMeans
    except:
        KMeans = None


class ParticleCollection(ABC):

    def __init__(self, pclass, lon, lat, depth, time, lonlatdepth_dtype, pid_orig, partitions=None, ngrid=1, **kwargs):
        """
        Parameters
        ----------
        ngrid :
            number of grids in the fieldset of the overarching ParticleSet - required for initialising the
            field references of the ctypes-link of particles that are allocated
        """
        self._ncount = -1
        self._pu_indicators = None
        self._pu_centers = None
        self._offset = 0
        self._pclass = None
        self._ptype = None
        self._latlondepth_dtype = np.float32
        self._data = None

        assert pid_orig is not None, "particle IDs are None - incompatible with the collection. Invalid state."
        pid = pid_orig + pclass.lastID

        self._sorted = np.all(np.diff(pid) >= 0)

        assert depth is not None, "particle's initial depth is None - incompatible with the collection. Invalid state."
        assert lon.size == lat.size and lon.size == depth.size, (
            'lon, lat, depth don''t all have the same lenghts')

        assert lon.size == time.size, (
            'time and positions (lon, lat, depth) don''t have the same lengths.')

        # If partitions is false, the partitions are already initialised
        if partitions is not None and partitions is not False:
            self._pu_indicators = convert_to_flat_array(partitions)

        for kwvar in kwargs:
            assert lon.size == kwargs[kwvar].size, (
                f"{kwvar} and positions (lon, lat, depth) don't have the same lengths.")

        offset = np.max(pid) if (pid is not None) and len(pid) > 0 else -1
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            mpi_size = mpi_comm.Get_size()

            if lon.size < mpi_size and mpi_size > 1:
                raise RuntimeError('Cannot initialise with fewer particles than MPI processors')

            if mpi_size > 1:
                if partitions is not False:
                    if (self._pu_indicators is None): # or (len(self._pu_indicators) != len(lon)):
                        if mpi_rank == 0:
                            coords = np.vstack((lon, lat)).transpose()
                            if KMeans:
                                kmeans = KMeans(n_clusters=mpi_size, random_state=0).fit(coords)
                                self._pu_indicators = kmeans.labels_
                            else:  # assigning random labels if no KMeans (see https://github.com/OceanParcels/parcels/issues/1261)
                                logger.warning_once('sklearn needs to be available if MPI is installed. '
                                                    'See https://docs.oceanparcels.org/en/latest/installation.html#installation-for-developers for more information')
                                self._pu_indicators = np.randint(0, mpi_size, size=len(lon))
                        else:
                            self._pu_indicators = None
                        self._pu_indicators = mpi_comm.bcast(self._pu_indicators, root=0)
                    elif np.max(self._pu_indicators) >= mpi_size:
                        raise RuntimeError('Particle partitions must vary between 0 and the number of mpi procs')
                    lon = lon[self._pu_indicators == mpi_rank]
                    lat = lat[self._pu_indicators == mpi_rank]
                    time = time[self._pu_indicators == mpi_rank]
                    depth = depth[self._pu_indicators == mpi_rank]
                    pid = pid[self._pu_indicators == mpi_rank]
                    for kwvar in kwargs:
                        kwargs[kwvar] = kwargs[kwvar][self._pu_indicators == mpi_rank]
                offset = MPI.COMM_WORLD.allreduce(offset, op=MPI.MAX)

        pclass.setLastID(offset+1)

        if lonlatdepth_dtype is None:
            self._lonlatdepth_dtype = np.float32
        else:
            self._lonlatdepth_dtype = lonlatdepth_dtype
        assert self._lonlatdepth_dtype in [np.float32, np.float64], \
            'lon lat depth precision should be set to either np.float32 or np.float64'
        pclass.set_lonlatdepth_dtype(self._lonlatdepth_dtype)
        self._pclass = pclass

        self._ptype = pclass.getPType()
        self._data = {}
        initialised = set()

        self._ncount = len(lon)

        for v in self.ptype.variables:
            if v.name in ['xi', 'yi', 'zi', 'ti']:
                self._data[v.name] = np.empty((len(lon), ngrid), dtype=v.dtype)
            else:
                self._data[v.name] = np.empty(self._ncount, dtype=v.dtype)

        if lon is not None and lat is not None:
            # Initialise from lists of lon/lat coordinates
            assert self.ncount == len(lon) and self.ncount == len(lat), (
                'Size of ParticleSet does not match length of lon and lat.')

            # mimic the variables that get initialised in the constructor
            self._data['lat'][:] = lat
            self._data['lat_nextloop'][:] = lat
            self._data['lon'][:] = lon
            self._data['lon_nextloop'][:] = lon
            self._data['depth'][:] = depth
            self._data['depth_nextloop'][:] = depth
            self._data['time'][:] = time
            self._data['time_nextloop'][:] = time
            self._data['id'][:] = pid
            self._data['obs_written'][:] = 0

            # special case for exceptions which can only be handled from scipy
            self._data['exception'] = np.empty(self.ncount, dtype=object)

            initialised |= {'lat', 'lat_nextloop', 'lon', 'lon_nextloop', 'depth', 'depth_nextloop', 'time', 'time_nextloop', 'id', 'obs_written'}

            # any fields that were provided on the command line
            for kwvar, kwval in kwargs.items():
                if not hasattr(pclass, kwvar):
                    raise RuntimeError(f'Particle class does not have Variable {kwvar}')
                self._data[kwvar][:] = kwval
                initialised.add(kwvar)

            # initialise the rest to their default values
            for v in self.ptype.variables:
                if v.name in initialised:
                    continue

                if isinstance(v.initial, attrgetter):
                    self._data[v.name][:] = v.initial(self)
                else:
                    self._data[v.name][:] = v.initial

                initialised.add(v.name)
        else:
            raise ValueError("Latitude and longitude required for generating ParticleSet")
        self._iterator = None
        self._riterator = None

    def __del__(self):
        """Collection - Destructor"""
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
        """Stores the actual class type of the particles allocated and managed in this collection."""
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

    @property
    def ncount(self):
        return self._ncount

    def __len__(self):
        """This function returns the length, in terms of 'number of elements, of a collection."""
        return self._ncount

    def empty(self):
        """
        This function retuns a boolean value, expressing if a collection is emoty (i.e. does not [anymore] contain any
        elements) or not.
        """
        return (self._ncount < 1)

    def iterator(self):
        self._iterator = ParticleCollectionIterator(self)
        return self._iterator

    def __iter__(self):
        """Returns an Iterator that allows for forward iteration over the
        elements in the ParticleCollection (e.g. `for p in pset:`).
        """
        return self.iterator()

    def reverse_iterator(self):
        self._riterator = ParticleCollectionIterator(self, True)
        return self._riterator

    def __reversed__(self):
        """Returns an Iterator that allows for backwards iteration over
        the elements in the ParticleCollection (e.g.
        `for p in reversed(pset):`).
        """
        return self.reverse_iterator()

    def __getitem__(self, index):
        """
        Access a particle in this collection using the fastest access
        method for this collection - by its index.

        Parameters
        ----------
        index : int
            Index of the particle to access
        """
        return self.get_single_by_index(index)

    def __getattr__(self, name):
        """
        Access a single property of all particles.

        Parameters
        ----------
        name : str
            Name of the property to access
        """
        for v in self.ptype.variables:
            if v.name == name and name in self._data:
                return self._data[name]
        return False

    def get_single_by_index(self, index):
        """
        This function gets a (particle) object from the collection based on its index within the collection. For
        collections that are not based on random access (e.g. ordered lists, sets, trees), this function involves a
        translation of the index into the specific object reference in the collection - or (if unavoidable) the
        translation of the collection from a none-indexable, none-random-access structure into an indexable structure.
        In cases where a get-by-index would result in a performance malus, it is highly-advisable to use a different
        get function, e.g. get-by-ID.
        """
        assert type(index) in [int, np.int32, np.intp], f"Trying to get a particle by index, but index {index} is not a 32-bit integer - invalid operation."
        return ParticleAccessor(self, index)

    def get_single_by_object(self, particle_obj):
        """
        This function gets a (particle) object from the collection based on its actual object. For collections that
        are random-access and based on indices (e.g. unordered list, vectors, arrays and dense matrices), this function
        would involve a parsing of the whole list and translation of the object into an index in the collection - which
        results in a significant performance malus.
        In cases where a get-by-object would result in a performance malus, it is highly-advisable to use a different
        get function, e.g. get-by-index or get-by-ID.

        In this specific implementation, we cannot look for the object
        directly, so we will look for one of its properties (the ID) that
        has the nice property of being stored in an ordered list (if the
        collection is sorted)
        """
        assert (isinstance(particle_obj, ParticleAccessor) or isinstance(particle_obj, ScipyParticle))
        return self.get_single_by_ID(particle_obj.id)

    def get_single_by_ID(self, id):
        """
        This function gets a (particle) object from the collection based on the object's ID. For some collections,
        this operation may involve a parsing of the whole list and translation of the object's ID into an index  or an
        object reference in the collection - which results in a significant performance malus.
        In cases where a get-by-ID would result in a performance malus, it is highly-advisable to use a different
        get function, e.g. get-by-index.

        This function uses binary search if we know the ID list to be sorted, and linear search otherwise. We assume
        IDs are unique.
        """
        assert type(id) in [np.int64, np.uint64], f"Trying to get a particle by ID, but ID {id} is not a 64-bit (signed or unsigned) iteger - invalid operation."

        # Use binary search if the collection is sorted, linear search otherwise
        index = -1
        if self._sorted:
            index = bisect_left(self._data['id'], id)
            if index == len(self._data['id']) or self._data['id'][index] != id:
                raise ValueError(f"Trying to access a particle with a non-existing ID: {id}.")
        else:
            index = np.where(self._data['id'] == id)[0][0]

        return self.get_single_by_index(index)

    def get_same(self, same_class):
        """
        This function gets particles from this collection that are themselves stored in another object of an equi-
        structured ParticleCollection.
        """
        assert same_class is not None, f"Trying to get another {type(self)} from this one, but the other one is None - invalid operation."
        assert type(same_class) is type(self)
        raise NotImplementedError

    def get_collection(self, pcollection):
        """
        This function gets particles from this collection that are themselves stored in a ParticleCollection, which
        is differently structured than this one. That means the other-collection has to be re-formatted first in an
        intermediary format.
        """
        assert pcollection is not None, "Trying to get another particle collection from this one, but the other one is None - invalid operation."
        assert isinstance(pcollection, ParticleCollection), "Trying to get another particle collection from this one, but the other is not of the type of 'ParticleCollection' - invalid operation."
        assert type(pcollection) is not type(self)
        raise NotImplementedError

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
        raise NotImplementedError

    def get_multi_by_indices(self, indices):
        """
        This function gets particles from this collection based on their indices. This works best for random-access
        collections (e.g. numpy's ndarrays, dense matrices and dense arrays), whereas internally ordered collections
        shall rather use a get-via-object-reference strategy.
        """
        raise NotTestedError
        # assert indices is not None, "Trying to get particles by their collection indices, but the index list is None - invalid operation."
        # assert type(indices) in [list, dict, np.ndarray], "Trying to get particles by their IDs, but the ID container is not a valid Python-collection - invalid operation."
        # if type(indices) is not dict:
        #     assert len(indices) == 0 or type(indices[0]) in [int, np.int32, np.intp], "Trying to get particles by their index, but the index type in the Python collection is not a 32-bit integer - invalid operation."
        # else:
        #     assert len(list(indices.values())) == 0 or type(list(indices.values())[0]) in [int, np.int32, np.intp], "Trying to get particles by their index, but the index type in the Python collection is not a 32-bit integer - invalid operation."
        # if type(indices) is dict:
        #     indices = list(indices.values())
        # return ParticleCollectionIterator(self, subset=indices)

    def get_multi_by_IDs(self, ids):
        """
        This function gets particles from this collection based on their IDs. For collections where this removal
        strategy would require a collection transformation or by-ID parsing, it is advisable to rather apply a get-
        by-objects or get-by-indices scheme.

        Note that this implementation assumes that IDs of particles are strictly increasing with increasing index. So
        a particle with a larger index will always have a larger ID as well. The assumption often holds for this
        datastructure as new particles always get a larger ID than any existing particle (IDs are not recycled)
        and their data are appended at the end of the list (largest index). This allows for the use of binary search
        in the look-up. The collection maintains a `sorted` flag to indicate whether this assumption holds.
        """
        raise NotTestedError
        # assert ids is not None, "Trying to get particles by their IDs, but the ID list is None - invalid operation."
        # assert type(ids) in [list, dict, np.ndarray], "Trying to get particles by their IDs, but the ID container is not a valid Python-collection - invalid operation."
        # if type(ids) is not dict:
        #     assert len(ids) == 0 or type(ids[0]) in [np.int64, np.uint64], "Trying to get particles by their IDs, but the ID type in the Python collection is not a 64-bit (signed or unsigned) integer - invalid operation."
        # else:
        #     assert len(list(ids.values())) == 0 or type(list(ids.values())[0]) in [np.int64, np.uint64], "Trying to get particles by their IDs, but the ID type in the Python collection is not a 64-bit (signed or unsigned) integer - invalid operation."
        # if type(ids) is dict:
        #     ids = list(ids.values())
        #
        # if len(ids) == 0:
        #     return None
        #
        # # Use binary search if the collection is sorted, linear search otherwise
        # indices = np.empty(len(ids), dtype=np.int32)
        # if self._sorted:
        #     # This is efficient if len(ids) << self.len
        #     sorted_ids = np.sort(np.array(ids))
        #     indices = self._recursive_ID_lookup(0, len(self._data['id']), sorted_ids)
        # else:
        #     indices = np.where(np.in1d(self._data['id'], ids))[0]
        #
        # return self.get_multi_by_indices(indices)

    def _recursive_ID_lookup(self, low, high, sublist):
        """Identify the middle element of the sublist and perform binary
        search on it.

        Parameters
        ----------
        low :
            Lowerbound on the indices to search for IDs.
        high :
            Upperbound on the indices to search for IDs.
        sublist : list
            Sublist of IDs to look for.
        """
        raise NotTestedError
        # median = floor(len(sublist) / 2)
        # index = bisect_left(self._data['id'][low:high], sublist[median])
        # if len(sublist) == 1:
        #     # edge case
        #     if index == len(self._data['id']) or \
        #        self._data['id'][index] != sublist[median]:
        #         return np.array([])
        #     return np.array([index])
        #
        # # The edge-cases have to be handled slightly differently
        # if index == len(self._data['id']):
        #     # Continue with the same bounds, but drop the median.
        #     return self._recursive_ID_lookup(low, high, np.delete(sublist, median))
        # elif self._data['id'][index] != sublist[median]:
        #     # We can split, because we received the index that the median
        #     # ID would have been inserted in, but we do not return the
        #     # index and keep it in our search space.
        #     left = self._recursive_ID_lookup(low, index, sublist[:median])
        #     right = self._recursive_ID_lookup(index, high, sublist[median + 1:])
        #     return np.concatenate((left, right))
        #
        # # Otherwise, we located the median, so we include it in our
        # # result, and split the search space on it, without including it.
        # left = self._recursive_ID_lookup(low, index, sublist[:median])
        # right = self._recursive_ID_lookup(index + 1, high, sublist[median + 1:])
        # return np.concatenate((left, np.array(index), right))

    def add_collection(self, pcollection):
        """
        Adds another, differently structured ParticleCollection to this collection. This is done by, for example,
        appending/adding the items of the other collection to this collection.
        """
        assert pcollection is not None, "Trying to add another particle collection to this one, but the other one is None - invalid operation."
        assert isinstance(pcollection, ParticleCollection), "Trying to add another particle collection to this one, but the other is not of the type of 'ParticleCollection' - invalid operation."
        assert type(pcollection) is not type(self)
        raise NotImplementedError

    def add_single(self, particle_obj):
        """
        Adding a single Particle to the collection - either as a 'Particle; object in parcels itself, or
        via its ParticleAccessor.
        """
        assert (isinstance(particle_obj, ParticleAccessor) or isinstance(particle_obj, ScipyParticle))
        raise NotImplementedError

    def add_same(self, same_class):
        """
        Adds another, equi-structured ParticleCollection to this collection. This is done by concatenating
        both collections. The fact that they are of the same ParticleCollection's derivative simplifies
        parsing and concatenation.
        """
        assert same_class is not None, f"Trying to add another {type(self)} to this one, but the other one is None - invalid operation."
        assert type(same_class) is type(self)

        if same_class.ncount == 0:
            return

        if self._ncount == 0:
            self._data = same_class._data
            self._ncount = same_class.ncount
            return

        # Determine order of concatenation and update the sorted flag
        if self._sorted and same_class._sorted \
           and self._data['id'][0] > same_class._data['id'][-1]:
            for d in self._data:
                self._data[d] = np.concatenate((same_class._data[d], self._data[d]))
            self._ncount += same_class.ncount
        else:
            if not (same_class._sorted
                    and self._data['id'][-1] < same_class._data['id'][0]):
                self._sorted = False
            for d in self._data:
                self._data[d] = np.concatenate((self._data[d], same_class._data[d]))
            self._ncount += same_class.ncount

    def __iadd__(self, same_class):
        """
        Performs an incremental addition of the equi-structured ParticleCollections, such to allow

        a += b,

        with 'a' and 'b' begin the two equi-structured objects (or: 'b' being and individual object).
        This operation is equal to an in-place addition of (an) element(s).
        """
        assert same_class is not None
        assert type(same_class) is type(self), f"Trying to increment-add collection of type {type(same_class)} into collection of type {type(self)} - invalid operation."
        self.add_same(same_class)
        return self

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
        raise NotImplementedError

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
        raise NotImplementedError

    def append(self, particle_obj):
        """
        This function appends a Particle (as object or via its accessor) to the end of a collection ('end' definition
        depends on the specific collection itself). For collections with an inherent indexing order (e.g. ordered lists,
        sets, trees), the function just includes the object at its pre-defined position (i.e. not necessarily at the
        end). For the collections, the function mapping equates to:

        append(particle_obj) -> add_single(particle_obj)

        The function - in contrast to 'push' - does not return the index of the inserted object.
        """
        raise NotImplementedError

    def __delitem__(self, key):
        """
        This is the high-performance method to delete a specific object from this collection.
        As the most-performant way depends on the specific collection in question, the function is abstract.

        Highlight for the specific implementation:
        The 'key' parameter should still be evaluated for being a single or a multi-entry delete, and needs to check
        that it received the correct type of 'indexing' argument (i.e. index, id or iterator).
        """
        self.delete_by_index(key)

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

    def delete_by_index(self, index):
        """
        This method deletes a particle from the  the collection based on its index. It does not return the deleted item.
        Semantically, the function appears similar to the 'remove' operation. That said, the function in OceanParcels -
        instead of directly deleting the particle - just raises the 'deleted' status flag for the indexed particle.
        In result, the particle still remains in the collection.
        """
        raise NotTestedError
        # assert type(index) in [int, np.int32, np.intp], f"Trying to delete a particle by index, but index {index} is not a 32-bit integer - invalid operation."
        #
        # self._data['state'][index] = StatusCode.Delete

    def delete_by_ID(self, id):
        """
        This method deletes a particle from the  the collection based on its ID. It does not return the deleted item.
        Semantically, the function appears similar to the 'remove' operation. That said, the function in OceanParcels -
        instead of directly deleting the particle - just raises the 'deleted' status flag for the indexed particle.
        In result, the particle still remains in the collection.
        """
        raise NotTestedError
        # assert type(id) in [np.int64, np.uint64], f"Trying to delete a particle by ID, but ID {id} is not a 64-bit (signed or unsigned) integer - invalid operation."
        #
        # # Use binary search if the collection is sorted, linear search otherwise
        # index = -1
        # if self._sorted:
        #     index = bisect_left(self._data['id'], id)
        #     if index == len(self._data['id']) or \
        #        self._data['id'][index] != id:
        #         raise ValueError("Trying to delete a particle with a non-existing ID: %s." % id)
        # else:
        #     index = np.where(self._data['id'] == id)[0][0]
        #
        # self.delete_by_index(index)

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
        assert type(index) in [int, np.int32, np.intp], f"Trying to remove a particle by index, but index {index} is not a 32-bit integer - invalid operation."

        for d in self._data:
            self._data[d] = np.delete(self._data[d], index, axis=0)

        self._ncount -= 1

    def remove_single_by_object(self, particle_obj):
        """
        This function removes a (particle) object from the collection based on its actual object. For collections that
        are random-access and based on indices (e.g. unordered list, vectors, arrays and dense matrices), this function
        would involves a parsing of the whole list and translation of the object into an index in the collection to
        perform the removal - which results in a significant performance malus.
        In cases where a removal-by-object would result in a performance malus, it is highly-advisable to use a different
        removal functions, e.g. remove-by-index or remove-by-ID.
        """
        assert (isinstance(particle_obj, ParticleAccessor) or isinstance(particle_obj, ScipyParticle))

        # We cannot look for the object directly, so we will look for one of
        # its properties that has the nice property of being stored in an
        # ordered list
        self.remove_single_by_ID(particle_obj.id)

    def remove_single_by_ID(self, id):
        """
        This function removes a (particle) object from the collection based on the object's ID. For some collections,
        this operation may involve a parsing of the whole list and translation of the object's ID into an index  or an
        object reference in the collection in order to perform the removal - which results in a significant performance
        malus.
        In cases where a removal-by-ID would result in a performance malus, it is highly-advisable to use a different
        removal functions, e.g. remove-by-object or remove-by-index.
        """
        raise NotTestedError
        # assert type(id) in [np.int64, np.uint64], f"Trying to remove a particle by ID, but ID {id} is not a 64-bit (signed or unsigned) iteger - invalid operation."
        #
        # # Use binary search if the collection is sorted, linear search otherwise
        # index = -1
        # if self._sorted:
        #     index = bisect_left(self._data['id'], id)
        #     if index == len(self._data['id']) or \
        #        self._data['id'][index] != id:
        #         raise ValueError("Trying to remove a particle with a non-existing ID: %s." % id)
        # else:
        #     index = np.where(self._data['id'] == id)[0][0]
        #
        # self.remove_single_by_index(index)

    def remove_same(self, same_class):
        """
        This function removes particles from this collection that are themselves stored in another object of an equi-
        structured ParticleCollection. As the structures of both collections are the same, a more efficient M-in-N
        removal can be applied without an in-between reformatting.
        """
        assert same_class is not None, f"Trying to remove another {type(self)} from this one, but the other one is None - invalid operation."
        assert type(same_class) is type(self)
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        if type(indices) is dict:
            indices = list(indices.values())

        for d in self._data:
            self._data[d] = np.delete(self._data[d], indices, axis=0)

        self._ncount -= len(indices)

    def remove_multi_by_IDs(self, ids):
        """
        This function removes particles from this collection based on their IDs. For collections where this removal
        strategy would require a collection transformation or by-ID parsing, it is advisable to rather apply a removal-
        by-objects or removal-by-indices scheme.
        """
        raise NotTestedError
        # assert ids is not None, "Trying to remove particles by their IDs, but the ID list is None - invalid operation."
        # assert type(ids) in [list, dict, np.ndarray], "Trying to remove particles by their IDs, but the ID container is not a valid Python-collection - invalid operation."
        # if type(ids) is not dict:
        #     assert len(ids) == 0 or type(ids[0]) in [np.int64, np.uint64], "Trying to remove particles by their IDs, but the ID type in the Python collection is not a 64-bit (signed or unsigned) integer - invalid operation."
        # else:
        #     assert len(list(ids.values())) == 0 or type(list(ids.values())[0]) in [np.int64, np.uint64], "Trying to remove particles by their IDs, but the ID type in the Python collection is not a 64-bit (signed or unsigned) integer - invalid operation."
        # if type(ids) is dict:
        #     ids = list(ids.values())
        #
        # if len(ids) == 0:
        #     return
        #
        # # Use binary search if the collection is sorted, linear search otherwise
        # indices = np.empty(len(ids), dtype=np.int32)
        # if self._sorted:
        #     # This is efficient if len(ids) << self.len
        #     sorted_ids = np.sort(np.array(ids))
        #     indices = self._recursive_ID_lookup(0, len(self._data['id']), sorted_ids)
        # else:
        #     indices = np.where(np.in1d(self._data['id'], ids))[0]
        #
        # self.remove_multi_by_indices(indices)

    def __isub__(self, other):
        """
        This method performs an incremental removal of the equi-structured ParticleCollections, such to allow

        a -= b,

        with 'a' and 'b' begin the two equi-structured objects (or: 'b' being and individual object).
        This operation is equal to an in-place removal of (an) element(s).
        """
        raise NotTestedError
        # if other is None:
        #     return
        # if type(other) is type(self):
        #     self.remove_same(other)
        # elif (isinstance(other, ParticleAccessor)
        #       or isinstance(other, ScipyParticle)):
        #     self.remove_single_by_object(other)
        # else:
        #     raise TypeError("Trying to do an incremental removal of an element of type %s, which is not supported." % type(other))
        # return self

    def pop_single_by_index(self, index):
        """
        Searches for Particle at index 'index', removes that Particle from the Collection and returns that Particle (or: ParticleAccessor).
        If index is None, return last item (-1);
        If index < 0: return from 'end' of collection.
        If index is out of bounds, throws and OutOfRangeException.
        If Particle cannot be retrieved, returns None.
        """
        assert type(index) in [int, np.int32, np.intp], f"Trying to pop a particle by index, but index {index} is not a 32-bit integer - invalid operation."
        raise NotImplementedError

    def pop_single_by_ID(self, id):
        """
        Searches for Particle with ID 'id', removes that Particle from the Collection and returns that Particle (or: ParticleAccessor).
        If Particle cannot be retrieved (e.g. because the ID is not available), returns None.
        """
        assert type(id) in [np.int64, np.uint64], f"Trying to pop a particle by ID, but ID {id} is not a 64-bit (signed or unsigned) iteger - invalid operation."
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def _clear_deleted_(self):
        """
        This (protected) function physically removes particles from the collection whose status is set to 'DELETE'.
        It is the logical finalisation method of physically deleting particles that have been marked for deletion and
        that have not otherwise been recovered.
        This methods in heavily dependent on the actual collection type and should be implemented very specific
        to the actual data structure, to remove objects 'the fastest way possible'.
        """
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def __str__(self):
        """
        This function returns and informative string about the collection (i.e. the type of collection) and a summary
        of its internal, distinct values.
        """
        return f"ParticleCollection - N: {self._ncount}"

    def __sizeof__(self):
        """
        This function returns the size in actual bytes required in memory to hold the collection. Ideally and simply,
        the size is computed as follows:

        sizeof(self) = len(self) * sizeof(pclass)
        """
        raise NotImplementedError

    def clear(self):
        """
        This function physically removes all elements of the collection, yielding an empty collection as result of the
        operation.
        """
        raise NotImplementedError

    def cstruct(self):
        """Returns the ctypes mapping of the particle data. This depends on the specific structure in question."""
        class CParticles(Structure):
            _fields_ = [(v.name, POINTER(np.ctypeslib.as_ctypes_type(v.dtype))) for v in self._ptype.variables]

        def flatten_dense_data_array(vname):
            data_flat = self._data[vname].view()
            data_flat.shape = -1
            return np.ctypeslib.as_ctypes(data_flat)

        cdata = [flatten_dense_data_array(v.name) for v in self._ptype.variables]
        cstruct = CParticles(*cdata)
        return cstruct

    def _to_write_particles(self, pd, time):
        """We don't want to write a particle that is not started yet.
        Particle will be written if particle.time is between time-dt/2 and time+dt (/2)
        """
        return np.where((np.less_equal(time - np.abs(pd['dt'] / 2), pd['time'], where=np.isfinite(pd['time']))
                        & np.greater_equal(time + np.abs(pd['dt'] / 2), pd['time'], where=np.isfinite(pd['time']))
                        | ((np.isnan(pd['dt'])) & np.equal(time, pd['time'], where=np.isfinite(pd['time']))))
                        & (np.isfinite(pd['id']))
                        & (np.isfinite(pd['time'])))[0]

    def getvardata(self, var, indices=None):
        if indices is None:
            return self._data[var]
        else:
            try:
                return self._data[var][indices]
            except:  # Can occur for zero-length ParticleSets
                return None

    def setvardata(self, var, index, val):
        self._data[var][index] = val

    def setallvardata(self, var, val):
        self._data[var][:] = val

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
        raise NotImplementedError

    def set_variable_write_status(self, var, write_status):
        """Method to set the write status of a Variable

        Parameters
        ----------
        var :
            Name of the variable (string)
        status :
            Write status of the variable (True, False or 'once')
        write_status :
        """
        var_changed = False
        for v in self._ptype.variables:
            if v.name == var:
                v.to_write = write_status
                var_changed = True
        if not var_changed:
            raise SyntaxError(f'Could not change the write status of {var}, because it is not a Variable name')


class ParticleAccessor(ABC):
    """Wrapper that provides access to particle data in the collection,
    as if interacting with the particle itself.

    Parameters
    ----------
    pcoll :
        ParticleCollection that the represented particle
        belongs to.
    index :
        The index at which the data for the represented
        particle is stored in the corresponding data arrays
        of the ParticleCollecion.
    """

    _pcoll = None
    _index = 0

    def __init__(self, pcoll, index):
        """Initializes the ParticleAccessor to provide access to one
        specific particle.
        """
        self._pcoll = pcoll
        self._index = index

    def __getattr__(self, name):
        """
        Get the value of an attribute of the particle.

        Parameters
        ----------
        name : str
            Name of the requested particle attribute.

        Returns
        -------
        any
            The value of the particle attribute in the underlying collection data array.
        """
        if name in self.__dict__.keys():
            result = self.__getattribute__(name)
        elif name in type(self).__dict__.keys():
            result = object.__getattribute__(self, name)
        else:
            result = self._pcoll.data[name][self._index]
        return result

    def __setattr__(self, name, value):
        """
        Set the value of an attribute of the particle.

        Parameters
        ----------
        name : str
            Name of the particle attribute.
        value : any
            Value that will be assigned to the particle attribute in the underlying collection data array.
        """
        if name in self.__dict__.keys():
            self.__setattr__(name, value)
        elif name in type(self).__dict__.keys():
            object.__setattr__(self, name, value)
        else:
            self._pcoll.data[name][self._index] = value

    def getPType(self):
        return self._pcoll.ptype

    def __repr__(self):
        time_string = 'not_yet_set' if self.time is None or np.isnan(self.time) else f"{self.time:f}"
        str = "P[%d](lon=%f, lat=%f, depth=%f, " % (self.id, self.lon, self.lat, self.depth)
        for var in self._pcoll.ptype.variables:
            if var.name in ['lon_nextloop', 'lat_nextloop', 'depth_nextloop', 'time_nextloop']:  # TODO check if time_nextloop is needed (or can work with time-dt?)
                continue
            if var.to_write is not False and var.name not in ['id', 'lon', 'lat', 'depth', 'time']:
                str += f"{var.name}={getattr(self, var.name):f}, "
        return str + f"time={time_string})"

    def delete(self):
        """Signal the underlying particle for deletion."""
        self.state = StatusCode.Delete

    def set_state(self, state):
        """Syntactic sugar for changing the state of the underlying particle."""
        self.state = state

    def succeeded(self):
        self.state = StatusCode.Success

    def isComputed(self):
        return self.state == StatusCode.Success

    def reset_state(self):
        self.state = StatusCode.Evaluate


class ParticleCollectionIterable(ABC):

    def __init__(self, pcoll, reverse=False, subset=None):
        self._pcoll_immutable = pcoll
        self._reverse = reverse
        self._subset = subset

    def __iter__(self):
        return ParticleCollectionIterator(pcoll=self._pcoll_immutable, reverse=self._reverse, subset=self._subset)

    def __len__(self):
        """Implementation needed for particle-particle interaction"""
        return len(self._subset)

    def __getitem__(self, items):
        """Implementation needed for particle-particle interaction"""
        return ParticleAccessor(self._pcoll_immutable, self._subset[items])


class ParticleCollectionIterator(ABC):
    """Iterator for looping over the particles in the ParticleCollection.

    Parameters
    ----------
    pcoll :
        ParticleCollection that stores the particles.
    reverse :
        Flag to indicate reverse iteration (i.e. starting at
        the largest index, instead of the smallest).
    subset :
        Subset of indices to iterate over, this allows the
        creation of an iterator that represents part of the
        collection.
    """

    def __init__(self, pcoll, reverse=False, subset=None):

        if subset is not None:
            if len(subset) > 0 and type(subset[0]) not in [int, np.int32, np.intp]:
                raise TypeError("Iteration over a subset of particles in the"
                                " particleset requires a list or numpy array"
                                " of indices (of type int or np.int32).")
            if reverse:
                self._indices = subset.reverse()
            else:
                self._indices = subset
            self.max_len = len(subset)
        else:
            self.max_len = len(pcoll)
            if reverse:
                self._indices = range(self.max_len - 1, -1, -1)
            else:
                self._indices = range(self.max_len)

        self._reverse = reverse
        self._pcoll = pcoll
        self._index = 0
        self._head = None
        self._tail = None
        if len(self._indices) > 0:
            self._head = ParticleAccessor(pcoll, self._indices[0])
            self._tail = ParticleAccessor(pcoll, self._indices[self.max_len - 1])
        self.p = self._head

    def __next__(self):
        """Returns a ParticleAccessor for the next particle in the
        ParticleSet.
        """
        if self._index < self.max_len:
            self.p = ParticleAccessor(self._pcoll, self._indices[self._index])
            self._index += 1
            return self.p

        raise StopIteration

    @property
    def head(self):
        """Returns a ParticleAccessor for the first particle in the ParticleSet."""
        return self._head

    @property
    def tail(self):
        """Returns a ParticleAccessor for the last particle in the ParticleSet."""
        return self._tail

    @property
    def current(self):
        return self.p

    def __repr__(self):
        dir_str = 'Backward' if self._reverse else 'Forward'
        return f"{dir_str} iteration at index {self._index} of {self.max_len}."

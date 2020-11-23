import numpy as np

from .collections import ParticleCollection
from .iterators import BaseParticleAccessor
from .iterators import BaseParticleCollectionIterator
from parcels.particle import ScipyParticle, JITParticle  # noqa
from parcels.field import Field
from parcels.tools.loggers import logger
from parcels.tools.statuscodes import OperationCode
from operator import attrgetter
from ctypes import Structure, POINTER
from bisect import bisect_left
from math import floor

try:
    from mpi4py import MPI
except:
    MPI = None
if MPI:
    try:
        from sklearn.cluster import KMeans
    except:
        raise EnvironmentError('sklearn needs to be available if MPI is installed. '
                               'See http://oceanparcels.org/#parallel_install for more information')

"""
Author: Dr. Christian Kehl
github relation: #913 (particleset_class_hierarchy)
purpose: defines all the specific functions for a ParticleCollection, ParticleAccessor, ParticleSet etc. that relates
         to a structure-of-array (SoA) data arrangement.
"""


def convert_to_flat_array(var):
    # Convert lists and single integers/floats to one-dimensional numpy arrays
    if isinstance(var, np.ndarray):
        return var.flatten()
    elif isinstance(var, (int, float, np.float32, np.int32)):
        return np.array([var])
    else:
        return np.array(var)


class ParticleCollectionSOA(ParticleCollection):

    # ==== already user-exposed ==== #
    def __init__(self, pclass, lon, lat, depth, time, lonlatdepth_dtype, partitions=None, pid_orig=None, ngrid=1, **kwargs):
        """
        :param ngrid: number of grids in the fieldset of the overarching ParticleSet - required for initialising the
        field references of the ctypes-link of particles that are allocated
        """

        super(ParticleCollection, self).__init__()
        # partitions = kwargs.pop('partitions', None)

        # lon = np.empty(shape=0) if lon is None else convert_to_flat_array(lon)  # input reformatting - particleset-task
        # lat = np.empty(shape=0) if lat is None else convert_to_flat_array(lat)  # input reformatting - particleset-task
        if isinstance(pid_orig, (type(None), type(False))):
            pid_orig = np.arange(lon.size)
        pid = pid_orig + pclass.lastID

        self._sorted = np.all(np.diff(pid) >= 0)

        # -- We expect the overarching particle set to take care of the depth-is-none-so-calc-from-field case -- #
        # if depth is None:
        #     mindepth = self.fieldset.gridset.dimrange('depth')[0] if self.fieldset is not None else 0
        #     depth = np.ones(lon.size) * mindepth
        # else:
        #     depth = convert_to_array(depth)
        assert depth is not None, "particle's initial depth is None - incompatible with the collection. Invalid state."
        # depth = convert_to_flat_array(depth)  # input reformatting - particleset-task
        assert lon.size == lat.size and lon.size == depth.size, (
            'lon, lat, depth don''t all have the same lenghts')

        # time = convert_to_flat_array(time)  # input reformatting - particleset-task
        # time = np.repeat(time, lon.size) if time.size == 1 else time  # input reformatting - particleset-task

        # -- Time field correction to be done in overarching particle set -- #
        # def _convert_to_reltime(time):
        #     if isinstance(time, np.datetime64) or (hasattr(time, 'calendar') and time.calendar in _get_cftime_calendars()):
        #         return True
        #     return False

        # if time.size > 0 and type(time[0]) in [datetime, date]:
        #     time = np.array([np.datetime64(t) for t in time])
        # self.time_origin = fieldset.time_origin if self.fieldset is not None else 0
        # if time.size > 0 and isinstance(time[0], np.timedelta64) and not self.time_origin:
        #     raise NotImplementedError('If fieldset.time_origin is not a date, time of a particle must be a double')
        # time = np.array([self.time_origin.reltime(t) if _convert_to_reltime(t) else t for t in time])

        assert lon.size == time.size, (
            'time and positions (lon, lat, depth) don''t have the same lengths.')

        if partitions is not None and partitions is not False:
            self._pu_indicators = convert_to_flat_array(partitions)

        for kwvar in kwargs:
            # kwargs[kwvar] = convert_to_flat_array(kwargs[kwvar])  # input reformatting - particleset-task
            assert lon.size == kwargs[kwvar].size, (
                '%s and positions (lon, lat, depth) don''t have the same lengths.' % kwvar)

        offset = np.max(pid) if len(pid) > 0 else -1
        if MPI:
            mpi_comm = MPI.COMM_WORLD
            mpi_rank = mpi_comm.Get_rank()
            mpi_size = mpi_comm.Get_size()

            if lon.size < mpi_size and mpi_size > 1:
                raise RuntimeError('Cannot initialise with fewer particles than MPI processors')

            if mpi_size > 1:
                if partitions is not False:
                    if self._pu_indicators is None:
                        if mpi_rank == 0:
                            coords = np.vstack((lon, lat)).transpose()
                            kmeans = KMeans(n_clusters=mpi_size, random_state=0).fit(coords)
                            self._pu_indicators = kmeans.labels_
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

        # -- Repeat-dt structure (and thus possibly also flat-array conversion) to be done by the particle set -- #
        # -- Makes sense that the overarching particle set reformats the input to what is necessary.           -- #
        # self.repeatdt = repeatdt.total_seconds() if isinstance(repeatdt, delta) else repeatdt
        # if self.repeatdt:
        #     if self.repeatdt <= 0:
        #         raise('Repeatdt should be > 0')
        #     if time[0] and not np.allclose(time, time[0]):
        #         raise ('All Particle.time should be the same when repeatdt is not None')
        #     self.repeat_starttime = time[0]
        #     self.repeatlon = lon
        #     self.repeatlat = lat
        #     self.repeatpid = pid - pclass.lastID
        #     self.repeatdepth = depth
        #     self.repeatpclass = pclass
        #     self.partitions = partitions
        #     self.repeatkwargs = kwargs
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
        # -- Kernel is a particle set-class thingy -- #
        # self.kernel = None

        # store particle data as an array per variable (structure of arrays approach)
        self._data = {}
        initialised = set()

        for v in self.ptype.variables:
            if v.name in ['xi', 'yi', 'zi', 'ti']:
                self._data[v.name] = np.empty((len(lon), ngrid), dtype=v.dtype)
            else:
                self._data[v.name] = np.empty(len(lon), dtype=v.dtype)

        self._ncount = self._data['lon'].shape[0]

        if lon is not None and lat is not None:
            # Initialise from lists of lon/lat coordinates
            assert self.ncount == len(lon) and self.ncount == len(lat), (
                'Size of ParticleSet does not match length of lon and lat.')

            # mimic the variables that get initialised in the constructor
            self._data['lat'][:] = lat
            self._data['lon'][:] = lon
            self._data['depth'][:] = depth
            self._data['time'][:] = time
            self._data['id'][:] = pid
            self._data['fileid'][:] = -1

            # special case for exceptions which can only be handled from scipy
            self._data['exception'] = np.empty(self.ncount, dtype=object)

            initialised |= {'lat', 'lon', 'depth', 'time', 'id'}

            # any fields that were provided on the command line
            for kwvar, kwval in kwargs.items():
                if not hasattr(pclass, kwvar):
                    raise RuntimeError('Particle class does not have Variable %s' % kwvar)
                self._data[kwvar][:] = kwval
                initialised.add(kwvar)

            # initialise the rest to their default values
            for v in self.ptype.variables:
                if v.name in initialised:
                    continue

                if isinstance(v.initial, Field):
                    for i in range(self.ncount):
                        if np.isnan(time[i]):
                            raise RuntimeError('Cannot initialise a Variable with a Field if no time provided. '
                                               'Add a "time=" to ParticleSet construction')
                        v.initial.fieldset.computeTimeChunk(time[i], 0)
                        self._data[v.name][i] = v.initial[
                            time[i], depth[i], lat[i], lon[i]
                        ]
                        logger.warning_once("Particle initialisation from field can be very slow as it is computed in scipy mode.")
                elif isinstance(v.initial, attrgetter):
                    self._data[v.name][:] = v.initial(self)
                else:
                    self._data[v.name][:] = v.initial

                initialised.add(v.name)
        else:
            raise ValueError("Latitude and longitude required for generating ParticleSet")

    # ==== already user-exposed ==== #
    def __del__(self):
        """
        Collection - Destructor
        """
        super().__del__()

    # ==== already user-exposed ==== #
    def __iter__(self):
        """Returns an Iterator that allows for forward iteration over the
        elements in the ParticleCollection (e.g. `for p in pset:`).
        """
        return ParticleCollectionIteratorSOA(self)

    # ==== already user-exposed ==== #
    def __reversed__(self):
        """Returns an Iterator that allows for backwards iteration over
        the elements in the ParticleCollection (e.g.
        `for p in reversed(pset):`).
        """
        return ParticleCollectionIteratorSOA(self, True)

    # ==== already user-exposed ==== #
    def __getitem__(self, index):
        """
        Access a particle in this collection using the fastest access
        method for this collection - by its index.

        :param index: int or np.int32 index of a particle in this collection
        """
        return self.get_single_by_index(index)

    def get_single_by_index(self, index):
        """
        This function gets a (particle) object from the collection based on its index within the collection. For
        collections that are not based on random access (e.g. ordered lists, sets, trees), this function involves a
        translation of the index into the specific object reference in the collection - or (if unavoidable) the
        translation of the collection from a none-indexable, none-random-access structure into an indexable structure.
        In cases where a get-by-index would result in a performance malus, it is highly-advisable to use a different
        get function, e.g. get-by-ID.
        """
        super().get_single_by_index(index)

        return ParticleAccessorSOA(self, index)

    def get_single_by_object(self, particle_obj):
        """
        This function gets a (particle) object from the collection based on its actual object. For collections that
        are random-access and based on indices (e.g. unordered list, vectors, arrays and dense matrices), this function
        would involve a parsing of the whole list and translation of the object into an index in the collection - which
        results in a significant performance malus.
        In cases where a get-by-object would result in a performance malus, it is highly-advisable to use a different
        get function, e.g. get-by-index or get-by-ID.
        """
        super().get_single_by_object(particle_obj)

        # We cannot look for the object directly, so we will look for one of
        # its properties that has the nice property of being stored in an
        # ordered list
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
        super().get_single_by_ID(id)

        # Use binary search if the collection is sorted, linear search otherwise
        index = -1
        if self._sorted:
            index = bisect_left(self._data['id'], id)
            if index == len(self._data['id']) or self._data['id'][index] != id:
                raise ValueError("Trying to access a particle with a non-existing"
                                 f" ID: {id}.")
        else:
            index = np.where(self._data['id'] == id)[0][0]

        return self.get_single_by_index(index)

    def get_same(self, same_class):
        """
        This function gets particles from this collection that are themselves stored in another object of an equi-
        structured ParticleCollection.
        """
        super().get_same(same_class)
        raise NotImplementedError

    def get_collection(self, pcollection):
        """
        This function gets particles from this collection that are themselves stored in a ParticleCollection, which
        is differently structured than this one. That means the other-collection has to be re-formatted first in an
        intermediary format.
        """
        super().get_collection(pcollection)
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
        super().get_multi_by_PyCollection_Particles(pycollectionp)
        raise NotImplementedError

    def get_multi_by_indices(self, indices):
        """
        This function gets particles from this collection based on their indices. This works best for random-access
        collections (e.g. numpy's ndarrays, dense matrices and dense arrays), whereas internally ordered collections
        shall rather use a get-via-object-reference strategy.
        """
        super().get_multi_by_indices(indices)
        if type(indices) is dict:
            indices = list(indices.values())
        return ParticleCollectionIteratorSOA(self, subset=indices)

    def get_multi_by_IDs(self, ids):
        """
        This function gets particles from this collection based on their IDs. For collections where this removal
        strategy would require a collection transformation or by-ID parsing, it is advisable to rather apply a get-
        by-objects or get-by-indices scheme.

        Note that this implementation assumes that IDs of particles are strictly increasing with increasing index. So
        a particle with a larger index will always have a larger ID as well. The assumption holds for this datastructure
        as new particles always get a larger ID than any existing particle (IDs are not recycled) and their data are
        appended at the end of the list (largest index). This allows for the use of binary search in the look-up.
        """
        super().get_multi_by_IDs(ids)
        if type(ids) is dict:
            ids = list(ids.values())

        if len(ids) == 0:
            return None

        # Use binary search if the collection is sorted, linear search otherwise
        indices = np.empty(len(ids), dtype=np.int32)
        if self._sorted:
            # This is efficient if len(ids) << self.len
            sorted_ids = np.sort(np.array(ids))
            indices = self._recursive_ID_lookup(0, len(self._data['id']), sorted_ids)
        else:
            indices = np.where(np.in1d(self._data['id'], ids))[0]

        return self.get_multi_by_indices(indices)

    def _recursive_ID_lookup(self, low, high, sublist):
        # Identify the middle element of the sublist and perform a binary
        # search on it.
        # NOTE: This implementation has not been tested yet
        median = floor(len(sublist) / 2)
        index = bisect_left(self._data['id'][low:high], sublist[median])
        if len(sublist) == 1:
            # edge case
            if index == len(self._data['id']) or \
               self._data['id'][index] != sublist[median]:
                # Should a ValueError be raised here as well?
                return np.array([])
            return np.array([index])

        # The edge-cases have to be handled slightly differently
        if index == len(self._data['id']):
            # Continue with the same bounds, but drop the median.
            return self._recursive_ID_lookup(low, high, np.delete(sublist, median))
        elif self._data['id'][index] != sublist[median]:
            # We can split, because we received the index that the median
            # ID would have been inserted in, but we do not return the
            # index and keep it in our search space.
            left = self._recursive_ID_lookup(low, index, sublist[:median])
            right = self._recursive_ID_lookup(index, high, sublist[median + 1:])
            return np.concatenate((left, right))

        # Otherwise, we located the median, so we include it in our
        # result, and split the search space on it, without including it.
        left = self._recursive_ID_lookup(low, index, sublist[:median])
        right = self._recursive_ID_lookup(index + 1, high, sublist[median + 1:])
        return np.concatenate((left, np.array(index), right))

    def add_collection(self, pcollection):
        """
        Adds another, differently structured ParticleCollection to this collection. This is done by, for example,
        appending/adding the items of the other collection to this collection.
        """
        super().add_collection(pcollection)
        raise NotImplementedError

    def add_single(self, particle_obj):
        """
        Adding a single Particle to the collection - either as a 'Particle; object in parcels itself, or
        via its ParticleAccessor.
        """
        super().add_single(particle_obj)
        raise NotImplementedError

    def add_same(self, same_class):
        """
        Adds another, equi-structured ParticleCollection to this collection. This is done by concatenating
        both collections. The fact that they are of the same ParticleCollection's derivative simplifies
        parsing and concatenation.
        """
        super().add_same(same_class)

        if len(same_class._data['id']) == 0:
            return

        if len(self._data['id']) == 0:
            self._data = same_class._data
            self._ncount = same_class.ncount

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

    # ==== already user-exposed ==== #
    def __iadd__(self, same_class):
        """
        Performs an incremental addition of the equi-structured ParticleCollections, such to allow

        a += b,

        with 'a' and 'b' begin the two equi-structured objects (or: 'b' being and individual object).
        This operation is equal to an in-place addition of (an) element(s).
        """
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

    # ==== already user-exposed ==== #
    def __delitem__(self, key):
        """
        This is the high-performance method to delete a specific object from this collection.
        As the most-performant way depends on the specific collection in question, the function is abstract.

        Highlight for the specific implementation:
        The 'key' parameter should still be evaluated for being a single or a multi-entry delete, and needs to check
        that it received the correct type of 'indexing' argument (i.e. index, id or iterator).
        """
        self.delete_by_index(key)

    def delete_by_index(self, index):
        """
        This method deletes a particle from the  the collection based on its index. It does not return the deleted item.
        Semantically, the function appears similar to the 'remove' operation. That said, the function in OceanParcels -
        instead of directly deleting the particle - just raises the 'deleted' status flag for the indexed particle.
        In result, the particle still remains in the collection. The functional interpretation of the 'deleted' status
        is handled by 'recovery' dictionary during simulation execution.
        """
        super().delete_by_index(index)

        self._data['state'][index] = OperationCode.Delete

    def delete_by_ID(self, id):
        """
        This method deletes a particle from the  the collection based on its ID. It does not return the deleted item.
        Semantically, the function appears similar to the 'remove' operation. That said, the function in OceanParcels -
        instead of directly deleting the particle - just raises the 'deleted' status flag for the indexed particle.
        In result, the particle still remains in the collection. The functional interpretation of the 'deleted' status
        is handled by 'recovery' dictionary during simulation execution.
        """
        super().delete_by_ID(id)

        # Use binary search if the collection is sorted, linear search otherwise
        index = -1
        if self._sorted:
            index = bisect_left(self._data['id'], id)
            if index == len(self._data['id']) or \
               self._data['id'][index] != id:
                raise ValueError("Trying to delete a particle with a non-"
                                 f"existing ID: {id}.")
        else:
            index = np.where(self._data['id'] == id)[0][0]

        self.delete_by_index(index)

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
        super().remove_single_by_index(index)

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
        super().remove_single_by_object(particle_obj)

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
        super().remove_single_by_ID(id)

        # Use binary search if the collection is sorted, linear search otherwise
        index = -1
        if self._sorted:
            index = bisect_left(self._data['id'], id)
            if index == len(self._data['id']) or \
               self._data['id'][index] != id:
                raise ValueError("Trying to remove a particle with a non-"
                                 f"existing ID: {id}.")
        else:
            index = np.where(self._data['id'] == id)[0][0]

        self.remove_single_by_index(index)

    def remove_same(self, same_class):
        """
        This function removes particles from this collection that are themselves stored in another object of an equi-
        structured ParticleCollection. As the structures of both collections are the same, a more efficient M-in-N
        removal can be applied without an in-between reformatting.
        """
        super().remove_same(same_class)
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
        super().remove_collection(pcollection)
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
        super().remove_multi_by_PyCollection_Particles(pycollectionp)
        raise NotImplementedError

    def remove_multi_by_indices(self, indices):
        """
        This function removes particles from this collection based on their indices. This works best for random-access
        collections (e.g. numpy's ndarrays, dense matrices and dense arrays), whereas internally ordered collections
        shall rather use a removal-via-object-reference strategy.
        """
        super().remove_multi_by_indices(indices)
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
        super().remove_multi_by_IDs(ids)
        if type(ids) is dict:
            ids = list(ids.values())

        if len(ids) == 0:
            return

        # Use binary search if the collection is sorted, linear search otherwise
        indices = np.empty(len(ids), dtype=np.int32)
        if self._sorted:
            # This is efficient if len(ids) << self.len
            sorted_ids = np.sort(np.array(ids))
            indices = self._recursive_ID_lookup(0, len(self._data['id']), sorted_ids)
        else:
            indices = np.where(np.in1d(self._data['id'], ids))[0]

        self.remove_multi_by_indices(indices)

    # ==== already user-exposed ==== #
    def __isub__(self, other):
        """
        This method performs an incremental removal of the equi-structured ParticleCollections, such to allow

        a -= b,

        with 'a' and 'b' begin the two equi-structured objects (or: 'b' being and individual object).
        This operation is equal to an in-place removal of (an) element(s).
        """
        if other is None:
            return
        if type(other) is type(self):
            self.remove_same(other)
        elif (isinstance(other, BaseParticleAccessor)
              or isinstance(other, ScipyParticle)):
            self.remove_single_by_object(other)
        else:
            raise TypeError("Trying to do an incremental removal of an"
                            f" element of type {type(other)}, which is"
                            " not supported.")
        return self

    def pop_single_by_index(self, index):
        """
        Searches for Particle at index 'index', removes that Particle from the Collection and returns that Particle (or: ParticleAccessor).
        If index is None, return last item (-1);
        If index < 0: return from 'end' of collection.
        If index is out of bounds, throws and OutOfRangeException.
        If Particle cannot be retrieved, returns None.
        """
        super().pop_single_by_index(index)
        raise NotImplementedError

    def pop_single_by_ID(self, id):
        """
        Searches for Particle with ID 'id', removes that Particle from the Collection and returns that Particle (or: ParticleAccessor).
        If Particle cannot be retrieved (e.g. because the ID is not available), returns None.
        """
        super().pop_single_by_ID(id)
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
        super().pop_multi_by_indices(indices)
        raise NotImplementedError

    def pop_multi_by_IDs(self, ids):
        """
        Searches for Particles with the IDs registered in 'ids', removes the Particles from the Collection and returns the Particles (or: their ParticleAccessors).
        If Particles cannot be retrieved (e.g. because the IDs are not available), returns None.
        """
        super().pop_multi_by_IDs(ids)
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
        # super().merge(same_class)
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
        # super().split(indices)
        raise NotImplementedError

    # ==== already user-exposed ==== #
    def __sizeof__(self):
        """
        This function returns the size in actual bytes required in memory to hold the collection. Ideally and simply,
        the size is computed as follows:

        sizeof(self) = len(self) * sizeof(pclass)
        """
        raise NotImplementedError

    # ==== already user-exposed ==== #
    def clear(self):
        """
        This function physically removes all elements of the collection, yielding an empty collection as result of the
        operation.
        """
        raise NotImplementedError

    def cstruct(self):
        """
        'cstruct' returns the ctypes mapping of the particle data. This depends on the specific structure in question.
        """
        class CParticles(Structure):
            # TODO adapt naming of _fields_ in CParticles (code generator) into _variables_
            _fields_ = [(v.name, POINTER(np.ctypeslib.as_ctypes_type(v.dtype))) for v in self._ptype.variables]

        def flatten_dense_data_array(v):
            data_flat = self._data[v.name].view()
            data_flat.shape = -1
            return np.ctypeslib.as_ctypes(data_flat)

        cdata = [flatten_dense_data_array(v) for v in self._ptype.variables]
        cstruct = CParticles(*cdata)
        return cstruct

    def toDictionary(self):     # formerly: ParticleSet.to_dict()
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
        raise NotImplementedError

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
        """
        Method to set the write status of a Variable
        :param var: Name of the variable (string)
        :param status: Write status of the variable (True, False or 'once')
        """
        var_changed = False
        for v in self._ptype.variables:
            if v.name == var:
                v.to_write = write_status
                var_changed = True
        if not var_changed:
            raise SyntaxError('Could not change the write status of %s, because it is not a Variable name' % var)


class ParticleAccessorSOA(BaseParticleAccessor):
    def __init__(self, pcoll, index):
        """Initializes the ParticleAccessor to provide access to one
        specific particle.
        """
        super().__init__(pcoll)
        self._index = index

    def __getattr__(self, name):
        return self.pcoll._data[name][self._index]

    def __setattr__(self, name, value):
        if name in ['pcoll', '_index']:
            object.__setattr__(self, name, value)
        else:
            # avoid recursion
            self.pcoll._data[name][self._index] = value

    def __repr__(self):
        time_string = 'not_yet_set' if self.time is None or np.isnan(self.time) else "{:f}".format(self.time)
        str = "P[%d](lon=%f, lat=%f, depth=%f, " % (self.id, self.lon, self.lat, self.depth)
        for var in self.pcoll._ptype.variables:
            if var.to_write is not False and var.name not in ['id', 'lon', 'lat', 'depth', 'time']:
                str += "%s=%f, " % (var.name, getattr(self, var.name))
        return str + "time=%s)" % time_string


class ParticleCollectionIteratorSOA(BaseParticleCollectionIterator):
    def __init__(self, pcoll, reverse=False, subset=None):
        # super().__init__(pcoll)  # Do not actually need this

        if subset is not None:
            if type(subset[0]) not in [int, np.int32, np.intp]:
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
        self._head = ParticleAccessorSOA(pcoll, self._indices[0])
        self._tail = ParticleAccessorSOA(pcoll,
                                         self._indices[self.max_len - 1])
        self.p = self._head

    def __next__(self):
        if self._index < self.max_len:
            self.p = ParticleAccessorSOA(self._pcoll,
                                         self._indices[self._index])
            self._index += 1
            return self.p

        # End of Iteration
        raise StopIteration

    @property
    def current(self):
        return self.p

    def __repr__(self):
        dir_str = 'Backward' if self._reverse else 'Forward'
        str = f"{dir_str} iteration at index {self._index} of {self.max_len}."
        return str

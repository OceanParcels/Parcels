import time as time_module
from parcels.application_kernels.advection import AdvectionRK4
from datetime import date
from datetime import datetime
from datetime import timedelta as delta

import os  # noqa: F401
import numpy as np  # noqa: F401
import xarray as xr
from ctypes import c_void_p
import cftime

from parcels.grid import GridCode
from parcels.field import Field
from parcels.compilation import GNUCompiler_MS
from parcels.tools import get_cache_dir, get_package_dir
from parcels.tools import GenerateID_Service, SequentialIdGenerator
from parcels.compilation import LibraryRegisterC
from parcels.kernel.kernelnodes import KernelNodes
from parcels.particle import Variable, ScipyParticle, JITParticle  # noqa: F401
from parcels.particlefile.particlefilenodes import ParticleFileNodes
from parcels.tools.statuscodes import StateCode, OperationCode    # noqa: F401
from parcels.particleset.baseparticleset import BaseParticleSet
from parcels.collection.collectionnodes import ParticleCollectionNodes

from parcels.tools.converters import _get_cftime_calendars
from parcels.tools.loggers import logger

try:
    from mpi4py import MPI
except:
    MPI = None

__all__ = ['ParticleSetNodes']


def _convert_to_array(var):
    """Convert lists and single integers/floats to one-dimensional numpy
    arrays
    """
    if isinstance(var, np.ndarray):
        return var.flatten()
    elif isinstance(var, (int, float, np.float32, np.float64, np.int32)):
        return np.array([var])
    else:
        return np.array(var)


def _convert_to_reltime(time):
    """Check to determine if the value of the time parameter needs to be
    converted to a relative value (relative to the time_origin).
    """
    if isinstance(time, np.datetime64) or (hasattr(time, 'calendar') and time.calendar in _get_cftime_calendars()):
        return True
    return False


class ParticleSetNodes(BaseParticleSet):
    """This particle set organizes actual particles in form of a double-linked list, such as:

    None  -> prev     | -> prev    | -> None
             next   - |    next  - |
             data          data

    :arg data is the node's payload, which is an actual ScipyParticle or JITParticle. Remember that, as a node-based
    list, this particle set is built upon a non-indexable collections. This can lead to a speed boost for a few
    applications, but it also means that indices to the particles are not preserved (i.e. they change) and that
    iterating through the particle set is done differently than with structured arrays. Also keep in mind, if you
    use the node-list in an indexing-manner, then the particle set will be very slow. For guidelines how to use it,
    consult the tests in 'test_particle_sets.py' and 'test_nodes.py'.

    :param fieldset: :mod:`parcels.fieldset.FieldSet` object from which to sample velocity.
           While fieldset=None is supported, this will throw a warning as it breaks most Parcels functionality
    :param pclass: Optional :mod:`parcels.particle.JITParticle` or
                 :mod:`parcels.particle.ScipyParticle` object that defines custom particle
    :param lon: List of initial longitude values for particles
    :param lat: List of initial latitude values for particles
    :param depth: Optional list of initial depth values for particles. Default is 0m
    :param time: Optional list of initial time values for particles. Default is fieldset.U.grid.time[0]
    :param repeatdt: Optional interval (in seconds) on which to repeat the release of the ParticleSet
    :param lonlatdepth_dtype: Floating precision for lon, lat, depth particle coordinates.
           It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
           and np.float64 if the interpolation method is 'cgrid_velocity'
    :param pid_orig: Optional list of (offsets for) the particle IDs
    :param partitions: List of cores on which to distribute the particles for MPI runs. Default: None, in which case particles
           are distributed automatically on the processors

    Other Variables can be initialised using further arguments (e.g. v=... for a Variable named 'v')
    """
    _c_lib_register = None
    _idgen = None

    def __init__(self, fieldset=None, pclass=JITParticle, lon=None, lat=None, depth=None, time=None,
                 repeatdt=None, lonlatdepth_dtype=None, pid_orig=None, idgen=None, c_lib_register=LibraryRegisterC(), **kwargs):
        super(ParticleSetNodes, self).__init__()
        self._idgen = idgen
        if self._idgen is None:
            logger.warning("A node-based particle set requires a global-context ID generator. Creating a default ID generator internally.")
            self._idgen = GenerateID_Service(SequentialIdGenerator)

        self._c_lib_register = c_lib_register
        if self._c_lib_register is None:
            logger.warning("A node-based particle set requires a global-context 'LibraryRegisterC'. Creating default 'LibraryRegisterC' internally.")
            self._c_lib_register = LibraryRegisterC()
        # ==== first: create a new subclass of the pclass that includes the required variables ==== #
        # ==== see dynamic-instantiation trick here: https://www.python-course.eu/python3_classes_and_type.php ==== #

        Alist = [pclass.__name__[0:6], pclass.__name__[0:5]]
        Blist = ["Array", "Object"]
        class_is_derived = any([key in Blist for key in Alist])
        class_name = "Object"+pclass.__name__ if not class_is_derived else pclass.__name__
        object_class = None
        if class_is_derived:
            logger.warn("Reusing original class '{}' instead of deriving an object-version again. This is potentially incorrect - please check your object naming.".format(pclass.__name__))
            object_class = pclass
        elif class_name not in dir():
            def ObjectScipyClass_init(self, *args, **kwargs):
                fieldset = kwargs.get('fieldset', None)
                ngrids = kwargs.get('ngrids', None)
                if type(self).ngrids.initial < 0:
                    numgrids = ngrids
                    if numgrids is None and fieldset is not None:
                        numgrids = fieldset.gridset.size
                    assert numgrids is not None, "Neither fieldsets nor number of grids are specified - exiting."
                    type(self).ngrids.initial = numgrids
                self.ngrids = type(self).ngrids.initial
                if self.ngrids >= 0:
                    for index in ['xi', 'yi', 'zi', 'ti']:
                        if index != 'ti':
                            setattr(self, index, np.zeros(self.ngrids, dtype=np.int32))
                        else:
                            setattr(self, index, -1*np.ones(self.ngrids, dtype=np.int32))
                super(type(self), self).__init__(*args, **kwargs)

            def ObjectJITClass_init(self, *args, **kwargs):
                super(type(self), self).__init__(*args, **kwargs)  # needs to go first cause it initialises the space for all other variables
                fieldset = kwargs.get('fieldset', None)
                ngrids = kwargs.get('ngrids', None)
                if type(self).ngrids.initial < 0:
                    numgrids = ngrids
                    if numgrids is None and fieldset is not None:
                        numgrids = fieldset.gridset.size
                    assert numgrids is not None, "Neither fieldsets nor number of grids are specified - exiting."
                    type(self).ngrids.initial = numgrids
                self.ngrids = type(self).ngrids.initial
                if self.ngrids >= 0:
                    for index in ['xi', 'yi', 'zi', 'ti']:
                        if index != 'ti':
                            setattr(self, index, np.zeros((self.ngrids), dtype=np.int32))
                        else:
                            setattr(self, index, -1*np.ones((self.ngrids), dtype=np.int32))
                        setattr(self, index+'p', getattr(self, index).ctypes.data_as(c_void_p))  # without direct (!) prior recast, throws error that the dtype (here: int32) has no ctypes-property
                        setattr(self, 'c'+index, getattr(self, index+'p').value)

            def ObjectClass_del_forward(self):
                super(type(self), self).__del__()

            def ObjectClass_repr_forward(self):
                return super(type(self), self).__repr__()

            def ObjectClass_eq_forward(self, other):
                return super(type(self), self).__eq__(other)

            def ObjectClass_ne_forward(self, other):
                return super(type(self), self).__ne__(other)

            def ObjectClass_lt_forward(self, other):
                return super(type(self), self).__lt__(other)

            def ObjectClass_le_forward(self, other):
                return super(type(self), self).__le__(other)

            def ObjectClass_gt_forward(self, other):
                return super(type(self), self).__gt__(other)

            def ObjectClass_ge_forward(self, other):
                return super(type(self), self).__ge__(other)

            def ObjectClass_sizeof_forward(self):
                return super(type(self), self).__sizeof__()

            object_scipy_class_vdict = {"ngrids": Variable('ngrids', dtype=np.int32, to_write=False, initial=-1),
                                        "xi": np.ndarray((1,), dtype=np.int32),
                                        "yi": np.ndarray((1,), dtype=np.int32),
                                        "zi": np.ndarray((1,), dtype=np.int32),
                                        "ti": np.ndarray((1,), dtype=np.int32),
                                        "__init__": ObjectScipyClass_init,
                                        "__del__": ObjectClass_del_forward,
                                        "__repr__": ObjectClass_repr_forward,
                                        "__eq__": ObjectClass_eq_forward,
                                        "__ne__": ObjectClass_ne_forward,
                                        "__lt__": ObjectClass_lt_forward,
                                        "__le__": ObjectClass_le_forward,
                                        "__gt__": ObjectClass_gt_forward,
                                        "__ge__": ObjectClass_ge_forward,
                                        "__sizeof__": ObjectClass_sizeof_forward
                                        }

            object_jit_class_vdict = {"ngrids": Variable('ngrids', dtype=np.int32, to_write=False, initial=-1),
                                      "_cptr": None,
                                      "xi": np.ndarray((1,), dtype=np.int32),
                                      "yi": np.ndarray((1,), dtype=np.int32),
                                      "zi": np.ndarray((1,), dtype=np.int32),
                                      "ti": np.ndarray((1,), dtype=np.int32),
                                      "xip": c_void_p,
                                      "yip": c_void_p,
                                      "zip": c_void_p,
                                      "tip": c_void_p,
                                      "cxi": Variable('cxi', dtype=np.dtype(c_void_p), to_write=False),
                                      "cyi": Variable('cyi', dtype=np.dtype(c_void_p), to_write=False),
                                      "czi": Variable('czi', dtype=np.dtype(c_void_p), to_write=False),
                                      "cti": Variable('cti', dtype=np.dtype(c_void_p), to_write=False),
                                      "__init__": ObjectJITClass_init,
                                      "__del__": ObjectClass_del_forward,
                                      "__repr__": ObjectClass_repr_forward,
                                      "__eq__": ObjectClass_eq_forward,
                                      "__ne__": ObjectClass_ne_forward,
                                      "__lt__": ObjectClass_lt_forward,
                                      "__le__": ObjectClass_le_forward,
                                      "__gt__": ObjectClass_gt_forward,
                                      "__ge__": ObjectClass_ge_forward,
                                      "__sizeof__": ObjectClass_sizeof_forward
                                      }
            object_class = None
            if issubclass(pclass, JITParticle):
                object_class = type("Object" + pclass.__name__, (pclass, ), object_jit_class_vdict)
            elif issubclass(pclass, ScipyParticle):
                object_class = type("Object" + pclass.__name__, (pclass,), object_scipy_class_vdict)
            if object_class is None:
                raise TypeError("ParticleSetAOS: Given Particle base class is invalid - derive from either ScipyParticle or JITParticle.")
        else:
            object_class = locals()[class_name]

        # ==== ==== ==== ==== ==== ==== dynamic re-classing completed ==== ==== ==== ==== ==== ==== #
        _pclass = object_class
        self._fieldset = fieldset
        if self._fieldset is None:
            logger.warning_once("No FieldSet provided in ParticleSet generation. This breaks most Parcels functionality")
        else:
            self._fieldset.check_complete()
        ngrids = self._fieldset.gridset.size if self._fieldset is not None else 0
        partitions = kwargs.pop('partitions', None)

        # ---- init common parameters to ParticleSets ---- #
        lon = np.empty(shape=0) if lon is None else _convert_to_array(lon)
        lat = np.empty(shape=0) if lat is None else _convert_to_array(lat)

        if isinstance(pid_orig, (type(None), type(False))):
            pid_orig = self._idgen.usable_length

        if depth is None:
            mindepth = self._fieldset.gridset.dimrange('depth')[0] if self.fieldset is not None else 0
            depth = np.ones(lon.size, dtype=lonlatdepth_dtype) * mindepth
        else:
            depth = _convert_to_array(depth)
        assert lon.size == lat.size and lon.size == depth.size, (
            'lon, lat, depth don''t all have the same lenghts')

        time = _convert_to_array(time)
        time = np.repeat(time, len(lon)) if time.size == 1 else time

        if time.size > 0 and type(time[0]) in [datetime, date]:
            time = np.array([np.datetime64(t) for t in time])
        self.time_origin = fieldset.time_origin if self._fieldset is not None else 0
        if time.size > 0 and isinstance(time[0], np.timedelta64) and not self.time_origin:
            raise NotImplementedError('If fieldset.time_origin is not a date, time of a particle must be a double')
        time = np.array([self.time_origin.reltime(t) if _convert_to_reltime(t) else t for t in time])
        assert lon.size == time.size, "time and positions (lon, lat, depth) do not have the same lengths."

        if lonlatdepth_dtype is not None:
            lonlatdepth_dtype = lonlatdepth_dtype
        else:
            if fieldset is not None:
                lonlatdepth_dtype = self.lonlatdepth_dtype_from_field_interp_method(fieldset.U)
            else:
                lonlatdepth_dtype = np.float32
        assert lonlatdepth_dtype in [np.float32, np.float64], \
            'lon lat depth precision should be set to either np.float32 or np.float64'

        # ---- particle data parameter length assertions ---- #
        for kwvar in kwargs:
            kwargs[kwvar] = _convert_to_array(kwargs[kwvar])
            assert lon.shape[0] == kwargs[kwvar].shape[0], (
                '%s and positions (lon, lat, depth) don''t have the same lengths.' % kwargs[kwvar])

        # ------------------------------------------------------------------------------------ #
        # ---- ---- ---- ---- set up repeated-release particles parameters ---- ---- ---- ---- #
        self.repeatdt = repeatdt.total_seconds() if isinstance(repeatdt, delta) else repeatdt
        if self.repeatdt:
            if self.repeatdt <= 0:
                raise('Repeatdt should be > 0')
            if time[0] and not np.allclose(time, time[0]):
                raise ('All Particle.time should be the same when repeatdt is not None')
            self.repeatpclass = _pclass

        self._collection = ParticleCollectionNodes(self._idgen, self._c_lib_register, _pclass, lon=lon, lat=lat, depth=depth, time=time, lonlatdepth_dtype=lonlatdepth_dtype, pid_orig=pid_orig, partitions=partitions, ngrid=ngrids, **kwargs)

        self.repeatlon = None
        self.repeatlat = None
        self.repeatdepth = None
        self.repeatkwargs = None
        self.repeat_starttime = None
        if self.repeatdt:
            self.repeatlon = np.empty(len(self._collection), dtype=self._lonlatdepth_dtype)
            self.repeatlat = np.empty(len(self._collection), dtype=self._lonlatdepth_dtype)
            self.repeatdepth = np.empty(len(self._collection), dtype=self._lonlatdepth_dtype)
            self.repeatkwargs = {}
            for kwvar in kwargs.keys():
                self.repeatkwargs[kwvar] = []

            collect_time = [pdata.time for pdata in self._collection]
            if len(time) > 0 and (time[0] is None or np.isnan(time[0])):
                self.repeat_starttime = time[0]
            else:
                if collect_time and not np.allclose(collect_time, collect_time[0]):
                    raise ValueError('All Particle.time should be the same when repeatdt is not None')
                self.repeat_starttime = collect_time[0]

            base_time = None
            pidx = 0
            for pdata in self._collection:
                self.repeatlon[pidx] = pdata.lon
                self.repeatlat[pidx] = pdata.lat
                self.repeatdepth[pidx] = pdata.depth
                for kwvar in kwargs:
                    self.repeatkwargs[kwvar].append(getattr(pdata, kwvar))
                pidx += 1

        self.repeatpid = None
        if self.repeatdt:
            if MPI and self._collection.pu_indicators is not None:
                mpi_comm = MPI.COMM_WORLD
                mpi_rank = mpi_comm.Get_rank()
                self.repeatpid = None if pid_orig is None else pid_orig[self._collection.pu_indicators == mpi_rank]
        # ---- ---- ---- ---- ---- finalize repeat-release parameters ---- ---- ---- ---- ---- #
        # ------------------------------------------------------------------------------------ #

        self._kernel = None
        self._kclass = KernelNodes

    def __del__(self):
        """
        ParticleSet - Destructor
        """
        if self._collection is not None:  # collection needs to be deleted here specifically.
            self._collection.clear()
            del self._collection
        self._collection = None
        super(ParticleSetNodes, self).__del__()

    def delete(self, key):
        """
        This is the generic super-method to indicate object deletion of a specific object from this collection.

        Comment/Annotation:
        Functions for deleting multiple objects are more specialised than just a for-each loop of single-item deletion,
        because certain data structures can delete multiple objects in-bulk faster with specialised function than making a
        roundtrip per-item delete operation. Because of the sheer size of those containers and the resulting
        performance demands, we need to make use of those specialised 'del' functions, where available.

        :arg key: ID(s) or index/indices of particles that are to be marked by a 'delete'-state
        """
        if key is None:
            return
        if type(key) in [int, np.int32, np.intp]:
            self._collection.delete_by_index(key)
        elif type(key) in [np.int64, np.uint64]:
            self._collection.delete_by_ID(key)

    def _set_particle_vector(self, name, value):
        """
        Set attributes of all particles to new values.

        :arg name: Name of the attribute (str).
        :arg value: New value to set the attribute of the particles to.
        """
        ndata = self._collection.begin()
        while ndata is not None:
            # ==== we need to skip here deleted nodes that have been queued for deletion, but are still bound in memory ==== #
            if not ndata.isvalid():
                ndata = ndata.next
                continue
            pdata = ndata.data
            setattr(pdata, name, value)
            ndata.set_data(pdata)
            ndata = ndata.next

    def _impute_release_times(self, default):
        """Set attribute 'time' to default if encountering NaN values.

        :arg default: Default release time.
        :return: Minimum and maximum release times.
        """
        max_rt = None
        min_rt = None
        for p in self:
            if np.isnan(p.time):
                p.time = default
            if max_rt is None or max_rt < p.time:
                max_rt = p.time
            if min_rt is None or min_rt > p.time:
                min_rt = p.time
        return min_rt, max_rt

    def data_indices(self, variable_name, compare_values, invert=False):
        """
        Idea:
        Get the indices of all particles where the value of
        `variable_name` equals (one of) `compare_values`.

        :param variable_name: Name of the variable to check.
        :param compare_values: Value or list of values to compare to.
        :param invert: Whether to invert the selection. I.e., when True,
                       return all indices that do not equal (one of)
                       `compare_values`.
        :return: Numpy array of indices that satisfy the test.

        Functionality:
        *not implemented*, as node-based data structure is _not_ index-based.
        Please use 'data_ids' or 'data_nodes'
        """
        raise NotImplementedError("'data_indices' function cannot be used as the node-based structure is not indexable.")

    def data_ids(self, variable_name, compare_values, invert=False):
        """Get the ID's of all particles where the value of
        `variable_name` equals (one of) `compare_values`.

        :param variable_name: Name of the variable to check.
        :param compare_values: Value or list of values to compare to.
        :param invert: Whether to invert the selection. I.e., when True,
                       return all indices that do not equal (one of)
                       `compare_values`.
        :return: Numpy array of ids that satisfy the test.
        """
        result = []
        ndata = self._collection.begin()
        i = 0
        while ndata is not None:
            # ==== we need to skip here deleted nodes that have been queued for deletion, but are still bound in memory ==== #
            if not ndata.isvalid():
                ndata = ndata.next
                continue
            pdata = ndata.data
            if type(compare_values) in [list, dict, np.ndarray]:
                valid = (getattr(pdata, variable_name) == compare_values[i])
                valid = not valid if invert else valid
                if valid:
                    result.append(pdata.id)
            else:
                valid = (getattr(pdata, variable_name) == compare_values)
                valid = not valid if invert else valid
                if valid:
                    result.append(pdata.id)
            ndata = ndata.next
            i += 1
        return np.array(result, dtype=np.int64)

    def data_nodes(self, variable_name, compare_values, invert=False):
        """Get the nodes of all particles where the value of
        `variable_name` equals (one of) `compare_values`.

        :param variable_name: Name of the variable to check.
        :param compare_values: Value or list of values to compare to.
        :param invert: Whether to invert the selection. I.e., when True,
                       return all indices that do not equal (one of)
                       `compare_values`.
        :return: Python list of Node objects
        """
        result = []
        ndata = self._collection.begin()
        i = 0
        while ndata is not None:
            # ==== we need to skip here deleted nodes that have been queued for deletion, but are still bound in memory ==== #
            if not ndata.isvalid():
                ndata = ndata.next
                continue
            pdata = ndata.data
            if type(compare_values) in [list, dict, np.ndarray]:
                valid = (getattr(pdata, variable_name) == compare_values[i])
                valid = not valid if invert else valid
                if valid:
                    result.append(ndata)
            else:
                valid = (getattr(pdata, variable_name) == compare_values)
                valid = not valid if invert else valid
                if valid:
                    result.append(ndata)
            ndata = ndata.next
            i += 1
        return result

    def indexed_subset(self, indices):
        """

        Functionality:
        *not implemented*, as node-based data structure is _not_ index-based.
        Please use 'data_ids' or 'data_nodes'

        """
        raise NotImplementedError("'indexed_subset' function shall not be used as the node-based structure is not indexable.")

    def populate_indices(self):
        """Pre-populate guesses of particle xi/yi indices using a kdtree.

        This is only intended for curvilinear grids, where the initial index search
        may be quite expensive.

        Functionality:
        *not implemented*, as node-based data structure is _not_ index-based.
        """
        raise NotImplementedError("'populate_indices' function cannot be used as the node-based structure is not indexable.")

    def cptr(self, index):
        return self._collection.cptr(index)

    def empty(self):
        return len(self._collection) <= 0

    def begin(self):
        """
        Returns the begin of the linked particle list (like C++ STL begin() function)
        :return: begin Node (Node whose prev element is None); returns None if ParticleSet is empty
        """
        return self._collection.begin()

    def end(self):
        """
        Returns the end of the linked partile list. UNLIKE in C++ STL, it returns the last element (valid element),
        not the element past the last element (invalid element). (see http://www.cplusplus.com/reference/list/list/end/)
        :return: end Node (Node whose next element is None); returns None if ParticleSet is empty
        """
        return self._collection.end()

    @property
    def size(self):
        return len(self._collection)

    @property
    def data(self):
        return self._collection.data

    @property
    def particles(self):
        return self._collection.data

    @property
    def particle_data(self):
        return self._collection.particle_data

    @property
    def idgen(self):
        return self._idgen

    @property
    def c_lib_register(self):
        return self._c_lib_register

    @property
    def error_particles(self):
        """Get an iterator over all particles that are in an error state.

        :return: Collection iterator over error particles.
        """
        err_particles = [
            ndata.data for ndata in self._collection
            if ndata.isvalid() and ndata.data.state not in [StateCode.Success, StateCode.Evaluate]]
        return err_particles

    @property
    def num_error_particles(self):
        """Get the number of particles that are in an error state.

        :return: The number of error particles.
        """
        return int(np.sum([True for ndata in self._collection if ndata.isvalid() and ndata.data.state not in [StateCode.Success, StateCode.Evaluate]]))

    def __iter__(self):
        """
        Returns an Iterator that allows for forward iteration over the
        elements in the ParticleSet (e.g. `for p in pset:`).
        """
        return super(ParticleSetNodes, self).__iter__()

    def iterator(self):
        """
        :returns forward iterator for ParticleSetNodes, used for a 'for'-loop, in a forward-manner
        """
        return super(ParticleSetNodes, self).iterator()

    def get_index(self, ndata):
        """
        :arg ndata: the Node object for which the index is to be retrieved
        :returns index of the requested Node
        """
        return self._collection.get_index_by_node(ndata)

    def get(self, id):
        """
        :arg id: Particle ID that is searched for
        :returns Particle with the requested ID
        """
        if type(id) not in [np.int64, np.uint64]:
            logger.warning("ParticleSetNodes.get() - provided ID as 32-bit number. Casting ID to 64-bit. Are you sure you intended to access the particle set by ID ?")
            id = np.int64(id)
        return self._collection.get_single_by_ID(id)

    def get_by_index(self, index):
        """
        This function returns the Particle object at a certain index.

        For a node-based particle set, the user is discouraged to use this functions as it performs very slowly due
        to its non-indexable collection. It is advised to use the "get(id)" function, which is more performant.

        :arg index: index of the object to be retrieved
        :returns Particle object (SciPy- or JIT) at the indexed location
        """
        return self._collection.get_single_by_index(index)

    def __getattr__(self, name):
        """
        Access a single property of all particles.

        :param name: name of the property
        :returns list of value of requested attribute
        """
        for v in self._collection.ptype.variables:
            if v.name == name:
                return getattr(self._collection, name)
        if name in self.__dict__ and name[0] != '_':
            return self.__dict__[name]
        else:
            return None

    def __getitem__(self, key):
        """
        :arg key: index (32-bit integer) or id (64-bit integer) of the requested particle
        returns: Particle of the requested key
        """
        return self._collection[key]

    def __setitem__(self, key, value):
        """
        Sets the 'data' portion of the Node list. Replacing the Node itself is error-prone,
        but it is possible to replace the data container (i.e. the particle data) or a specific
        Node.
        :param key: index (int; np.int32) or Node
        :param value: particle data of the particle class
        :return: modified Node
        """
        self._collection[key].set_data(value)
        return self._collection[key]

    def __iadd__(self, pset):
        """Add particles to the ParticleSet. Note that this is an
        incremental add, the particles will be added to the ParticleSet
        on which this function is called.

        :param particles: Another ParticleSet containing particles to add
                          to this one.
        :return: The current ParticleSet
        """
        if isinstance(pset, type(self)):
            self._collection += pset.collection
        elif isinstance(pset, BaseParticleSet):
            self._collection.merge_collection(pset.collection)
        else:
            pass
        return self

    def add(self, value):
        """Add particles to the ParticleSet. Note that this is an
        incremental add, the particles will be added to the ParticleSet
        on which this function is called.

        :param particles: Another ParticleSet, an numpy.ndarray or a particle
                          to add to this one.
        :return: The current ParticleSet
        """
        if isinstance(value, type(self)):
            self._collection.merge_same(value.collection)
        elif isinstance(value, BaseParticleSet):
            self._collection.merge_collection(value.collection)
        elif isinstance(value, np.ndarray) or isinstance(value, dict) or isinstance(value, list) or isinstance(value, tuple):
            self._collection.add_multiple(value)
        elif isinstance(value, ScipyParticle):
            self._collection.add_single(value)

    def split_by_index(self, indices):
        """
        This function splits this collection into two disect equi-structured collections using the indices as subset.
        The reason for it can, for example, be that the set exceeds a pre-defined maximum number of elements, which for
        performance reasons mandates a split.

        The function shall return the newly created or extended Particle collection, i.e. either the collection that
        results from a collection split or this very collection, containing the newly-split particles.

        :arg indices: indices of elements within the particle set that are to be split off
        :returns a new ParticleSetNodes containing the split-off elements
        """
        super().split_by_index(indices)
        assert len(self._collection) > 0
        result = ParticleSetNodes(pclass=self.pclass, lon=[], lat=[], time=[], lonlatdepth_dtype=self._collection.lonlatdepth_dtype, fieldset=self.fieldset, pid_orig=None, idgen=self._collection._idgen, c_lib_register=self._collection._c_lib_register)
        for index in sorted(indices, reverse=True):  # pop-based process needs to start from the back of the queue
            ndata = self._collection.pop_single_by_index(index=index)
            pdata = ndata.data
            ndata.set_data(None)
            del ndata
            result.add(pdata)
        return result

    def split_by_id(self, ids):
        """
        This function splits this collection into two disect equi-structured collections using the indices as subset.
        The reason for it can, for example, be that the set exceeds a pre-defined maximum number of elements, which for
        performance reasons mandates a split.

        The function shall return the newly created or extended Particle collection, i.e. either the collection that
        results from a collection split or this very collection, containing the newly-split particles.

        :arg ids: IDs of elements within the particle set that are to be split-off
        :returns a new ParticleSetNodes containing the split-off elements
        """
        super().split_by_id(ids)
        assert len(self._collection) > 0
        result = ParticleSetNodes(pclass=self.pclass, lon=[], lat=[], time=[], lonlatdepth_dtype=self._collection.lonlatdepth_dtype, fieldset=self.fieldset, pid_orig=None, idgen=self._collection._idgen, c_lib_register=self._collection._c_lib_register)
        for id in ids:
            ndata = self._collection.pop_single_by_ID(id)
            pdata = ndata.data
            ndata.set_data(None)
            del ndata
            result.add(pdata)
        return result

    def push(self, pdata, deepcopy_elem=False):
        """
        Pushes a given particle object into this particle set.

        :arg padata: particle object to be pushed
        :arg deepcopy_elem: boolean if the object shall be copied (True) or just referenced (False)
        :returns index, i.e. position of the new element
        """
        if deepcopy_elem:
            self._collection.push(self._collection.pclass(pdata))
        else:
            self._collection.push(pdata)

    def __isub__(self, pset):
        """
        This method performs an incremental removal of the equi-structured ParticleSet, such to allow

        a -= b,

        with 'a' and 'b' begin the two equi-structured objects (or: 'b' being and individual object).
        This operation is equal to an in-place removal of (an) element(s).

        :arg other: a single Particle or a collection of objects or keys that are to be removed.
        """
        if isinstance(pset, type(self)):
            self._collection -= pset.collection
        elif isinstance(pset, BaseParticleSet):
            self._collection.remove_collection(pset.collection)
        else:
            pass
        return self

    def remove(self, value):
        """
        Removes a specific Node from the list. The Node can either be given directly or determined via its index
        or it's data package (i.e. particle data). When using the index, note though that Nodes are shifting
        (non-coherent indices), so the reliable method is to provide the Node to-be-removed directly
        (similar to an iterator in C++).
        :param ndata: Node object, Particle object or int index to the Node to-be-removed
        """
        if isinstance(value, type(self)):
            self._collection.remove_same(value.data)
        elif isinstance(value, BaseParticleSet):
            self._collection.remove_collection(value.data)
        elif isinstance(value, ScipyParticle):
            self._collection.remove_single_by_object(value)

    def pop(self, idx=-1, deepcopy_elem=False):
        """
        Pops (i.e. retrieves and removes) the particle at the requested ID

        :arg idx: index of the particle that is requested
        :arg deepcopy_elem: boolean,setting if the returned element is just referenced (False) or firmly copied and recreated (True)
        :returns requested Particle object
        """
        return self._collection.data.pop(idx, deepcopy_elem)

    def get_deleted_item_indices(self):
        """
        Obtains indices of particles marked with a 'delete'-status
        :returns indices of to-be-deleted particles
        """
        return self._collection.get_deleted_item_indices()

    def get_deleted_item_IDs(self):
        """
        Obtains IDs of particles marked with a 'delete'-status
        :returns IDs of to-be-deleted particles
        """
        return self._collection.get_deleted_item_IDs()

    def remove_indices(self, indices):
        """
        Renamed forwarding method to 'remove_items_by_indices' (as it is semantically more consistent).
        :arg indices: 32-bit integer indices of the particles that are to be removed
        """
        self.remove_items_by_indices(indices)

    def remove_items_by_indices(self, indices):
        """
        Method to remove particles from the ParticleSet, based on their `indices`
        :arg indices: 32-bit integer indices of the particles that are to be removed
        """
        self._collection.remove_multi_by_indices(indices)

    def remove_deleted_items(self):
        """
        Removes all items that are marked with a 'delete'-state
        """
        self._collection.remove_deleted()

    def remove_booleanvector(self, indices):
        """
        Method to remove particles from the ParticleSet, based on an array of booleans.
        :arg indices: a boolean numpy.ndarray that marks all to-be-deleted elements as 'True'
        """
        indices = np.nonzero(indices)[0]
        self.remove_items_by_indices(indices)

    def cstruct(self):
        raise NotImplementedError("A node-based collection does not comprise into a contiguous-memory structure (i.e. cstruct). For using the structure in ctypes, please just start with 'pset.begin()' the ctypes-function.")

    @property
    def ctypes_struct(self):
        raise NotImplementedError("A node-based collection does not comprise into a contiguous-memory structure (i.e. cstruct). For using the structure in ctypes, please just start with 'pset.begin()' the ctypes-function.")

    def __repr__(self):
        return repr(self._collection)

    def to_dict(self, pfile, time, deleted_only=False):
        """
        Convert all Particle data from one time step to a python dictionary.
        :param pfile: ParticleFile object requesting the conversion
        :param time: Time at which to write ParticleSet
        :param deleted_only: Flag to write only the deleted Particles or one of the following options:
            i) boolean [True, False], where if 'True', we gather deleted indices internally
            ii) list or np.array (type: [u]int[32]) of deleted indices to write
            iii) list or np.array (type: [u]int64) of deleted IDs to write
            iv) list of type(Node or derivatives) of deleted nodes to write
            v) list of type(ScipyParticle or derivatives) of deleted Particles to write
        :returns two dictionaries: one for all variables to be written each outputdt,
         and one for all variables to be written once
        """
        return self._collection.toDictionary(pfile=pfile, time=time,
                                             deleted_only=deleted_only)

    @classmethod
    def monte_carlo_sample(cls, start_field, size, mode='monte_carlo'):
        """
        Converts a starting field into a monte-carlo sample of lons and lats.

        :param start_field: :mod:`parcels.fieldset.Field` object for initialising particles stochastically (horizontally)  according to the presented density field.

        :returns list(lon), list(lat)
        """
        if mode == 'monte_carlo':
            data = start_field.data if isinstance(start_field.data, np.ndarray) else np.array(start_field.data)
            if start_field.interp_method == 'cgrid_tracer':
                p_interior = np.squeeze(data[0, 1:, 1:])
            else:  # if A-grid
                d = data
                p_interior = (d[0, :-1, :-1] + d[0, 1:, :-1] + d[0, :-1, 1:] + d[0, 1:, 1:])/4.
                p_interior = np.where(d[0, :-1, :-1] == 0, 0, p_interior)
                p_interior = np.where(d[0, 1:, :-1] == 0, 0, p_interior)
                p_interior = np.where(d[0, 1:, 1:] == 0, 0, p_interior)
                p_interior = np.where(d[0, :-1, 1:] == 0, 0, p_interior)
            p = np.reshape(p_interior, (1, p_interior.size))
            inds = np.random.choice(p_interior.size, size, replace=True, p=p[0] / np.sum(p))
            xsi = np.random.uniform(size=len(inds))
            eta = np.random.uniform(size=len(inds))
            j, i = np.unravel_index(inds, p_interior.shape)
            grid = start_field.grid
            lon, lat = ([], [])
            if grid.gtype in [GridCode.RectilinearZGrid, GridCode.RectilinearSGrid]:
                lon = grid.lon[i] + xsi * (grid.lon[i + 1] - grid.lon[i])
                lat = grid.lat[j] + eta * (grid.lat[j + 1] - grid.lat[j])
            else:
                lons = np.array([grid.lon[j, i], grid.lon[j, i+1], grid.lon[j+1, i+1], grid.lon[j+1, i]])
                if grid.mesh == 'spherical':
                    lons[1:] = np.where(lons[1:] - lons[0] > 180, lons[1:]-360, lons[1:])
                    lons[1:] = np.where(-lons[1:] + lons[0] > 180, lons[1:]+360, lons[1:])
                lon = (1-xsi)*(1-eta) * lons[0] +\
                    xsi*(1-eta) * lons[1] +\
                    xsi*eta * lons[2] +\
                    (1-xsi)*eta * lons[3]
                lat = (1-xsi)*(1-eta) * grid.lat[j, i] +\
                    xsi*(1-eta) * grid.lat[j, i+1] +\
                    xsi*eta * grid.lat[j+1, i+1] +\
                    (1-xsi)*eta * grid.lat[j+1, i]
            return list(lon), list(lat)
        else:
            raise NotImplementedError('Mode %s not implemented. Please use "monte carlo" algorithm instead.' % mode)

    @classmethod
    def from_list(cls, fieldset, pclass, lon, lat, depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None, **kwargs):
        """Initialise the ParticleSet from lists of lon and lat

        :param fieldset: :mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        :param pclass: mod:`parcels.particle.JITParticle` or :mod:`parcels.particle.ScipyParticle`
                 object that defines custom particle
        :param lon: List of initial longitude values for particles
        :param lat: List of initial latitude values for particles
        :param depth: Optional list of initial depth values for particles. Default is 0m
        :param time: Optional list of start time values for particles. Default is fieldset.U.time[0]
        :param repeatdt: Optional interval (in seconds) on which to repeat the release of the ParticleSet
        :param lonlatdepth_dtype: Floating precision for lon, lat, depth particle coordinates.
               It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
               and np.float64 if the interpolation method is 'cgrid_velocity'
        Other Variables can be initialised using further arguments (e.g. v=... for a Variable named 'v')
       """
        c_lib_register = kwargs.pop("c_lib_register", None)
        if c_lib_register is None:
            logger.warning("A node-based particle set requires a global-context 'LibraryRegisterC'. Creating default 'LibraryRegisterC'.")
            c_lib_register = LibraryRegisterC()
        idgen = kwargs.pop("idgen", None)
        if idgen is None:
            logger.warning("A node-based particle set requires a global-context ID generator. Creating a default ID generator.")
            idgen = GenerateID_Service(SequentialIdGenerator)

        return cls(fieldset=fieldset, pclass=pclass, lon=lon, lat=lat, depth=depth, time=time, repeatdt=repeatdt, lonlatdepth_dtype=lonlatdepth_dtype, c_lib_register=c_lib_register, idgen=idgen, **kwargs)

    @classmethod
    def from_line(cls, fieldset, pclass, start, finish, size, depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None, **kwargs):
        """Initialise the ParticleSet from start/finish coordinates with equidistant spacing
        Note that this method uses simple numpy.linspace calls and does not take into account
        great circles, so may not be a exact on a globe

        :param fieldset: :mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        :param pclass: mod:`parcels.particle.JITParticle` or :mod:`parcels.particle.ScipyParticle`
                 object that defines custom particle
        :param start: Starting point for initialisation of particles on a straight line.
        :param finish: End point for initialisation of particles on a straight line.
        :param size: Initial size of particle set
        :param depth: Optional list of initial depth values for particles. Default is 0m
        :param time: Optional start time value for particles. Default is fieldset.U.time[0]
        :param repeatdt: Optional interval (in seconds) on which to repeat the release of the ParticleSet
        :param lonlatdepth_dtype: Floating precision for lon, lat, depth particle coordinates.
               It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
               and np.float64 if the interpolation method is 'cgrid_velocity'
        """
        c_lib_register = kwargs.pop("c_lib_register", None)
        if c_lib_register is None:
            logger.warning("A node-based particle set requires a global-context 'LibraryRegisterC'. Creating default 'LibraryRegisterC'.")
            c_lib_register = LibraryRegisterC()
        idgen = kwargs.pop("idgen", None)
        if idgen is None:
            logger.warning("A node-based particle set requires a global-context ID generator. Creating a default ID generator.")
            idgen = GenerateID_Service(SequentialIdGenerator)

        lon = np.linspace(start[0], finish[0], size)
        lat = np.linspace(start[1], finish[1], size)
        if type(depth) in [int, float]:
            depth = [depth] * size
        return cls(fieldset=fieldset, pclass=pclass, lon=lon, lat=lat, depth=depth, time=time, repeatdt=repeatdt, lonlatdepth_dtype=lonlatdepth_dtype, c_lib_register=c_lib_register, idgen=idgen)

    @classmethod
    def from_field(cls, fieldset, pclass, start_field, size, mode='monte_carlo', depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None, **kwargs):
        """Initialise the ParticleSet randomly drawn according to distribution from a field

        :param fieldset: :mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        :param pclass: mod:`parcels.particle.JITParticle` or :mod:`parcels.particle.ScipyParticle`
                 object that defines custom particle
        :param start_field: Field for initialising particles stochastically (horizontally)  according to the presented density field.
        :param size: Initial size of particle set
        :param mode: Type of random sampling. Currently only 'monte_carlo' is implemented
        :param depth: Optional list of initial depth values for particles. Default is 0m
        :param time: Optional start time value for particles. Default is fieldset.U.time[0]
        :param repeatdt: Optional interval (in seconds) on which to repeat the release of the ParticleSet
        :param lonlatdepth_dtype: Floating precision for lon, lat, depth particle coordinates.
               It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
               and np.float64 if the interpolation method is 'cgrid_velocity'
        """
        c_lib_register = kwargs.pop("c_lib_register", None)
        if c_lib_register is None:
            logger.warning("A node-based particle set requires a global-context 'LibraryRegisterC'. Creating default 'LibraryRegisterC'.")
            c_lib_register = LibraryRegisterC()
        idgen = kwargs.pop("idgen", None)
        if idgen is None:
            logger.warning("A node-based particle set requires a global-context ID generator. Creating a default ID generator.")
            idgen = GenerateID_Service(SequentialIdGenerator)

        lon, lat = cls.monte_carlo_sample(start_field, size, mode)

        return cls(fieldset=fieldset, pclass=pclass, lon=lon, lat=lat, depth=depth, time=time, lonlatdepth_dtype=lonlatdepth_dtype, repeatdt=repeatdt, c_lib_register=c_lib_register, idgen=idgen)

    @classmethod
    def from_particlefile(cls, fieldset, pclass, filename, restart=False, restarttime=None, repeatdt=None, lonlatdepth_dtype=None, **kwargs):
        """Initialise the ParticleSet from a netcdf ParticleFile.
        This creates a new ParticleSet based on the last locations and time of all particles
        in the netcdf ParticleFile. Particle IDs are *not* preserved (as IDs are auto-generated).

        :param fieldset: :mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        :param pclass: mod:`parcels.particle.JITParticle` or :mod:`parcels.particle.ScipyParticle`
                 object that defines custom particle
        :param filename: Name of the particlefile from which to read initial conditions
        :param restart: Boolean to signal if pset is used for a restart (default is False).
               Not applicable to node-based particle sets as ID's are overriden by default.
        :param repeatdt: Optional interval (in seconds) on which to repeat the release of the ParticleSet
        :param lonlatdepth_dtype: Floating precision for lon, lat, depth particle coordinates.
               It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
               and np.float64 if the interpolation method is 'cgrid_velocity'
        """
        c_lib_register = kwargs.pop("c_lib_register", None)
        if c_lib_register is None:
            logger.warning("A node-based particle set requires a global-context 'LibraryRegisterC'. Creating default 'LibraryRegisterC'.")
            c_lib_register = LibraryRegisterC()
        idgen = kwargs.pop("idgen", None)
        if idgen is None:
            logger.warning("A node-based particle set requires a global-context ID generator. Creating a default ID generator.")
            idgen = GenerateID_Service(SequentialIdGenerator)

        if repeatdt is not None:
            logger.warning('Note that the `repeatdt` argument is not retained from %s, and that '
                           'setting a new repeatdt will start particles from the _new_ particle '
                           'locations.' % filename)

        pfile = xr.open_dataset(str(filename), decode_cf=True)
        pfile_vars = [v for v in pfile.data_vars]

        pvars = {}
        to_write = {}
        for v in pclass.getPType().variables:
            if v.name in pfile_vars:
                pvars[v.name] = np.ma.filled(pfile.variables[v.name], np.nan)
            elif v.name not in ['xi', 'yi', 'zi', 'ti', 'dt', '_next_dt', 'depth', 'id', 'index', 'state'] \
                    and v.to_write:
                # , 'fileid'
                raise RuntimeError('Variable %s is in pclass but not in the particlefile' % v.name)
            to_write[v.name] = v.to_write
        pvars['depth'] = np.ma.filled(pfile.variables['z'], np.nan)
        pvars['id'] = np.ma.filled(pfile.variables['trajectory'], np.nan)

        if isinstance(pvars['time'][0, 0], np.timedelta64):
            pvars['time'] = np.array([t/np.timedelta64(1, 's') for t in pvars['time']])

        if restarttime is None:
            restarttime = np.nanmax(pvars['time'])
        elif callable(restarttime):
            restarttime = restarttime(pvars['time'])
        else:
            restarttime = restarttime

        inds = np.where(pvars['time'] == restarttime)
        for v in pvars:
            if to_write[v] is True:
                pvars[v] = pvars[v][inds]
            elif to_write[v] == 'once':
                pvars[v] = pvars[v][inds[0]]
            if v not in ['lon', 'lat', 'depth', 'time', 'id', 'index']:
                kwargs[v] = pvars[v]
        pfile.close()
        pvars['id'] = None

        return cls(fieldset=fieldset, pclass=pclass, lon=pvars['lon'], lat=pvars['lat'],
                   depth=pvars['depth'], pid_orig=pvars['id'], time=pvars['time'],
                   lonlatdepth_dtype=lonlatdepth_dtype, repeatdt=repeatdt, c_lib_register=c_lib_register, idgen=idgen, **kwargs)

    def execute(self, pyfunc=AdvectionRK4, pyfunc_inter=None, endtime=None, runtime=None, dt=1.,
                moviedt=None, recovery=None, output_file=None, movie_background_field=None,
                verbose_progress=None, postIterationCallbacks=None, callbackdt=None):
        """Execute a given kernel function over the particle set for
        multiple timesteps. Optionally also provide sub-timestepping
        for particle output.

        :param pyfunc: Kernel function to execute. This can be the name of a
                       defined Python function or a :class:`parcels.kernel.BaseKernel` object.
                       Kernels can be concatenated using the + operator
        :param endtime: End time for the timestepping loop.
                        It is either a datetime object or a positive double.
        :param runtime: Length of the timestepping loop. Use instead of endtime.
                        It is either a timedelta object or a positive double.
        :param dt: Timestep interval to be passed to the kernel.
                   It is either a timedelta object or a double.
                   Use a negative value for a backward-in-time simulation.
        :param moviedt:  Interval for inner sub-timestepping (leap), which dictates
                         the update frequency of animation.
                         It is either a timedelta object or a positive double.
                         None value means no animation.
        :param output_file: :mod:`parcels.particlefile.ParticleFile` object for particle output
        :param recovery: Dictionary with additional `:mod:parcels.tools.error`
                         recovery kernels to allow custom recovery behaviour in case of
                         kernel errors.
        :param movie_background_field: field plotted as background in the movie if moviedt is set.
                                       'vector' shows the velocity as a vector field.
        :param verbose_progress: Boolean for providing a progress bar for the kernel execution loop.
        :param postIterationCallbacks: (Optional) Array of functions that are to be called after each iteration (post-process, non-Kernel)
        :param callbackdt: (Optional, in conjecture with 'postIterationCallbacks) timestep inverval to (latestly) interrupt the running kernel and invoke post-iteration callbacks from 'postIterationCallbacks'
        """

        # check if pyfunc has changed since last compile. If so, recompile
        if self._kernel is None or (self._kernel.pyfunc is not pyfunc and self._kernel is not pyfunc):
            # Generate and store Kernel
            if isinstance(pyfunc, self.kernelclass):
                if pyfunc.ptype.name == self.collection.ptype.name:
                    self._kernel = pyfunc
                elif pyfunc.pyfunc is not None:
                    self._kernel = self.Kernel(pyfunc.pyfunc)
                else:
                    raise RuntimeError("Cannot reuse concatenated kernels that were compiled for different particle types. Please rebuild the 'pyfunc' or 'kernel' given to the execute function.")
            else:
                self._kernel = self.Kernel(pyfunc)
            # Prepare JIT kernel execution
            if self._collection.ptype.uses_jit:
                self._kernel.remove_lib()
                cppargs = ['-DDOUBLE_COORD_VARIABLES'] if self._collection.lonlatdepth_dtype == np.float64 else None
                self._kernel.compile(compiler=GNUCompiler_MS(cppargs=cppargs, incdirs=[os.path.join(get_package_dir(), 'include'), os.path.join(get_package_dir(), 'nodes'), "."], tmp_dir=get_cache_dir()))
                self._kernel.load_lib()

        # Convert all time variables to seconds
        if isinstance(endtime, delta):
            raise RuntimeError('endtime must be either a datetime or a double')
        if isinstance(endtime, datetime):
            endtime = np.datetime64(endtime)
        elif isinstance(endtime, cftime.datetime):
            endtime = self.time_origin.reltime(endtime)

        if isinstance(endtime, np.datetime64):
            if self.time_origin.calendar is None:
                raise NotImplementedError('If fieldset.time_origin is not a date, execution endtime must be a double')
            endtime = self.time_origin.reltime(endtime)

        if isinstance(runtime, delta):
            runtime = runtime.total_seconds()
        if isinstance(dt, delta):
            dt = dt.total_seconds()
        outputdt = output_file.outputdt if output_file else np.infty
        if isinstance(outputdt, delta):
            outputdt = outputdt.total_seconds()

        if isinstance(moviedt, delta):
            moviedt = moviedt.total_seconds()
        if isinstance(callbackdt, delta):
            callbackdt = callbackdt.total_seconds()

        assert runtime is None or runtime >= 0, 'runtime must be positive'
        assert outputdt is None or outputdt >= 0, 'outputdt must be positive'
        assert moviedt is None or moviedt >= 0, 'moviedt must be positive'

        # Derive _starttime and endtime from arguments or fieldset defaults
        if runtime is not None and endtime is not None:
            raise RuntimeError('Only one of (endtime, runtime) can be specified')

        mintime, maxtime = self.fieldset.gridset.dimrange('time_full') if self.fieldset is not None else (0, 1)
        _starttime = min([n.data.time for n in self._collection.data if not np.isnan(n.data.time)]) if dt >= 0 else max([n.data.time for n in self._collection.data if not np.isnan(n.data.time)])
        _fieldstarttime = mintime if dt >= 0 else maxtime
        _starttime = _fieldstarttime if _starttime is None or np.isnan(_starttime) else _starttime
        if self.repeatdt is not None and (self.repeat_starttime is None or np.isnan(self.repeat_starttime)):
            self.repeat_starttime = _starttime
        if runtime is not None:
            endtime = _starttime + runtime * np.sign(dt)
        elif endtime is None:
            mintime, maxtime = self._fieldset.gridset.dimrange('time_full') if self._fieldset is not None else (0, 1)
            endtime = maxtime if dt >= 0 else mintime

        execute_once = False
        if abs(endtime - _starttime) < 1e-5 or dt == 0 or runtime == 0:
            dt = 0
            runtime = 0
            endtime = _starttime
            logger.warning_once("dt or runtime are zero, or endtime is equal to Particle.time. "
                                "The kernels will be executed once, without incrementing time")
            execute_once = True

        # ==== Initialise particle timestepping
        _starttime = _starttime[0] if isinstance(_starttime, np.ndarray) or type(_starttime) in [list, tuple] else _starttime
        ndata = self._collection.begin()
        while ndata is not None:
            # ==== we need to skip here deleted nodes that have been queued for deletion, but are still bound in memory ==== #
            if not ndata.isvalid():
                ndata = ndata.next
                continue
            pdata = ndata.data
            pdata.dt = dt
            if np.isnan(pdata.time):
                pdata.time = _starttime
            ndata.set_data(pdata)
            ndata = ndata.next

        # First write output_file, because particles could have been added
        if output_file is not None:
            output_file.write(self, _starttime)
        else:
            logger.warning("No output file defined.")

        if moviedt:
            self.show(field=movie_background_field, show_time=_starttime, animation=True)
        if moviedt is None:
            moviedt = np.infty
        if callbackdt is None:
            interupt_dts = [np.infty, moviedt, outputdt]
            if self.repeatdt is not None:
                interupt_dts.append(self.repeatdt)
            callbackdt = np.min(np.array(interupt_dts))

        time = _starttime
        if self.repeatdt:
            next_prelease = self.repeat_starttime + (abs(time - self.repeat_starttime) // self.repeatdt + 1) * self.repeatdt * np.sign(dt)
        else:
            next_prelease = np.infty if dt > 0 else - np.infty
        next_output = time + outputdt if dt > 0 else time - outputdt

        next_movie = time + moviedt if dt > 0 else time - moviedt
        next_callback = time + callbackdt if dt > 0 else time - callbackdt

        next_input = self._fieldset.computeTimeChunk(time, np.sign(dt)) if self._fieldset is not None else np.inf

        tol = 1e-12

        pbar = None
        walltime_start = None
        if verbose_progress is None:
            walltime_start = time_module.time()
        if verbose_progress:
            pbar = self._create_progressbar_(_starttime, endtime)

        while (time < endtime and dt > 0) or (time > endtime and dt < 0) or dt == 0:

            if verbose_progress is None and time_module.time() - walltime_start > 10:
                # Showing progressbar if runtime > 10 seconds
                if output_file:
                    logger.info('Temporary output files are stored in %s.' % output_file.tempwritedir_base)
                    logger.info('You can use "parcels_convert_npydir_to_netcdf %s" to convert these '
                                'to a NetCDF file during the run.' % output_file.tempwritedir_base)
                pbar = self._create_progressbar_(_starttime, endtime)
                verbose_progress = True

            if dt > 0:
                time = min(next_prelease, next_input, next_output, next_movie, next_callback, endtime)
            else:
                time = max(next_prelease, next_input, next_output, next_movie, next_callback, endtime)
            self._kernel.execute(self, endtime=time, dt=dt, recovery=recovery, output_file=output_file, execute_once=execute_once)
            if abs(time-next_prelease) < tol:
                ngrids = self.fieldset.gridset.size if self.fieldset is not None else 0
                add_iter = 0
                while add_iter < len(self.repeatlon):
                    gen_id = None if self.repeatpid is None or type(self.repeatpid) not in [list, tuple, dict, np.ndarray] else self.repeatid[add_iter]
                    lon = self.repeatlon[add_iter]
                    lat = self.repeatlat[add_iter]
                    pdepth = self.repeatdepth[add_iter]
                    ptime = time if self.release_starttime is None else self.release_starttime
                    pid = self._idgen.nextID(lon, lat, pdepth, ptime) if gen_id is None else gen_id
                    pdata = self.repeatpclass(lon, lat, pid=pid, ngrids=ngrids, depth=pdepth, time=ptime)
                    pdata.dt = dt
                    # Set other Variables if provided
                    for kwvar in self.repeatkwargs:
                        if isinstance(kwvar, Field):
                            continue
                        if not hasattr(pdata, kwvar):
                            raise RuntimeError('Particle class does not have Variable %s' % kwvar)
                        setattr(pdata, kwvar, self.repeatkwargs[kwvar][add_iter])
                    self.add(pdata)
                    add_iter += 1
                next_prelease += self.repeatdt * np.sign(dt)
            if abs(time-next_output) < tol or execute_once:
                for fld in self._fieldset.get_fields():
                    if hasattr(fld, 'to_write') and fld.to_write:
                        if fld.grid.tdim > 1:
                            raise RuntimeError('Field writing during execution only works for Fields with one snapshot in time')
                        fldfilename = str(output_file.name).replace('.nc', '_%.4d' % fld.to_write)
                        fld.write(fldfilename)
                        fld.to_write += 1
                if output_file is not None:
                    output_file.write(self, time)
                next_output += outputdt * np.sign(dt)
            if abs(time-next_movie) < tol:
                self.show(field=movie_background_field, show_time=time, animation=True)
                next_movie += moviedt * np.sign(dt)
            # ==== insert post-process here to also allow for memory clean-up via external func ==== #
            if abs(time-next_callback) < tol:
                if postIterationCallbacks is not None:
                    for extFunc in postIterationCallbacks:
                        extFunc()
                next_callback += callbackdt * np.sign(dt)
            if time != endtime:
                next_input = self.fieldset.computeTimeChunk(time, dt)
            if dt == 0:
                break
            if verbose_progress:
                pbar.update(abs(time - pbar.prevtime))
                pbar.prevtime = time
        if output_file is not None:
            output_file.write(self, time)
        if verbose_progress:
            pbar.close()

    def show(self, with_particles=True, show_time=None, field=None, domain=None, projection=None,
             land=True, vmin=None, vmax=None, savefile=None, animation=False, **kwargs):
        """Method to 'show' a Parcels ParticleSet

        :param with_particles: Boolean whether to show particles
        :param show_time: Time at which to show the ParticleSet
        :param field: Field to plot under particles (either None, a Field object, or 'vector')
        :param domain: dictionary (with keys 'N', 'S', 'E', 'W') defining domain to show
        :param projection: type of cartopy projection to use (default PlateCarree)
        :param land: Boolean whether to show land. This is ignored for flat meshes
        :param vmin: minimum colour scale (only in single-plot mode)
        :param vmax: maximum colour scale (only in single-plot mode)
        :param savefile: Name of a file to save the plot to
        :param animation: Boolean whether result is a single plot, or an animation
        """
        from parcels.plotting import plotparticles
        plotparticles(particles=self, with_particles=with_particles, show_time=show_time, field=field, domain=domain,
                      projection=projection, land=land, vmin=vmin, vmax=vmax, savefile=savefile, animation=animation, **kwargs)

    def density(self, field_name=None, particle_val=None, relative=False, area_scale=False):
        """Method to calculate the density of particles in a ParticleSet from their locations,
        through a 2D histogram.

        :param field_name: Optional :mod:`parcels.field.Field` object to calculate the histogram
                           on. Default is `fieldset.U`
        :param particle_val: Optional numpy-array of values to weigh each particle with,
                             or string name of particle variable to use weigh particles with.
                             Default is None, resulting in a value of 1 for each particle
        :param relative: Boolean to control whether the density is scaled by the total
                         weight of all particles. Default is False
        :param area_scale: Boolean to control whether the density is scaled by the area
                           (in m^2) of each grid cell. Default is False
        """

        field_name = field_name if field_name else "U"
        field = getattr(self._fieldset, field_name)

        f_str = """
def search_kernel(particle, fieldset, time):
    x = fieldset.{}[time, particle.depth, particle.lat, particle.lon]
        """.format(field_name)

        k = KernelNodes(
            self._fieldset,
            self._collection.ptype,
            funcname="search_kernel",
            funcvars=["particle", "fieldset", "time", "x"],
            funccode=f_str,
        )
        self.execute(pyfunc=k, runtime=0)
        if isinstance(particle_val, str):
            particle_val = [getattr(p, particle_val) for p in self.particles]
        else:
            particle_val = particle_val if particle_val else np.ones(len(self.particles))
        density = np.zeros((field.grid.lat.size, field.grid.lon.size), dtype=np.float32)

        for pi, ndata in enumerate(self.particles):
            p = ndata.data
            try:  # breaks if either p.xi, p.yi, p.zi, p.ti do not exist (in scipy) or field not in fieldset
                if p.ti[field.igrid] < 0:  # xi, yi, zi, ti, not initialised
                    raise('error')
                xi = p.xi[field.igrid]
                yi = p.yi[field.igrid]
            except:
                _, _, _, xi, yi, _ = field.search_indices(p.lon, p.lat, p.depth, 0, 0, search2D=True)
            density[yi, xi] += particle_val[pi]

        if relative:
            density /= np.sum(particle_val)
        if area_scale:
            density /= field.cell_areas()
        return density

    def Kernel(self, pyfunc, c_include="", delete_cfiles=True):
        """
        Wrapper method to convert a `pyfunc` into a :class:`parcels.kernel.Kernel` object
        based on `fieldset` and `ptype` of the ParticleSet
        :param delete_cfiles: Boolean whether to delete the C-files after compilation in JIT mode (default is True)
        :returns a Kernel object specific to this particle set
        """
        return self.kernelclass(self.fieldset, self.collection.ptype, pyfunc=pyfunc, c_include=c_include, delete_cfiles=delete_cfiles)

    def ParticleFile(self, *args, **kwargs):
        """
        Wrapper method to initialise a :class:`parcels.particlefile.ParticleFile` object from the ParticleSet.
        """
        return ParticleFileNodes(*args, particleset=self, **kwargs)

    def set_variable_write_status(self, var, write_status):
        """
        Method to set the write status of a Variable
        :param var: Name of the variable (string)
        :param write_status: Write status of the variable (True, False or
                             'once')
        """
        self._collection.set_variable_write_status(var, write_status)

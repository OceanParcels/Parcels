from datetime import date
from datetime import datetime
from datetime import timedelta as delta

import sys
import numpy as np
import xarray as xr
from ctypes import c_void_p
from copy import copy

from parcels.grid import GridCode
from parcels.field import NestedField
from parcels.field import SummedField
from parcels.kernel.kernelaos import KernelAOS
from parcels.particle import Variable, ScipyParticle, JITParticle # NOQA
from parcels.particlefile.particlefileaos import ParticleFileAOS
from parcels.tools.statuscodes import StateCode, OperationCode  # NOQA
from parcels.particleset.baseparticleset import BaseParticleSet
from parcels.collection.collectionaos import ParticleCollectionAOS
from parcels.collection.collectionaos import ParticleCollectionIteratorAOS, ParticleCollectionIterableAOS  # NOQA

from parcels.tools.converters import _get_cftime_calendars
from parcels.tools.loggers import logger
try:
    from mpi4py import MPI
except:
    MPI = None

__all__ = ['ParticleSetAOS']


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


class ParticleSetAOS(BaseParticleSet):
    """Container class for storing particle and executing kernel over them.

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

    def __init__(self, fieldset=None, pclass=JITParticle, lon=None, lat=None, depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None, pid_orig=None, **kwargs):
        super(ParticleSetAOS, self).__init__()

        # ==== first: create a new subclass of the pclass that includes the required variables ==== #
        # ==== see dynamic-instantiation trick here: https://www.python-course.eu/python3_classes_and_type.php ==== #
        class_name = "Object"+pclass.__name__
        object_class = None
        if class_name not in dir():
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
        # ==== dynamic re-classing completed ==== #
        _pclass = object_class

        self.fieldset = fieldset
        if self.fieldset is None:
            logger.warning_once("No FieldSet provided in ParticleSet generation. "
                                "This breaks most Parcels functionality")
        else:
            self.fieldset.check_complete()
        partitions = kwargs.pop('partitions', None)

        lon = np.empty(shape=0) if lon is None else _convert_to_array(lon)
        lat = np.empty(shape=0) if lat is None else _convert_to_array(lat)

        if isinstance(pid_orig, (type(None), type(False))):
            pid_orig = np.arange(lon.size)

        if depth is None:
            mindepth = self.fieldset.gridset.dimrange('depth')[0] if self.fieldset is not None else 0
            depth = np.ones(lon.size) * mindepth
        else:
            depth = _convert_to_array(depth)
        assert lon.size == lat.size and lon.size == depth.size, (
            'lon, lat, depth don''t all have the same lenghts')

        time = _convert_to_array(time)
        time = np.repeat(time, lon.size) if time.size == 1 else time

        if time.size > 0 and type(time[0]) in [datetime, date]:
            time = np.array([np.datetime64(t) for t in time])
        self.time_origin = fieldset.time_origin if self.fieldset is not None else 0
        if time.size > 0 and isinstance(time[0], np.timedelta64) and not self.time_origin:
            raise NotImplementedError('If fieldset.time_origin is not a date, time of a particle must be a double')
        time = np.array([self.time_origin.reltime(t) if _convert_to_reltime(t) else t for t in time])
        assert lon.size == time.size, (
            'time and positions (lon, lat, depth) don''t have the same lengths.')

        if lonlatdepth_dtype is None:
            if fieldset is not None:
                lonlatdepth_dtype = self.lonlatdepth_dtype_from_field_interp_method(fieldset.U)
            else:
                lonlatdepth_dtype = np.float32
        assert lonlatdepth_dtype in [np.float32, np.float64], \
            'lon lat depth precision should be set to either np.float32 or np.float64'

        for kwvar in kwargs:
            kwargs[kwvar] = _convert_to_array(kwargs[kwvar])
            assert lon.size == kwargs[kwvar].size, (
                '%s and positions (lon, lat, depth) don''t have the same lengths.' % kwvar)

        self.repeatdt = repeatdt.total_seconds() if isinstance(repeatdt, delta) else repeatdt
        if self.repeatdt:
            if self.repeatdt <= 0:
                raise('Repeatdt should be > 0')
            if time[0] and not np.allclose(time, time[0]):
                raise ('All Particle.time should be the same when repeatdt is not None')
            self.repeatpclass = pclass
            self.repeatkwargs = kwargs

        ngrids = fieldset.gridset.size if fieldset is not None else 0
        self._collection = ParticleCollectionAOS(_pclass, lon=lon, lat=lat, depth=depth, time=time, lonlatdepth_dtype=lonlatdepth_dtype, pid_orig=pid_orig, partitions=partitions, ngrid=ngrids, **kwargs)

        if self.repeatdt:
            if len(time) > 0 and time[0] is None:
                self.repeat_starttime = time[0]
            else:
                collect_time = self._collection.time
                if collect_time[0] and not np.allclose(collect_time, collect_time[0]):
                    raise ValueError('All Particle.time should be the same when repeatdt is not None')
                self.repeat_starttime = copy(collect_time[0])
            self.repeatlon = copy(self._collection.lon)
            self.repeatlat = copy(self._collection.lat)
            self.repeatdepth = copy(self._collection.depth)
            for kwvar in kwargs:
                self.repeatkwargs[kwvar] = copy(getattr(self._collection, kwvar))

        if self.repeatdt:
            if MPI and self._collection.pu_indicators is not None:
                mpi_comm = MPI.COMM_WORLD
                mpi_rank = mpi_comm.Get_rank()
                self.repeatpid = pid_orig[self._collection.pu_indicators == mpi_rank]

        self.kernel = None

    def __del__(self):
        super(ParticleSetAOS, self).__del__()

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
        In result, the particle still remains in the collection. The functional interpretation of the 'deleted' status
        is handled by 'recovery' dictionary during simulation execution.
        """
        self._collection[index].state = OperationCode.Delete

    def delete_by_ID(self, id):
        """
        This method deletes a particle from the  the collection based on its ID. It does not return the deleted item.
        Semantically, the function appears similar to the 'remove' operation. That said, the function in OceanParcels -
        instead of directly deleting the particle - just raises the 'deleted' status flag for the indexed particle.
        In result, the particle still remains in the collection. The functional interpretation of the 'deleted' status
        is handled by 'recovery' dictionary during simulation execution.
        """
        p = self._collection.get_single_by_ID(id)
        p.state = OperationCode.Delete

    def _set_particle_vector(self, name, value):
        """Set attributes of all particles to new values.

        :param name: Name of the attribute (str).
        :param value: New value to set the attribute of the particles to.
        """
        [setattr(p, name, value) for p in self._collection.data]

    def _impute_release_times(self, default):
        """Set attribute 'time' to default if encountering NaN values.

        :param default: Default release time.
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
        """Get the indices of all particles where the value of
        `variable_name` equals (one of) `compare_values`.

        :param variable_name: Name of the variable to check.
        :param compare_values: Value or list of values to compare to.
        :param invert: Whether to invert the selection. I.e., when True,
                       return all indices that do not equal (one of)
                       `compare_values`.
        :return: Numpy array of indices that satisfy the test.
        """
        compare_values = np.array([compare_values, ]) if type(compare_values) not in [list, dict, np.ndarray] else compare_values
        result = []
        if not invert:
            result = [i for i, p in enumerate(self._collection.data) if getattr(p, variable_name) in compare_values]
        else:
            result = [i for i, p in enumerate(self._collection.data) if getattr(p, variable_name) not in compare_values]
        return np.array(result)

    def indexed_subset(self, indices):
        return self._collection.data[indices]

    def populate_indices(self):
        """Pre-populate guesses of particle xi/yi indices using a kdtree.

        This is only intended for curvilinear grids, where the initial index search
        may be quite expensive.
        """
        raise NotImplementedError()

    @property
    def particle_data(self):
        return self._collection.particle_data

    @property
    def error_particles(self):
        """Get an iterator over all particles that are in an error state.

        :return: Collection iterator over error particles.
        """
        error_indices = [
            i for i, p in enumerate(self)
            if p.state not in [StateCode.Success, StateCode.Evaluate]]
        return self._collection.get_multi_by_indices(indices=error_indices)

    @property
    def num_error_particles(self):
        """Get the number of particles that are in an error state.

        :return: The number of error particles.
        """
        return np.sum([True for p in self._collection if p.state not in [StateCode.Success, StateCode.Evaluate]])

    def __iter__(self):
        return super(ParticleSetAOS, self).__iter__()

    def iterator(self):
        return super(ParticleSetAOS, self).iterator()

    def __getitem__(self, index):
        """Get a single particle by index"""
        return self._collection.get_single_by_index(index)

    def __getattr__(self, name):
        """
        Access a single property of all particles.

        :param name: name of the property
        """
        for v in self._collection.ptype.variables:
            if v.name == name:
                return getattr(self._collection, name)
        if name in self.__dict__ and name[0] != '_':
            return self.__dict__[name]
        else:
            return False

    @property
    def size(self):
        return len(self._collection)

    def __repr__(self):
        return "\n".join([str(p) for p in self])

    def __len__(self):
        return len(self._collection)

    def __sizeof__(self):
        return sys.getsizeof(self._collection)

    def cstruct(self):
        return self._collection.cstruct()

    @property
    def ctypes_struct(self):
        return self.cstruct()

    @property
    def ptype(self):
        return self._collection.ptype

    @staticmethod
    def lonlatdepth_dtype_from_field_interp_method(field):
        if type(field) in [SummedField, NestedField]:
            for f in field:
                if f.interp_method == 'cgrid_velocity':
                    return np.float64
        else:
            if field.interp_method == 'cgrid_velocity':
                return np.float64
        return np.float32

    @classmethod
    def monte_carlo_sample(cls, start_field, size, mode='monte_carlo'):
        """
        Converts a starting field into a monte-carlo sample of lons and lats.

        :param start_field: :mod:`parcels.fieldset.Field` object for initialising particles stochastically (horizontally)  according to the presented density field.

        returns list(lon), list(lat)
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
    def from_particlefile(cls, fieldset, pclass, filename, restart=True, restarttime=None, repeatdt=None, lonlatdepth_dtype=None, **kwargs):
        """Initialise the ParticleSet from a netcdf ParticleFile.
        This creates a new ParticleSet based on the last locations and time of all particles
        in the netcdf ParticleFile. Particle IDs are preserved if restart=True

        :param fieldset: :mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        :param pclass: mod:`parcels.particle.JITParticle` or :mod:`parcels.particle.ScipyParticle`
                 object that defines custom particle
        :param filename: Name of the particlefile from which to read initial conditions
        :param restart: Boolean to signal if pset is used for a restart (default is True).
               In that case, Particle IDs are preserved.
        :param repeatdt: Optional interval (in seconds) on which to repeat the release of the ParticleSet
        :param lonlatdepth_dtype: Floating precision for lon, lat, depth particle coordinates.
               It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
               and np.float64 if the interpolation method is 'cgrid_velocity'
        """
        if repeatdt is not None:
            logger.warning('Note that the `repeatdt` argument is not retained from %s, and that '
                           'setting a new repeatdt will start particles from the _new_ particle '
                           'locations.' % filename)

        pfile = xr.open_dataset(str(filename), decode_cf=True)
        pfile_vars = [v for v in pfile.data_vars]

        vars = {}
        to_write = {}
        for v in pclass.getPType().variables:
            if v.name in pfile_vars:
                vars[v.name] = np.ma.filled(pfile.variables[v.name], np.nan)
            elif v.name not in ['xi', 'yi', 'zi', 'ti', 'dt', '_next_dt', 'depth', 'id', 'fileid', 'state'] \
                    and v.to_write:
                raise RuntimeError('Variable %s is in pclass but not in the particlefile' % v.name)
            to_write[v.name] = v.to_write
        vars['depth'] = np.ma.filled(pfile.variables['z'], np.nan)
        vars['id'] = np.ma.filled(pfile.variables['trajectory'], np.nan)

        if isinstance(vars['time'][0, 0], np.timedelta64):
            vars['time'] = np.array([t/np.timedelta64(1, 's') for t in vars['time']])

        if restarttime is None:
            restarttime = np.nanmax(vars['time'])
        elif callable(restarttime):
            restarttime = restarttime(vars['time'])
        else:
            restarttime = restarttime

        inds = np.where(vars['time'] == restarttime)
        for v in vars:
            if to_write[v] is True:
                vars[v] = vars[v][inds]
            elif to_write[v] == 'once':
                vars[v] = vars[v][inds[0]]
            if v not in ['lon', 'lat', 'depth', 'time', 'id']:
                kwargs[v] = vars[v]

        if restart:
            pclass.setLastID(0)  # reset to zero offset
        else:
            vars['id'] = None

        return cls(fieldset=fieldset, pclass=pclass, lon=vars['lon'], lat=vars['lat'],
                   depth=vars['depth'], time=vars['time'], pid_orig=vars['id'],
                   lonlatdepth_dtype=lonlatdepth_dtype, repeatdt=repeatdt, **kwargs)

    def to_dict(self, pfile, time, deleted_only=False):
        """
        Convert all Particle data from one time step to a python dictionary.
        :param pfile: ParticleFile object requesting the conversion
        :param time: Time at which to write ParticleSet
        :param deleted_only: Flag to write only the deleted Particles
        returns two dictionaries: one for all variables to be written each outputdt, and one for all variables to be written once
        """
        return self._collection.toDictionary(pfile=pfile, time=time,
                                             deleted_only=deleted_only)

    def __iadd__(self, particles):
        """Add particles to the ParticleSet. Note that this is an
        incremental add, the particles will be added to the ParticleSet
        on which this function is called.

        :param particles: Another ParticleSet containing particles to add
                          to this one.
        :return: The current ParticleSet
        """
        self.add(particles)
        return self

    def add(self, particles):
        """Add particles to the ParticleSet. Note that this is an
        incremental add, the particles will be added to the ParticleSet
        on which this function is called.

        :param particles: Another ParticleSet containing particles to add
                          to this one.
        :return: The current ParticleSet
        """
        if isinstance(particles, BaseParticleSet):
            particles = particles.collection
        self._collection += particles
        return self

    def remove_indices(self, indices):
        """Method to remove particles from the ParticleSet, based on their `indices`"""
        if type(indices) in [int, np.int32, np.intp]:
            self._collection.remove_single_by_index(indices)
        else:
            self._collection.remove_multi_by_indices(indices)

    def remove_booleanvector(self, indices):
        """Method to remove particles from the ParticleSet, based on an array of booleans"""
        # indices = np.where(indices)[0]
        indices = np.nonzero(indices)[0]
        self.remove_indices(indices)

    def density(self, field_name=None, particle_val=None, relative=False, area_scale=False):
        """Method to calculate the density of particles in a ParticleSet from their locations,
        through a 2D histogram.

        :param field: Optional :mod:`parcels.field.Field` object to calculate the histogram
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
        field = getattr(self.fieldset, field_name)

        f_str = """
def search_kernel(particle, fieldset, time):
    x = fieldset.{}[time, particle.depth, particle.lat, particle.lon]
        """.format(field_name)

        k = KernelAOS(
            self.fieldset,
            self._collection.ptype,
            funcname="search_kernel",
            funcvars=["particle", "fieldset", "time", "x"],
            funccode=f_str,
        )
        self.execute(pyfunc=k, runtime=0)

        if isinstance(particle_val, str):
            particle_val = [getattr(p, particle_val) for p in self._collection]
        else:
            particle_val = particle_val if particle_val else np.ones(self.size)
        density = np.zeros((field.grid.lat.size, field.grid.lon.size), dtype=np.float32)

        for i, p in enumerate(self):
            try:  # breaks if either p.xi, p.yi, p.zi, p.ti do not exist (in scipy) or field not in fieldset
                if p.ti[field.igrid] < 0:  # xi, yi, zi, ti, not initialised
                    raise('error')
                xi = p.xi[field.igrid]
                yi = p.yi[field.igrid]
            except:
                _, _, _, xi, yi, _ = field.search_indices(p.lon, p.lat, p.depth, 0, 0, search2D=True)
            density[yi, xi] += particle_val[i]

        if relative:
            density /= np.sum(particle_val)

        if area_scale:
            density /= field.cell_areas()

        return density

    def Kernel(self, pyfunc, c_include="", delete_cfiles=True):
        """Wrapper method to convert a `pyfunc` into a :class:`parcels.kernel.Kernel` object
        based on `fieldset` and `ptype` of the ParticleSet

        :param delete_cfiles: Boolean whether to delete the C-files after compilation in JIT mode (default is True)
        """
        return KernelAOS(self.fieldset, self.collection.ptype, pyfunc=pyfunc, c_include=c_include, delete_cfiles=delete_cfiles)

    def ParticleFile(self, *args, **kwargs):
        """Wrapper method to initialise a :class:`parcels.particlefile.ParticleFile`
        object from the ParticleSet"""
        return ParticleFileAOS(*args, particleset=self, **kwargs)

    def set_variable_write_status(self, var, write_status):
        """
        Method to set the write status of a Variable
        :param var: Name of the variable (string)
        :param write_status: Write status of the variable (True, False or 'once')
        """
        self._collection.set_variable_write_status(var, write_status)

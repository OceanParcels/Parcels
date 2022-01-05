from ctypes import c_void_p
from operator import attrgetter

import numpy as np

from parcels.field import Field
from parcels.tools.statuscodes import StateCode, OperationCode
from parcels.tools.loggers import logger

__all__ = ['ScipyParticle', 'JITParticle', 'Variable', 'ScipyInteractionParticle']

indicators_64bit = [np.float64, np.uint64, np.int64, c_void_p]


class Variable(object):
    """Descriptor class that delegates data access to particle data

    :param name: Variable name as used within kernels
    :param dtype: Data type (numpy.dtype) of the variable
    :param initial: Initial value of the variable. Note that this can also be a Field object,
             which will then be sampled at the location of the particle
    :param to_write: Boolean or 'once'. Controls whether Variable is written to NetCDF file.
             If to_write = 'once', the variable will be written as a time-independent 1D array
    :type to_write: (bool, 'once', optional)
    """
    def __init__(self, name, dtype=np.float32, initial=0, to_write=True):
        self.name = name
        self.dtype = dtype
        self.initial = initial
        self.to_write = to_write

    def __get__(self, instance, cls):
        if instance is None:
            return self
        if issubclass(cls, JITParticle):
            return instance.get_cptr().__getitem__(self.name)
        else:
            return getattr(instance, "_%s" % self.name, self.initial)

    def __set__(self, instance, value):
        if isinstance(instance, JITParticle):
            instance.get_cptr().__setitem__(self.name, value)
        else:
            setattr(instance, "_%s" % (self.name,), value)

    def __repr__(self):
        return "PVar<%s|%s>" % (self.name, self.dtype)

    @property
    def size(self):
        """Size of the underlying particle struct in bytes"""
        return (8 if self.is64bit() else 4)

    def is64bit(self):
        """Check whether variable is 64-bit"""
        return True if self.dtype in indicators_64bit else False


class ParticleType(object):
    """Class encapsulating the type information for custom particles

    :arg pclass: custom, specialised Particle class used in actual simulations
    """

    def __init__(self, pclass):
        """
        ParticleType - Constructor

        :arg pclass: custom, specialised Particle class used in actual simulations
        """
        if not isinstance(pclass, type):
            raise TypeError("Class object required to derive ParticleType")
        if not issubclass(pclass, ScipyParticle):
            raise TypeError("Class object does not inherit from parcels.ScipyParticle")
        self.name = pclass.__name__
        self.uses_jit = issubclass(pclass, JITParticle)
        # Pick Variable objects out of __dict__.
        self.variables = [v for v in pclass.__dict__.values() if isinstance(v, Variable)]
        for cls in pclass.__bases__:
            if issubclass(cls, ScipyParticle):
                # Add inherited particle variables
                ptype = cls.getPType()
                for v in self.variables:
                    if v.name in [v.name for v in ptype.variables]:
                        raise AttributeError(
                            "Custom Variable name '%s' is not allowed, as it is also a built-in variable" % v.name)
                    if v.name == 'z':
                        raise AttributeError(
                            "Custom Variable name 'z' is not allowed, as it is used for depth in ParticleFile")
                self.variables = ptype.variables + self.variables
        # Sort variables with all the 64-bit first so that they are aligned for the JIT cptr
        self.variables = [v for v in self.variables if v.is64bit()] + \
                         [v for v in self.variables if not v.is64bit()]

    def __repr__(self):
        """
        :returns binary string representation of the particle type
        """
        return "PType<%s>::%s" % (self.name, self.variables)

    @property
    def _cache_key(self):
        return "-".join(["%s:%s" % (v.name, v.dtype) for v in self.variables])

    @property
    def dtype(self):
        """Numpy.dtype object that defines the C struct"""
        type_list = [(v.name, v.dtype) for v in self.variables]
        for v in self.variables:
            if v.dtype not in self.supported_dtypes:
                raise RuntimeError(str(v.dtype) + " variables are not implemented in JIT mode")
        if self.size % 8 > 0:
            # Add padding to be 64-bit aligned
            type_list += [('pad', np.float32)]
        return np.dtype(type_list)

    @property
    def size(self):
        """Size of the underlying particle struct in bytes"""
        return sum([v.size for v in self.variables])

    @property
    def supported_dtypes(self):
        """List of all supported numpy dtypes. All others are not supported"""

        # Developer note: other dtypes (mostly 2-byte ones) are not supported now
        # because implementing and aligning them in cgen.GenerableStruct is a
        # major headache. Perhaps in a later stage
        return [np.int32, np.uint32, np.int64, np.uint64, np.float32, np.double, np.float64, c_void_p]


class _Particle(object):
    """Private base class for all particle types"""
    lastID = 0  # class-level variable keeping track of last Particle ID used

    def __init__(self):
        ptype = self.getPType()
        # Explicit initialisation of all particle variables
        for v in ptype.variables:
            if isinstance(v.initial, attrgetter):
                initial = v.initial(self)
            elif isinstance(v.initial, Field):
                lon = self.getInitialValue(ptype, name='lon')
                lat = self.getInitialValue(ptype, name='lat')
                depth = self.getInitialValue(ptype, name='depth')
                time = self.getInitialValue(ptype, name='time')
                if time is None:
                    raise RuntimeError('Cannot initialise a Variable with a Field if no time provided. '
                                       'Add a "time=" to ParticleSet construction')
                if v.initial.grid.ti < 0:
                    v.initial.fieldset.computeTimeChunk(time, 0)
                initial = v.initial[time, depth, lat, lon]
                logger.warning_once("Particle initialisation from field can be very slow as it is computed in scipy mode.")
            else:
                initial = v.initial
            # Enforce type of initial value
            if v.dtype != c_void_p:
                setattr(self, v.name, v.dtype(initial))

        # Placeholder for explicit error handling
        self.exception = None

    def __del__(self):
        pass  # superclass is 'object', and object itself has no destructor, hence 'pass'

    @classmethod
    def getPType(cls):
        return ParticleType(cls)

    @classmethod
    def getInitialValue(cls, ptype, name):
        return next((v.initial for v in ptype.variables if v.name is name), None)

    @classmethod
    def setLastID(cls, offset):
        _Particle.lastID = offset


class ScipyParticle(_Particle):
    """Class encapsulating the basic attributes of a particle,
    to be executed in SciPy mode

    :param lon: Initial longitude of particle
    :param lat: Initial latitude of particle
    :param pid: externally-defined particle ID
    :param fieldset: :mod:`parcels.fieldset.FieldSet` object to track this particle on
    :param ngrids: number of grids of the related fieldset - optional alternative parameter to :param fieldset
    :param depth: Initial depth of particle
    :param time: Current time of the particle
    :param cptr: legacy parameter to ctypes-bound pointer into a C-array to store the particle information for ctypes

    Additional Variables can be added via the :Class Variable: objects
    """

    lon = Variable('lon', dtype=np.float32)
    lat = Variable('lat', dtype=np.float32)
    depth = Variable('depth', dtype=np.float32)
    time = Variable('time', dtype=np.float64)
    id = Variable('id', dtype=np.int64)
    dt = Variable('dt', dtype=np.float64, to_write=False)
    state = Variable('state', dtype=np.int32, initial=StateCode.Evaluate, to_write=False)
    next_dt = Variable('next_dt', dtype=np.float64, initial=np.nan, to_write=False)

    def __init__(self, lon, lat, pid, fieldset=None, ngrids=None, depth=0., time=0., cptr=None):
        """
        ScipyParticle - Constructor

        :param lon: Initial longitude of particle
        :param lat: Initial latitude of particle
        :param pid: externally-defined particle ID
        :param fieldset: :mod:`parcels.fieldset.FieldSet` object to track this particle on
        :param ngrids: number of grids of the related fieldset - optional alternative parameter to :param fieldset
        :param depth: Initial depth of particle
        :param time: Current time of the particle
        :param cptr: legacy parameter to ctypes-bound pointer into a C-array to store the particle information for ctypes
        """
        # Enforce default values through Variable descriptor
        type(self).lon.initial = lon
        type(self).lat.initial = lat
        type(self).depth.initial = depth
        type(self).time.initial = time
        type(self).id.initial = pid
        _Particle.lastID = max(_Particle.lastID, pid)
        type(self).dt.initial = 0.
        type(self).next_dt.initial = np.nan

        super(ScipyParticle, self).__init__()

    def __del__(self):
        """
        ScipyParticle - Destructor
        """
        super(ScipyParticle, self).__del__()

    def __repr__(self):
        """
        :returns binary string representation of the ScipyParticle
        """
        time_string = "not_yet_set" if self.time is None or np.isnan(self.time) else "%f" % self.time
        str = "P[%d](lon=%f, lat=%f, depth=%f, " % (self.id, self.lon, self.lat, self.depth)
        for var in vars(type(self)):
            if type(getattr(type(self), var)) is Variable and getattr(type(self), var).to_write is True:
                str += "%s=%f, " % (var, getattr(self, var))
        return str + "time=%s)" % time_string

    def delete(self):
        """
        This function sets the state of the particle to 'being deleted'.
        """
        self.state = OperationCode.Delete

    def set_state(self, state):
        """
        This function sets the state of the particle.

        :arg state: StatusCode, OperationCode or ErrorCode to be set for the particle
        """
        self.state = state

    def succeeded(self):
        """
        This function sets the state of the particle to 'being successfully computed'.
        """
        self.state = StateCode.Success

    def isComputed(self):
        """
        :returns if this particle has been successfully computed
        """
        return self.state == StateCode.Success

    def reset_state(self):
        """
        This functions resets the state of the particle to the default of "being ready for evaluation (i.e. computation)'
        """
        self.state = StateCode.Evaluate

    @classmethod
    def set_lonlatdepth_dtype(cls, dtype):
        cls.lon.dtype = dtype
        cls.lat.dtype = dtype
        cls.depth.dtype = dtype

    def update_next_dt(self, next_dt=None):
        if next_dt is None:
            if self.next_dt is not None and not np.isnan(self.next_dt):
                self.dt = self.next_dt
                self.next_dt = np.nan
        else:
            self.next_dt = next_dt

    def __eq__(self, other):
        """:returns if this particle object is equal to :arg other"""
        if type(self) is not type(other):
            return False
        ids_eq = (self.id == other.id)
        attr_eq = True
        attr_eq &= (self.lon == other.lon)
        attr_eq &= (self.lat == other.lat)
        attr_eq &= (self.depth == other.depth)
        attr_eq &= (self.time == other.time)
        attr_eq &= (self.dt == other.dt)
        return ids_eq and attr_eq

    def __ne__(self, other):
        """:returns if this particle object is not equal to :arg other"""
        return not (self == other)

    def __lt__(self, other):
        """:returns if this particle object's ID is smaller than :arg other 's ID"""
        if type(self) is not type(other):
            err_msg = "This object and the other object (type={}) do note have the same type.".format(str(type(other)))
            raise AttributeError(err_msg)
        return self.id < other.id

    def __le__(self, other):
        """:returns if this particle object's ID is smaller-or-equal to :arg other 's ID"""
        if type(self) is not type(other):
            err_msg = "This object and the other object (type={}) do note have the same type.".format(str(type(other)))
            raise AttributeError(err_msg)
        return self.id <= other.id

    def __gt__(self, other):
        """:returns if this particle object's ID is larger than :arg other 's ID"""
        if type(self) is not type(other):
            err_msg = "This object and the other object (type={}) do note have the same type.".format(str(type(other)))
            raise AttributeError(err_msg)
        return self.id > other.id

    def __ge__(self, other):
        """:returns if this particle object's ID is larger-or-equal to :arg other 's ID"""
        if type(self) is not type(other):
            err_msg = "This object and the other object (type={}) do note have the same type.".format(str(type(other)))
            raise AttributeError(err_msg)
        return self.id >= other.id

    def __sizeof__(self):
        """:returns the byte size of the particle"""
        ptype = self.getPType()
        return sum([v.size for v in ptype.variables])


class ScipyInteractionParticle(ScipyParticle):
    vert_dist = Variable("vert_dist", dtype=np.float32)
    horiz_dist = Variable("horiz_dist", dtype=np.float32)


class JITParticle(ScipyParticle):
    """Particle class for JIT-based (Just-In-Time) Particle objects

    :param lon: Initial longitude of particle
    :param lat: Initial latitude of particle
    :param pid: externally-defined particle ID
    :param fieldset: :mod:`parcels.fieldset.FieldSet` object to track this particle on
    :param ngrids: number of grids of the related fieldset - optional alternative parameter to :param fieldset
    :param depth: Initial depth of particle
    :param time: Current time of the particle
    :param cptr: legacy parameter to ctypes-bound pointer into a C-array to store the particle information for ctypes

    Additional Variables can be added via the :Class Variable: objects

    Users should use JITParticles for faster advection computation.

    """

    def __init__(self, *args, **kwargs):
        """
        JITParticle - Constructor

        :param lon: Initial longitude of particle
        :param lat: Initial latitude of particle
        :param pid: externally-defined particle ID
        :param fieldset: :mod:`parcels.fieldset.FieldSet` object to track this particle on
        :param ngrids: number of grids of the related fieldset - optional alternative parameter to :param fieldset
        :param depth: Initial depth of particle
        :param time: Current time of the particle
        :param cptr: legacy parameter to ctypes-bound pointer into a C-array to store the particle information for ctypes
        """
        self._cptr = kwargs.pop('cptr', None)
        if self._cptr is None:
            # Allocate data for a single particle
            ptype = self.getPType()
            self._cptr = np.empty(1, dtype=ptype.dtype)
        super(JITParticle, self).__init__(*args, **kwargs)

    def __del__(self):
        """
        JITParticle - Destructor
        """
        super(JITParticle, self).__del__()

    def cdata(self):
        """
        :returns the ctypes pointer to the particle's data in the underlying C-array
        """
        if self._cptr is None:
            return None
        return self._cptr.ctypes.data_as(c_void_p)

    def set_cptr(self, value):
        """
        This function sets the ctypes pointer in which the particle's information are to be stored in a C-accessible way.
        """
        if isinstance(value, np.ndarray):
            ptype = self.getPType()
            self._cptr = np.array(value, dtype=ptype.dtype)
        else:
            self._cptr = None

    def get_cptr(self):
        """
        :returns the first position of the Numpy.ndarray (ctypes-bound) of the particle's data
        """
        if isinstance(self._cptr, np.ndarray):
            return self._cptr[0]
        return self._cptr

    def reset_cptr(self):
        """
        This function resets (i.e. nullifies) the stored ctypes pointer.
        """
        self._cptr = None

    def __eq__(self, other):
        """:returns if this particle object is equal to :arg other"""
        return super(JITParticle, self).__eq__(other)

    def __ne__(self, other):
        """:returns if this particle object is not equal to :arg other"""
        return not (self == other)

    def __lt__(self, other):
        """:returns if this particle object's ID is smaller than :arg other 's ID"""
        return super(JITParticle, self).__lt__(other)

    def __le__(self, other):
        """:returns if this particle object's ID is smaller-or-equal to :arg other 's ID"""
        return super(JITParticle, self).__le__(other)

    def __gt__(self, other):
        """:returns if this particle object's ID is larger than :arg other 's ID"""
        return super(JITParticle, self).__gt__(other)

    def __ge__(self, other):
        """:returns of this particle object's ID is larger-or-equal to :arg other 's ID"""
        return super(JITParticle, self).__ge__(other)

    def __sizeof__(self):
        """:returns the byte size of the particle"""
        ptype = self.getPType()
        return sum([v.size for v in ptype.variables])

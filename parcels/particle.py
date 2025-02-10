from ctypes import c_void_p
from operator import attrgetter
from typing import Literal

import numpy as np

from parcels.tools.statuscodes import StatusCode

__all__ = ["JITParticle", "ScipyInteractionParticle", "ScipyParticle", "Variable"]

indicators_64bit = [np.float64, np.uint64, np.int64, c_void_p]


class Variable:
    """Descriptor class that delegates data access to particle data.

    Parameters
    ----------
    name : str
        Variable name as used within kernels
    dtype :
        Data type (numpy.dtype) of the variable
    initial :
        Initial value of the variable. Note that this can also be a Field object,
        which will then be sampled at the location of the particle
    to_write : bool, 'once', optional
        Boolean or 'once'. Controls whether Variable is written to NetCDF file.
        If to_write = 'once', the variable will be written as a time-independent 1D array
    """

    def __init__(self, name, dtype=np.float32, initial=0, to_write: bool | Literal["once"] = True):
        self._name = name
        self.dtype = dtype
        self.initial = initial
        self.to_write = to_write

    @property
    def name(self):
        return self._name

    def __get__(self, instance, cls):
        if instance is None:
            return self
        if issubclass(cls, JITParticle):
            return instance._cptr.__getitem__(self.name)
        else:
            return getattr(instance, f"_{self.name}", self.initial)

    def __set__(self, instance, value):
        if isinstance(instance, JITParticle):
            instance._cptr.__setitem__(self.name, value)
        else:
            setattr(instance, f"_{self.name}", value)

    def __repr__(self):
        return f"Variable(name={self._name}, dtype={self.dtype}, initial={self.initial}, to_write={self.to_write})"

    def is64bit(self):
        """Check whether variable is 64-bit."""
        return True if self.dtype in indicators_64bit else False


class ParticleType:
    """Class encapsulating the type information for custom particles.

    Parameters
    ----------
    user_vars :
        Optional list of (name, dtype) tuples for custom variables
    """

    def __init__(self, pclass):
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
                            f"Custom Variable name '{v.name}' is not allowed, as it is also a built-in variable"
                        )
                    if v.name == "z":
                        raise AttributeError(
                            "Custom Variable name 'z' is not allowed, as it is used for depth in ParticleFile"
                        )
                self.variables = ptype.variables + self.variables
        # Sort variables with all the 64-bit first so that they are aligned for the JIT cptr
        self.variables = [v for v in self.variables if v.is64bit()] + [v for v in self.variables if not v.is64bit()]

    def __repr__(self):
        return f"{type(self).__name__}(pclass={self.name})"

    def __getitem__(self, item):
        for v in self.variables:
            if v.name == item:
                return v

    @property
    def _cache_key(self):
        return "-".join([f"{v.name}:{v.dtype}" for v in self.variables])

    @property
    def dtype(self):
        """Numpy.dtype object that defines the C struct."""
        type_list = [(v.name, v.dtype) for v in self.variables]
        for v in self.variables:
            if v.dtype not in self.supported_dtypes:
                raise RuntimeError(str(v.dtype) + " variables are not implemented in JIT mode")
        if self.size % 8 > 0:
            # Add padding to be 64-bit aligned
            type_list += [("pad", np.float32)]
        return np.dtype(type_list)

    @property
    def size(self):
        """Size of the underlying particle struct in bytes."""
        return sum([8 if v.is64bit() else 4 for v in self.variables])

    @property
    def supported_dtypes(self):
        """List of all supported numpy dtypes. All others are not supported."""
        # Developer note: other dtypes (mostly 2-byte ones) are not supported now
        # because implementing and aligning them in cgen.GenerableStruct is a
        # major headache. Perhaps in a later stage
        return [np.int32, np.uint32, np.int64, np.uint64, np.float32, np.double, np.float64, c_void_p]


class ScipyParticle:
    """Class encapsulating the basic attributes of a particle, to be executed in SciPy mode.

    Parameters
    ----------
    lon : float
        Initial longitude of particle
    lat : float
        Initial latitude of particle
    depth : float
        Initial depth of particle
    fieldset : parcels.fieldset.FieldSet
        mod:`parcels.fieldset.FieldSet` object to track this particle on
    time : float
        Current time of the particle


    Notes
    -----
    Additional Variables can be added via the :Class Variable: objects
    """

    lon = Variable("lon", dtype=np.float32)
    lon_nextloop = Variable("lon_nextloop", dtype=np.float32, to_write=False)
    lat = Variable("lat", dtype=np.float32)
    lat_nextloop = Variable("lat_nextloop", dtype=np.float32, to_write=False)
    depth = Variable("depth", dtype=np.float32)
    depth_nextloop = Variable("depth_nextloop", dtype=np.float32, to_write=False)
    time = Variable("time", dtype=np.float64)
    time_nextloop = Variable("time_nextloop", dtype=np.float64, to_write=False)
    id = Variable("id", dtype=np.int64, to_write="once")
    obs_written = Variable("obs_written", dtype=np.int32, initial=0, to_write=False)
    dt = Variable("dt", dtype=np.float64, to_write=False)
    state = Variable("state", dtype=np.int32, initial=StatusCode.Evaluate, to_write=False)

    lastID = 0  # class-level variable keeping track of last Particle ID used

    def __init__(self, lon, lat, pid, fieldset=None, ngrids=None, depth=0.0, time=0.0, cptr=None):
        # Enforce default values through Variable descriptor
        type(self).lon.initial = lon
        type(self).lon_nextloop.initial = lon
        type(self).lat.initial = lat
        type(self).lat_nextloop.initial = lat
        type(self).depth.initial = depth
        type(self).depth_nextloop.initial = depth
        type(self).time.initial = time
        type(self).time_nextloop.initial = time
        type(self).id.initial = pid
        type(self).lastID = max(type(self).lastID, pid)
        type(self).obs_written.initial = 0
        type(self).dt.initial = None

        ptype = self.getPType()
        # Explicit initialisation of all particle variables
        for v in ptype.variables:
            if isinstance(v.initial, attrgetter):
                initial = v.initial(self)
            else:
                initial = v.initial
            # Enforce type of initial value
            if v.dtype != c_void_p:
                setattr(self, v.name, v.dtype(initial))

    def __del__(self):
        pass  # superclass is 'object', and object itself has no destructor, hence 'pass'

    def __repr__(self):
        time_string = "not_yet_set" if self.time is None or np.isnan(self.time) else f"{self.time:f}"
        p_string = f"P[{self.id}](lon={self.lon:f}, lat={self.lat:f}, depth={self.depth:f}, "
        for var in vars(type(self)):
            if var in ["lon_nextloop", "lat_nextloop", "depth_nextloop", "time_nextloop"]:
                continue
            if type(getattr(type(self), var)) is Variable and getattr(type(self), var).to_write is True:
                p_string += f"{var}={getattr(self, var):f}, "
        return p_string + f"time={time_string})"

    @classmethod
    def add_variable(cls, var, *args, **kwargs):
        """Add a new variable to the Particle class

        Parameters
        ----------
        var : str, Variable or list of Variables
            Variable object to be added. Can be the name of the Variable,
            a Variable object, or a list of Variable objects
        """
        if isinstance(var, list):
            return cls.add_variables(var)
        if not isinstance(var, Variable):
            if len(args) > 0 and "dtype" not in kwargs:
                kwargs["dtype"] = args[0]
            if len(args) > 1 and "initial" not in kwargs:
                kwargs["initial"] = args[1]
            if len(args) > 2 and "to_write" not in kwargs:
                kwargs["to_write"] = args[2]
            dtype = kwargs.pop("dtype", np.float32)
            initial = kwargs.pop("initial", 0)
            to_write = kwargs.pop("to_write", True)
            var = Variable(var, dtype=dtype, initial=initial, to_write=to_write)

        class NewParticle(cls):
            pass

        setattr(NewParticle, var.name, var)
        return NewParticle

    @classmethod
    def add_variables(cls, variables):
        """Add multiple new variables to the Particle class

        Parameters
        ----------
        variables : list of Variable
            Variable objects to be added. Has to be a list of Variable objects
        """
        NewParticle = cls
        for var in variables:
            NewParticle = NewParticle.add_variable(var)
        return NewParticle

    @classmethod
    def getPType(cls):
        return ParticleType(cls)

    @classmethod
    def set_lonlatdepth_dtype(cls, dtype):
        cls.lon.dtype = dtype
        cls.lat.dtype = dtype
        cls.depth.dtype = dtype
        cls.lon_nextloop.dtype = dtype
        cls.lat_nextloop.dtype = dtype
        cls.depth_nextloop.dtype = dtype

    @classmethod
    def setLastID(cls, offset):
        ScipyParticle.lastID = offset


ScipyInteractionParticle = ScipyParticle.add_variables(
    [Variable("vert_dist", dtype=np.float32), Variable("horiz_dist", dtype=np.float32)]
)


class JITParticle(ScipyParticle):
    """Particle class for JIT-based (Just-In-Time) Particle objects.

    Parameters
    ----------
    lon : float
        Initial longitude of particle
    lat : float
        Initial latitude of particle
    fieldset : parcels.fieldset.FieldSet
        mod:`parcels.fieldset.FieldSet` object to track this particle on
    dt :
        Execution timestep for this particle
    time :
        Current time of the particle


    Notes
    -----
    Additional Variables can be added via the :Class Variable: objects

    Users should use JITParticles for faster advection computation.
    """

    def __init__(self, *args, **kwargs):
        self._cptr = kwargs.pop("cptr", None)
        if self._cptr is None:
            # Allocate data for a single particle
            ptype = self.getPType()
            self._cptr = np.empty(1, dtype=ptype.dtype)[0]
        super().__init__(*args, **kwargs)

    def __del__(self):
        super().__del__()

from typing import Literal

import numpy as np
import xarray as xr

from parcels.tools.statuscodes import StatusCode

__all__ = ["InteractionParticle", "Particle", "Variable"]


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
        return getattr(instance, f"_{self.name}", self.initial)

    def __set__(self, instance, value):
        setattr(instance, f"_{self.name}", value)

    def __repr__(self):
        return f"Variable(name={self._name}, dtype={self.dtype}, initial={self.initial}, to_write={self.to_write})"


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
        if not issubclass(pclass, Particle):
            raise TypeError("Class object does not inherit from parcels.Particle")
        self.name = pclass.__name__
        # Pick Variable objects out of __dict__.
        self.variables = [v for v in pclass.__dict__.values() if isinstance(v, Variable)]
        for cls in pclass.__bases__:
            if issubclass(cls, Particle):
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

    def __repr__(self):
        return f"{type(self).__name__}(pclass={self.name})"

    def __getitem__(self, item):
        for v in self.variables:
            if v.name == item:
                return v


class Particle:
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

    def __init__(self, data: xr.Dataset, index: int):
        self._data = data
        self._index = index

    def __getattr__(self, name):
        return self._data[name].values[self._index]

    def __setattr__(self, name, value):
        if name in ["_data", "_index"]:
            object.__setattr__(self, name, value)
        else:
            self._data[name][self._index] = value

    def delete(self):
        """Signal the particle for deletion."""
        self.state = StatusCode.Delete

    @classmethod
    def add_variable(cls, variable: Variable | list[Variable]):
        """Add a new variable to the Particle class

        Parameters
        ----------
        variable : Variable or list[Variable]
            Variable or list of Variables to be added to the Particle class.
            If a list is provided, all variables will be added to the class.
        """

        class NewParticle(cls):
            pass

        if isinstance(variable, Variable):
            setattr(NewParticle, variable.name, variable)
        elif isinstance(variable, list):
            for var in variable:
                if not isinstance(var, Variable):
                    raise TypeError(f"Expected Variable, got {type(var)}")
                setattr(NewParticle, var.name, var)
        return NewParticle

    @classmethod
    def getPType(cls):
        return ParticleType(cls)


InteractionParticle = Particle.add_variable(
    [Variable("vert_dist", dtype=np.float32), Variable("horiz_dist", dtype=np.float32)]
)


class JITParticle(Particle):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "JITParticle has been deprecated in Parcels v4. Use Particle instead."
        )  # TODO v4: link to migration guide

import operator
from keyword import iskeyword
from typing import Literal

import numpy as np

from parcels._compat import _attrgetter_helper
from parcels._reprs import _format_list_items_multiline
from parcels.tools.statuscodes import StatusCode

__all__ = ["KernelParticle", "Particle", "ParticleClass", "Variable"]
_TO_WRITE_OPTIONS = [True, False, "once"]


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
    attrs : dict, optional
        Attributes to be stored with the variable when written to file. This can include metadata such as units, long_name, etc.
    """

    def __init__(
        self, name, dtype=np.float32, initial=0, to_write: bool | Literal["once"] = True, attrs: dict | None = None
    ):
        if not isinstance(name, str):
            raise TypeError(f"Variable name must be a string. Got {name=!r}")
        _assert_valid_python_varname(name)

        try:
            dtype = np.dtype(dtype)
        except (TypeError, ValueError):
            raise TypeError(f"Variable dtype must be a valid numpy dtype. Got {dtype=!r}")

        if to_write not in _TO_WRITE_OPTIONS:
            raise ValueError(f"to_write must be one of {_TO_WRITE_OPTIONS!r}. Got {to_write=!r}")

        if attrs is None:
            attrs = {}

        if not to_write:
            if attrs != {}:
                raise ValueError(f"Attributes cannot be set if {to_write=!r}.")

        self._name = name
        self.dtype = dtype
        self.initial = initial
        self.to_write = to_write
        self.attrs = attrs

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return f"Variable(name={self._name!r}, dtype={self.dtype!r}, initial={self.initial!r}, to_write={self.to_write!r}, attrs={self.attrs!r})"

    # def __get__(self, instance, cls):
    #     if instance is None:
    #         return self
    #     return getattr(instance, f"_{self.name}", self.initial)

    # def __set__(self, instance, value):
    #     setattr(instance, f"_{self.name}", value)


class ParticleClass:
    """Define a class of particles. This is used to generate the particle data which is then used in the simulation.

    Parameters
    ----------
    variables : list[Variable]
        List of Variable objects that define the particle's attributes.

    """

    def __init__(self, variables: list[Variable]):
        if not isinstance(variables, list):
            raise TypeError(f"Expected list of Variable objects, got {type(variables)}")
        if not all(isinstance(var, Variable) for var in variables):
            raise ValueError(f"All items in variables must be instances of Variable. Got {variables=!r}")

        self.variables = variables

    def __repr__(self):
        vars = [repr(v) for v in self.variables]
        return f"ParticleClass(variables={_format_list_items_multiline(vars)})"

    def add_variable(self, variable: Variable | list[Variable]):
        """Add a new variable to the Particle class. This returns a new Particle class with the added variable(s).

        Parameters
        ----------
        variable : Variable or list[Variable]
            Variable or list of Variables to be added to the Particle class.
            If a list is provided, all variables will be added to the class.
        """
        if isinstance(variable, Variable):
            variable = [variable]

        for var in variable:
            if not isinstance(var, Variable):
                raise TypeError(f"Expected Variable, got {type(var)}")

        _assert_no_duplicate_variable_names(existing_vars=self.variables, new_vars=variable)

        return ParticleClass(variables=self.variables + variable)


class KernelParticle:
    """Simple class to be used in a kernel that links a particle (on the kernel level) to a particle dataset."""

    def __init__(self, data, index):
        self._data = data
        self._index = index

    def __getattr__(self, name):
        return self._data[name][self._index]

    def __setattr__(self, name, value):
        if name in ["_data", "_index"]:
            object.__setattr__(self, name, value)
        else:
            self._data[name][self._index] = value


def _assert_no_duplicate_variable_names(*, existing_vars: list[Variable], new_vars: list[Variable]):
    existing_names = {var.name for var in existing_vars}
    for var in new_vars:
        if var.name in existing_names:
            raise ValueError(f"Variable name already exists: {var.name}")


def _assert_valid_python_varname(name):
    if name.isidentifier() and not iskeyword(name):
        return
    raise ValueError(f"Particle variable has to be a valid Python variable name. Got {name=!r}")


Particle = ParticleClass(
    variables=[
        Variable("lon", dtype=np.float32),
        Variable("lon_nextloop", dtype=np.float32, to_write=False),
        Variable("lat", dtype=np.float32),
        Variable("lat_nextloop", dtype=np.float32, to_write=False),
        Variable("depth", dtype=np.float32),
        Variable("depth_nextloop", dtype=np.float32, to_write=False),
        Variable("time", dtype=np.float64),
        Variable("time_nextloop", dtype=np.float64, to_write=False),
        Variable("id", dtype=np.int64, to_write="once"),
        Variable("obs_written", dtype=np.int32, initial=0, to_write=False),
        Variable("dt", dtype="timedelta64[ns]", to_write=False),
        Variable("state", dtype=np.int32, initial=StatusCode.Evaluate, to_write=False),
    ]
)


def create_particle_data(*, pclass: ParticleClass, nparticles, ngrids, **initial: dict[str, np.array]):
    variables = {var.name: var for var in pclass.variables}

    assert "ei" not in initial, "'ei' is for internal use, and is unique since is only non 1D array"

    for var_name in initial:
        if var_name not in variables:
            raise ValueError(f"Variable {var_name} is not defined in the ParticleClass.")

        values = initial[var_name]
        if values.shape != (nparticles,):
            raise ValueError(f"Initial value for {var_name} must have shape ({nparticles},). Got {values.shape=}")

        initial[var_name] = values.astype(variables[var_name].dtype)

    data = {"ei": np.zeros((nparticles, ngrids), dtype=np.int32), **initial}

    vars_to_create = {k: v for k, v in variables.items() if k not in data}

    for var in vars_to_create.values():
        if isinstance(var.initial, operator.attrgetter):
            name_to_copy = var.initial(_attrgetter_helper)
            data[var.name] = data[name_to_copy].copy()
        else:
            data[var.name] = _create_array_for_variable(variables[var], nparticles)
    return data


def _create_array_for_variable(variable: Variable, nparticles: int):
    assert not isinstance(variable.initial, operator.attrgetter), (
        "This function cannot handle attrgetter initial values."
    )
    return np.full(
        shape=(nparticles,),
        fill_value=variable.initial,
        dtype=variable.dtype,
    )

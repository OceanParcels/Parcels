import numpy as np
from abc import ABC
from abc import abstractmethod
import warnings

from parcels.tools.statuscodes import OperationCode
from .collections import ParticleCollection


class NDCluster(ABC):
    """Interface."""


class BaseParticleCollectionIterator(ABC):
    """Interface for the ParticleCollection iterator. Provides the
    ability to iterate over the particles in the ParticleCollection.
    """
    def ___init___(self):
        self._head = None
        self._tail = None
        self._current = None

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        """Returns the next value from ParticleSet object's lists."""
        pass

    @property
    def head(self):
        return self._head

    @property
    def tail(self):
        return self._tail

    @property
    def current(self):
        return self._current

    @abstractmethod
    def __repr__(self):
        """Represents the current position in the iteration.
        """
        pass


class BaseParticleAccessor(ABC):
    """Interface for the ParticleAccessor. Implements a wrapper around
    particles to provide easy access."""
    def __init__(self, pcoll):
        """Initialize the ParticleAccessor object with at least a
        reference to the ParticleSet it encapsulates.
        """
        self.pcoll = pcoll

    def set_index(self, index):
        # Convert into a "proper" property?
        self._index = index

    def update_next_dt(self, next_dt=None):
        if next_dt is None:
            if not np.isnan(self._next_dt):
                self.dt, self._next_dt = self._next_dt, np.nan
        else:
            self._next_dt = next_dt

    def delete(self):
        self.state = OperationCode.Delete

    def set_state(self, state):
        # Convert into a "proper" property?
        # Why is this even separate? It sets the state of the particle,
        # so should be handled by the __setattr__ function, right?
        # Seems to be coppied directly from ScipyParticle.
        self.state = state

    @abstractmethod
    def __getattr__(self, name):
        """The ParticleAccessor should provide an implementation of this
        built-in function to allow accessing particle attributes in its
        corresponding ParticleSet datastructure.
        """
        pass

    @abstractmethod
    def __setattr__(self, name, value):
        """The ParticleAccessor should provide an implementation of this
        built-in function to allow setting particle attributes in its
        corresponding ParticleSet datastructure.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """The ParticleAccessor should provide an implementation of this
        built-in function that returns a string representation of the
        Particle that it currently provides access to.
        """
        pass


class BaseParticleSet(ParticleCollection, NDCluster):
    """Base ParticleSet."""
    def data_accessor(self):
        """Returns an Accessor for the particles in this ParticleSet.
        Deprecated - the accessor must not be manipulated directly.
        """
        warnings.warn(
            "This method of accessing particle data is deprecated",
            DeprecationWarning,
            stacklevel=2
        )

    @abstractmethod
    def execute(self, pyfunc, endtime, runtime, dt, moviedt, recovery,
                output_file, movie_background_field, verbose_progress,
                postIterationCallbacks, callbackdt):
        """Execute a given kernel function over the particle set for
        multiple timesteps. Optionally also provide sub-timestepping
        for particle output.

        :param pyfunc: Kernel function to execute. This can be the name of a
                       defined Python function or a :class:`parcels.kernel.Kernel` object.
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
        pass

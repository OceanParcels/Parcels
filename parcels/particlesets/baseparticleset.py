import numpy as np
from abc import ABC
from abc import abstractmethod

from parcels.tools.statuscodes import OperationCode


class ParticleCollection(ABC):
    """Interface."""


class NDCluster(ABC):
    """Interface."""


class BaseParticleSetIterator(ABC):
    """Interface for the ParticleSet iterator. Provides the ability to
    iterate over the particles in the ParticleSet."""
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
    def __init__(self, pset):
        """Initialize the ParticleAccessor object with at least a
        reference to the ParticleSet it encapsulates.
        """
        self.pset = pset

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

class BaseParticleSet(ParticleCollection, NDCluster):
    """Base ParticleSet."""
    @abstractmethod
    def __iter__(self):
        """Returns an Iterator for the ParticleSet."""
        pass

    @abstractmethod
    def data_accessor(self):
        """Returns an Accessor for the particles in this ParticleSet."""
        pass

    @property
    @abstractmethod
    def size(self):
        pass

    def __len__(self):
        return self.size

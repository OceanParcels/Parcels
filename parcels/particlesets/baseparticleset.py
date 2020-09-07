from abc import ABC
from abc import abstractmethod


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


class BaseParticleAccessor(ABC):
    """Interface for the ParticleAccessor. Implements a wrapper around
    particles to provide easy access."""


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

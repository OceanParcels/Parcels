from abc import ABC
from abc import abstractmethod  # noqa


class ParticleCollection(ABC):
    """Interface."""


class NDCluster(ABC):
    """Interface."""


class ParticleSetIterator:
    """Interface?"""
    def __init__(self, pset):
        self.p = pset.data_accessor()
        self.max_len = pset.size
        self._index = 0

    def __next__(self):
        ''''Returns the next value from ParticleSet object's lists '''
        if self._index < self.max_len:
            self.p.set_index(self._index)
            result = self.p
            self._index += 1
            return result
        # End of Iteration
        raise StopIteration


class BaseParticleAccessor(ABC):
    """Interface."""


class BaseParticleSet(ParticleCollection, NDCluster):
    """Base ParticleSet."""
    def __iter__(self):
        return ParticleSetIterator(self)

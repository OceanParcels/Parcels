from abc import ABC
from abc import abstractmethod  # noqa


class ParticleCollection(ABC):
    """Interface."""


class NDCluster(ABC):
    """Interface."""


class BaseParticleAccessor(ABC):
    """Interface."""
    @abstractmethod
    def __next__(self):
        """Returns the next value from ParticleSet object's lists"""
        pass


class BaseParticleSet(ParticleCollection, NDCluster):
    """Base ParticleSet."""
    @abstractmethod
    def __iter__(self):
        pass

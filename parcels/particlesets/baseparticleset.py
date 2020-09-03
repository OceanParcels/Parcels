from abc import ABC
from abc import abstractmethod  # noqa


class ParticleCollection(ABC):
    """Interface."""


class NDCluster(ABC):
    """Interface."""


class BaseParticleAccessor(ABC):
    """Interface."""


class BaseParticleSet(ParticleCollection, NDCluster):
    """Base ParticleSet."""
    def __iter__(self):
        pass

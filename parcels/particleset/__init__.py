from .baseparticleset import BaseParticleSet  # noqa
from .particlesetaos import ParticleSetAOS  # noqa
from .particlesetsoa import ParticleSetSOA  # noqa

# ParticleSet is an alias for ParticleSetSOA, i.e. the default
# implementation for storing particles is the Structure of Arrays
# approach.
ParticleSet = ParticleSetSOA

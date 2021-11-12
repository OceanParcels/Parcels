from .baseparticleset import BaseParticleSet  # noqa
from .benchmarkparticleset import BaseBenchmarkParticleSet  # noqa
from .particlesetaos import ParticleSetAOS, BenchmarkParticleSetAOS  # noqa
from .particlesetsoa import ParticleSetSOA, BenchmarkParticleSetSOA  # noqa
from .particlesetnodes import ParticleSetNodes, BenchmarkParticleSetNodes  # noqa

# ParticleSet is an alias for ParticleSetSOA, i.e. the default
# implementation for storing particles is the Structure of Arrays
# approach.
ParticleSet = ParticleSetSOA

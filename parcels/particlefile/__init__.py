"""Module controlling the writing of ParticleSets to NetCDF file"""

from .baseparticlefile import BaseParticleFile  # noqa: F401
from .baseparticlefile import _set_calendar  # noqa: F401
from .particlefileaos import ParticleFileAOS  # noqa: F401
from .particlefilesoa import ParticleFileSOA  # noqa: F401

ParticleFile = ParticleFileSOA

"""Module controlling the writing of ParticleSets to NetCDF file"""

from parcels.baseparticlefile import BaseParticleFile  # noqa: F401
from parcels.baseparticlefile import _set_calendar  # noqa: F401
from parcels.particlefilesoa import ParticleFileSOA

ParticleFile = ParticleFileSOA

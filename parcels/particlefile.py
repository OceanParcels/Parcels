"""Module controlling the writing of ParticleSets to NetCDF file"""
from parcels.baseparticlefile import BaseParticleFile, _set_calendar  # NOGA
from parcels.particlefilesoa import ParticleFileSOA

ParticleFile = ParticleFileSOA

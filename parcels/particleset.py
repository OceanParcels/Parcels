from parcels.particlesets.particlesetsoa import ParticleSetSOA

__all__ = ['ParticleSet']

# ParticleSet is an alias for ParticleSetSOA, i.e. the default
# implementation for storing particles is the Structure of Arrays
# approach.
ParticleSet = ParticleSetSOA

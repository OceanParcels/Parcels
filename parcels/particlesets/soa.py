import numpy as np

from .collections import ParticleCollection, Collection
from .baseparticleset import BaseParticleAccessor
from parcels.particle import ScipyParticle

"""
Author: Dr. Christian Kehl
github relation: #913 (particleset_class_hierarchy)
purpose: defines all the specific functions for a ParticleCollection, ParticleAccessor, ParticleSet etc. that relates
         to a structure-of-array (SoA) data arrangement.
"""


class ParticleCollection_SoA(ParticleCollection):

    def __init__(self):
        super(ParticleCollection, self).__init__()

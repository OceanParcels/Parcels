try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
import numpy as np

numpy_includes = [np.get_include()]
particle_sources = ['parcels/particle.pyx']

setup(name='parcels',
      version = '0.0.1',
      description = """Framework for Lagrangian tracking of virtual
      ocean particles in the petascale age.""",
      author = "Imperial College London",
      packages = ['parcels'],
      ext_modules=[Extension('parcels.particle', particle_sources,
                             include_dirs=numpy_includes),]
)

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
import numpy as np

setup(name='parcels',
      version = '0.0.1',
      description = """Framework for Lagrangian tracking of virtual
      ocean particles in the petascale age.""",
      author = "Imperial College London",
      packages = ['parcels'],
)

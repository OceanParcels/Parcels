try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup
import numpy as np

setup(name='parcels',
      version='0.0.1',
      description="""Framework for Lagrangian tracking of virtual
      ocean particles in the petascale age.""",
      author="Imperial College London",
      packages=find_packages(exclude=['docs', 'examples', 'indlude', 'scripts', 'tests']),
)

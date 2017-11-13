try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

setup(name='parcels',
      version='0.0.1',
      description="""Framework for Lagrangian tracking of virtual
      ocean particles in the petascale age.""",
      author="Imperial College London",
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      packages=find_packages(exclude=['docs', 'examples', 'scripts', 'tests']) + ['include'],
      include_package_data=True,
      )

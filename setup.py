"""Install Parcels and dependencies."""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='parcels',
      version='0.0.1',
      description="""Framework for Lagrangian tracking of virtual
      ocean particles in the petascale age.""",
      author="Imperial College London",
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      packages=['parcels'],
      package_data={'parcels': ['parcels/include/*',
                                'parcels/examples/*']},
      include_package_data=True,
      entry_points={'console_scripts': [
          'parcels_get_examples = parcels.scripts.get_examples:main']}
      )

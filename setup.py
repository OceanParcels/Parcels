"""Install Parcels and dependencies."""

try:
    from setuptools import find_packages, setup
except ImportError:
    from distutils.core import find_packages, setup

with open("README.md") as fh:
    long_description = fh.read()

setup(name='parcels',
      description="""Framework for Lagrangian tracking of virtual ocean particles in the petascale age.""",
      author="oceanparcels.org team",
      use_scm_version={'write_to': 'parcels/_version_setup.py', 'local_scheme': 'no-local-version'},
      long_description=long_description,
      long_description_content_type="text/markdown",
      setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
      python_requires='>=3.8',
      packages=find_packages(),
      package_data={'parcels': ['include/*']},
      entry_points={'console_scripts': [
          'parcels_convert_npydir_to_netcdf = parcels.scripts.convert_npydir_to_netcdf:main']}
      )

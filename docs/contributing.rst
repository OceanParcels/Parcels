Contributing to parcels
=========================

We welcome all contributions to the OceanParcels project! Whether you're a seasoned programmer or just getting started, there are many ways you can help us make our project better.
Even if you don't want to contribute directly to the codebase, getting involved in discussions and issues are great ways to help out with the project.

This document outlines some guidelines for contributing to the parcels codebase. If you have any questions or suggestions for how we can improve this document, please let us know.

Getting Started
---------------
To get started contributing to parcels, you'll need to do a few things:

1. Fork our repository on GitHub.

2. Clone your fork of the repository to your local machine, and ``cd`` into your project folder.

3. Install `Anaconda <https://www.anaconda.com/products/distribution>`__ (if you haven't already).

4. Install the development Anaconda environment by installing from the environment file corresponding to your OS (``environment_py3_linux.yml``, ``environment_py3_osx.yml`` or ``environment_py3_win.yml``) then activate it. E.g. on Linux: ``conda env create -f environment_py3_linux.yml`` then ``conda activate py3_parcels``.

5. Install your local copy of parcels in editable mode by running ``pip install -e .``.

6. **Optional:** Setup ``pre-commit`` hooks by running ``pre-commit install``. This ensures that code is formatted correctly before you commit it.


Making Changes
--------------

Once you have a working environment, you can start making changes to the code! Here are some general guidelines to follow when making changes:

* Follow the `NumPy docstring conventions <https://numpydoc.readthedocs.io/en/latest/format.html>`__ when adding or modifying docstrings.

* Follow the `PEP 8 <https://peps.python.org/pep-0008/>`__ style guide when writing code. This codebase also uses `flake8 <https://flake8.pycqa.org/en/latest/>`__ and `isort <https://pycqa.github.io/isort/>`__ to ensure a consistent code style (these tools are run automatically by pre-commit).

* Use Git to manage your changes. Create a new branch for each feature or bug fix.

* Write clear commit messages that explain the changes you've made.

* Include tests for any new code you write.

* Submit your changes as a pull request to the ``master`` branch, and wait for feedback!


.. note::
   Feel free to create an issue for your proposed feature or change to the codebase so that feedback can be given.
   Submitting a "draft pull request" is also a great way to give visibility to your branch during development, allowing for faster feedback.


Working with documentation
--------------------------
The documentation for this project is processed by Sphinx. To view documentation from your changes, you can run
``sphinx-autobuild docs docs/_build`` to create a server to automatically rebuild the documentation when you make changes.


If you have any questions or issues while contributing to parcels, please feel free to `open a discussion <https://github.com/OceanParcels/parcels/discussions>`__. Thank you for your contributions!

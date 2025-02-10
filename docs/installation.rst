Basic installation
==================

The simplest way to install the Parcels code is to use Anaconda and the `Parcels conda-forge package <https://anaconda.org/conda-forge/parcels>`_ with the latest release of Parcels. This package will automatically install all the requirements for a fully functional installation of Parcels. This is the “batteries-included” solution probably suitable for most users. Note that we support Python 3.10 and higher.

If you want to install the latest development version of Parcels and work with features that have not yet been officially released, you can follow the instructions for a `developer installation <#installation-for-developers>`_.

The steps below are the installation instructions for Linux, macOS and Windows.

.. _step 1 above:

**Step 1:** Install Anaconda's Miniconda following the steps at https://docs.anaconda.com/miniconda/. If you're on Linux /macOS, the following assumes that you installed Miniconda to your home directory.

**Step 2:** Start a terminal (Linux / macOS) or the Anaconda prompt (Windows). Activate the ``base`` environment of your Miniconda and create an environment containing Parcels, all its essential dependencies, ``trajan`` (a trajectory plotting dependency used in the notebooks) and the nice-to-have cartopy and jupyter packages:

.. code-block:: bash

    conda activate base
    conda create -n parcels -c conda-forge parcels trajan cartopy jupyter

.. note::

    For some of the examples, ``pytest`` also needs to be installed. This can be quickly done with ``conda install -n parcels pytest`` which installs ``pytest`` directly into the newly created ``parcels`` environment.

**Step 3:** Activate the newly created Parcels environment:

.. code-block:: bash

    conda activate parcels

**Step 4:** Download `a zipped copy <https://docs.oceanparcels.org/en/latest/_downloads/307c382eb1813dc691e8a80d6c0098f7/parcels_tutorials.zip>`_ of the Parcels tutorials and examples and unzip it.

**Step 5:** Go to the unzipped folder and run one of the examples to validate that you have a working Parcels setup:

.. code-block:: bash

  python example_peninsula.py --fieldset 100 100

.. note::
  If you are on macOS and get a compilation error, you may need to accept the Apple xcode license ``xcode-select --install``. If this does not solve the compilation error, you may want to try running ``export CC=gcc``. If the compilation error remains, you may want to check `this solution <https://stackoverflow.com/a/58323411/5172570>`_.

*Optionally:* if you want to run all the examples and tutorials, start Jupyter and open the tutorial notebooks:

.. code-block:: bash

  jupyter notebook


.. note::

  The next time you start a terminal and want to work with Parcels, activate the environment with:

  .. code-block:: bash

    conda activate parcels



Installation for developers
===========================

If you would prefer to have a development installation of Parcels (i.e., where the code can be actively edited), you can do so by setting up Miniconda (as detailed in step 1 above), cloning the Parcels repo, installing dependencies using the environment file, and then installing Parcels in an editable mode such that changes to the cloned code can be tested during development.

**Step 1:** Same as `step 1 above`_.

**Step 2:** Clone the Parcels repo and create a new environment with the development dependencies:

.. code-block:: bash

  git clone https://github.com/OceanParcels/parcels.git
  cd parcels
  conda env create -n parcels-dev -f environment.yml

**Step 3:** Activate the environment and install Parcels in editable mode:

.. code-block:: bash

  conda activate parcels-dev
  pip install --no-build-isolation --no-deps -e .

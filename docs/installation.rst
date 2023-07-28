Basic installation
==================

The simplest way to install the Parcels code is to use Anaconda and the `Parcels conda-forge package <https://anaconda.org/conda-forge/parcels>`_ with the latest release of Parcels. This package will automatically install all the requirements for a fully functional installation of Parcels. This is the “batteries-included” solution probably suitable for most users. If you want to install the latest development version of Parcels and work with features that have not yet been officially released, you can follow the instructions for a `developer installation <#installation-for-developers>`_.

The steps below are the installation instructions for Linux / macOS and for Windows. If the commands for Linux / macOS and Windows differ, this is indicated with a comment at the end of the line.

#. Install Anaconda's Miniconda following the steps at https://conda.io/docs/user-guide/install/, making sure to select the Python-3 version. If you're on Linux /macOS, it also assumes that you installed Miniconda-3 to your home directory.

#. Start a terminal (Linux / macOS) or the Anaconda prompt (Windows). Activate the ``base`` environment of your Miniconda and create an environment containing Parcels, all its essential dependencies, and the nice-to-have cartopy and jupyter packages:

    .. code-block:: bash

      conda activate base
      conda create -n parcels -c conda-forge parcels cartopy

    .. note::

        For some of the examples, ``pytest`` also needs to be installed. This can be quickly done with ``conda install -n parcels pytest`` which installs ``pytest`` directly into the newly created ``parcels`` environment.

#. Activate the newly created Parcels environment:

    .. code-block:: bash

      conda activate parcels  # Linux / macOS
      activate parcels        # Windows</code></pre>

#. Get a zipped copy of the Parcels tutorials and examples from `here <https://docs.oceanparcels.org/en/latest/_downloads/307c382eb1813dc691e8a80d6c0098f7/parcels_tutorials.zip>`_ and unzip it.

#. Go to the unzipped folder and run one of the examples to validate that you have a working Parcels setup:

    .. code-block:: bash

      python example_peninsula.py --fieldset 100 100

    .. note::
      If you are on macOS and get a compilation error, you may need to accept the Apple xcode license ``xcode-select --install``. If this does not solve the compilation error, you may want to try running ``export CC=gcc``. If the compilation error remains, you may want to check `this solution <https://stackoverflow.com/a/58323411/5172570>`_.

#. Optionally, if you want to run all the examples and tutorials, start Jupyter and open the tutorial notebooks:

    .. code-block:: bash

      jupyter notebook

#. The next time you start a terminal and want to work with Parcels, activate the environment with:

    .. code-block:: bash

      conda activate parcels  # Linux / macOS
      activate parcels        # Windows


Installation for developers
===========================

Parcels depends on a working Python installation, a netCDF installation, a C compiler, and various Python packages. If you prefer to maintain your own Python installation providing all this, download one of the environment files for `linux <https://raw.githubusercontent.com/OceanParcels/parcels/master/environment_py3_linux.yml>`_, for `macOS <https://raw.githubusercontent.com/OceanParcels/parcels/master/environment_py3_osx.yml>`_ or for `Windows <https://raw.githubusercontent.com/OceanParcels/parcels/master/environment_py3_win.yml>`_ and install it with:

    .. code-block:: bash

      git clone https://github.com/OceanParcels/parcels.git
      conda env create -f environment_py3_<OS>.yml  # where <OS> is either linux, osx or win

Then, add the directory where you stored the Parcels code to your ``$PYTHONPATH`` environment variable. For example, if you cloned the Parcels code to ``/home/username/parcels``, add the following line to your ``.bashrc`` or ``.zshrc`` file:

    .. code-block:: bash

      export PYTHONPATH=/home/username/parcels:$PYTHONPATH

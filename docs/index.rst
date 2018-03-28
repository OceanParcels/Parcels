.. parcels documentation master file, created by
   sphinx-quickstart on Tue Oct 20 09:58:20 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Parcels
===================================

**Parcels** (**P**\ robably **A** **R**\ eally **C**\ omputationally **E**\ fficient **L**\ agrangian **S**\ imulator) is an experimental prototype code aimed at exploring novel approaches for Lagrangian tracking of virtual ocean particles in the petascale age.

Its code is licensed under an `open source MIT license <https://github.com/OceanParcels/parcels/blob/master/LICENSE.md>`_ and can be downloaded from https://github.com/OceanParcels/parcels.

.. figure:: http://oceanparcels.org/animated-gifs/globcurrent_fullyseeded.gif
   :align: center

   *Animation of virtual particles carried by ocean surface flow in the* `Agulhas Current <https://en.wikipedia.org/wiki/Agulhas_Current>`_ *off South Africa. The particles are advected with* `Parcels <http://oceanparcels.org/>`_ *in data from the* `GlobCurrent Project <http://globcurrent.ifremer.fr/products-data/products-overview>`_. *See* `this tutorial <http://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_Agulhasparticles.ipynb>`_ *for the Parcels code behind this animated gif.*

Parcels manuscript and code
===================================
The manuscript detailing this first release of Parcels, version 0.9, has been published in `Geoscientific Model Development <https://www.geosci-model-dev.net/10/4175/2017/gmd-10-4175-2017.html>`_ and can be cited as

*Lange, M. and E van Sebille (2017) Parcels v0.9: prototyping a Lagrangian Ocean Analysis framework for the petascale age. Geoscientific Model Development, 10, 4175-4186. doi:10.5194/gmd-2017-167*

The latest version of the code is available at https://github.com/OceanParcels/parcels.

Parcels development status
===================================

The current release of Parcels, version 1.0, is a fully-functional, feature-complete code for offline Lagrangian ocean analysis. See below for a list of features, or keep an eye
on the `Github Development Timeline page
<https://github.com/OceanParcels/parcels/projects/1>`_

**Major features**

* Advection of particles in 2D using inbuilt kernels for Runge-Kutta4, Runge-Kutta45 and Euler Forward and in 3D using the inbuilt kernel for Runge-Kutta4_3D (see :mod:`parcels.kernels.advection`)
* Simple horizontal diffusion of particles using inbuilt Brownian Motion kernel (see :mod:`parcels.kernels.diffusion`)
* Ability to define and execute custom kernels (see `the Adding-a-custom-behaviour-kernel part of the Tutorial <http://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/parcels_tutorial.ipynb#Adding-a-custom-behaviour-kernel>`_)
* Ability to add custom Variables to Particles (see `the Sampling-a-Field-with-Particles part of the Tutorial <http://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/parcels_tutorial.ipynb#Sampling-a-Field-with-Particles>`_)
* Ability to add and remove Particles (see :func:`parcels.particleset.ParticleSet.add` and :func:`parcels.particleset.ParticleSet.remove`)
* Ability to run in both Scipy and JIT (Just-In-Time compilation) mode. The former is easier to debug, but the latter can be a factor 1,000 faster (see the `JIT-vs-Scipy tutorial <http://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_jit_vs_scipy.ipynb>`_)
* Ability to read in hydrodynamic field in NetCDF format from a suite of models (see `the Reading-in-data-from-arbritrary-NetCDF-files part of the Tutorial <http://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/parcels_tutorial.ipynb#Reading-in-data-from-arbritrary-NetCDF-files>`_). Parcels currently supports Rectilinear and Curvilinear grids in the horizontal (see also the `NEMO curvilinear grids tutorial <https://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_nemo_curvilinear.ipynb>`_).
* Output particles in NetCDF format (see :mod:`parcels.particlefile`)
* Basic plotting of particles, both on the fly and from netcdf output files (see the `plotting tutorial <http://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_plotting.ipynb>`_)

**Experimental features**

*The features below are implemented, but not yet thoroughly tested. Please use with caution*

* Ability to run particles in any vertical coordinate system: z-level, sigma-level (terrain-following), or rho-level (density-following).

**Future development goals**

* More types of diffusion of particles using suite of inbuilt kernels
* Support for unstructured grids
* Implementation of parallel execution using tiling of the domain
* Faster and more efficient code
* Advanced control of particles near land boundaries


Parcels Tutorials
===================================

The best way to get started with Parcels is to have a look at the Jupyter notebooks below:

* `Parcels tutorial <http://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/parcels_tutorial.ipynb>`_ for a general introduction and overview into the main features of Parcels

* `Periodic boundaries tutorial <https://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_periodic_boundaries.ipynb>`_ for a tutorial on how to implement periodic boundary conditions

* `NEMO curvilinear grids tutorial <https://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_nemo_curvilinear.ipynb>`_ for a tutorial on how to run Parcels on curvilinear grids such as those of the NEMO models

* `Delayed start of particles tutorial <http://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_delaystart.ipynb>`_ for a tutorial on how to add particles to a ParticleSet during runtime, so that not all particles are released on the start of the run.

* `JIT-vs-Scipy tutorial <http://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_jit_vs_scipy.ipynb>`_ for a tutorial showing how JIT  and Scipy mode compare.

* `Argo float tutorial <http://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_Argofloats.ipynb>`_ for a tutorial on how to write a Kernel that mimics the vertical movement of Argo floats

* `Animated Gif tutorial <http://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_Agulhasparticles.ipynb>`_ for a  tutorial on how to created the Agulhas region animated gif on the top of this page. This tutorial also features a brief explanation of how to handle `OutOfBounds` errors.

* `Plotting tutorial <http://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_plotting.ipynb>`_ for further explanation on the plotting capabilities of Parcels


.. _installing-parcels:

Installing Parcels
==================

This is the “batteries-included” solution probably suitable for most users.

The simplest way to install Parcels is to use Anaconda and the Parcels Conda-Forge package with the latest release of Parcels.  This package will automatically install (almost) all the requirements for a fully functional installaion of Parcels.

The steps below are the installation instructions for Linux / macOS and for Windows.  If the commands for Linux / macOS and Windows differ, this is indicated with a comment at the end of the line.

1. Install Anaconda's Miniconda following the steps at https://conda.io/docs/user-guide/install/.  All the code below assumes that you download the Python-2 version, although Parcels also works with Python-3.  If you're on Linux / macOS, it also assumes that you installed Miniconda-2 to your home directory.

2. Start a terminal (Linux / macOS) or the Anaconda prompt (Windows). Activate the root (or base) environment of your Miniconda and create an environment containing Parcels, all its essential dependencies, and the nice-to-have Jupyter and Basemap package::

    source $HOME/miniconda2/bin/activate root  # Linux / macOS
    activate root                              # Windows

    conda create -n py2_parcels -c conda-forge parcels jupyter basemap basemap-data-hires

3. Activate the newly created Parcels environment, get a copy of the the Parcels tutorials and examples, and run the simplest of the examples to validate that you have a working Parcels setup::

    source $HOME/miniconda2/bin/activate py2_parcels  # Linux / macOS
    activate py2_parcels                              # Windows

    parcels_get_examples parcels_examples
    cd parcels_examples

    python example_peninsula.py --fieldset 100 100

4. Optionally, if you want to run all the examples and tutorials, start Jupyter and open the tutorial notebooks::

    jupyter notebook

5. The next time you start a terminal and want to work with Parcels, activate the environment with::

    source $HOME/miniconda2/bin/activate py2_parcels  # Linux / macOS
    activate py2_parcels                              # Windows


.. _installing-arbitrary-Git-reference:

Installing a non-released version of Parcels
============================================

There might be cases where you want to install a version of Parcels that has not been released yet.  (Perhaps, if you want to use a bleeding-edge feature which already is included on Github, but not in the conda-forge package.)

Then, just after step 2 of :ref:`installing-parcels` above, remove the conda-forge package again, and use Pip to install Parcels from Github::

    source $HOME/miniconda2/bin/activate py2_parcels  # Linux / macOS
    activate py2_parcels                              # Windows

    conda remove parcels
    pip install git+https://github.com/OceanParcels/parcels.git@master

.. _installation-dev:

Installation for developers
===========================

Parcels depends on a working Python installation, a netCDF installation, a C compiler, and various Python packages.  If you prefer to maintain your own Python installation providing all this, ``git clone`` the `master branch of Parcels <https://github.com/OceanParcels/parcels>`_ and manually ``pip install`` all packages lister under ``dependencies`` in the environment files

    * `environment_py2_linux.yml <https://raw.githubusercontent.com/OceanParcels/parcels/master/environment_py2_linux.yml>`_ for Linux,
    * `environment_py2_osx.yml <https://raw.githubusercontent.com/OceanParcels/parcels/master/environment_py2_osx.yml>`_ for OSX, or
    * `environment_py2_win.yml <https://raw.githubusercontent.com/OceanParcels/parcels/master/environment_py2_win.yml>`_ for Windows.

Then, install Parcels in an `editable way <https://pip.pypa.io/en/stable/reference/pip_install/?highlight=editable#cmdoption-e>`_ by running::

    pip install -e .


Getting involved
===================================

Parcels development is supported by Utrecht University and Imperial College London, with contributions
from the people listed on the `Contributors page
<https://github.com/OceanParcels/parcels/graphs/contributors>`_.

If you want to help out with developing, testing or get involved in another way,
please join the `mailing list
<https://mailman.ic.ac.uk/mailman/listinfo/oceanparcels>`_.


Python design overview
===================================

.. figure:: ParcelsDesign.png
   :align: center

   *The figure above gives a brief overview of how the most important classes and methods in Parcels are related.*


See below for links to the full documentation of the python code for Parcels


Writing Parcels Kernels
===================================

One of the most powerful features of Parcels is the ability to write custom Kernels (see e.g. `the Adding-a-custom-behaviour-kernel part of the Tutorial <http://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/parcels_tutorial.ipynb#Adding-a-custom-behaviour-kernel>`_). These Kernels are little snippets of code that get executed by Parcels, giving the ability to add 'behaviour' to particles.

However, there are some key limitations to the Kernels that everyone who wants to write their own should be aware of:

* Every Kernel must be a function with the following (and only those) arguments: ``(particle, fieldset, time, dt)``

* In order to run successfully in JIT mode, Kernel definitions can only contain the following types of commands:

  * Basic arithmetical operators (``+``, ``-``, ``*``, ``/``) and assignments (``=``).

  * Basic logical operators (``<``, ``==``, ``>``, ``&``, ``|``)

  * ``if`` and ``while`` loops, as well as ``break`` statements. Note that ``for``-loops are not supported in JIT mode

  * Interpolation of a ``Field`` from the ``fieldset`` at a (time, lon, lat, depth) point, using using square brackets notation. For example, to interpolate the zonal velocity (`U`) field at the particle location, use the following statement::

      value = fieldset.U[time, particle.lon, particle.lat, particle.depth]

  * Functions from the ``maths`` standard library and from the custom ``random`` library at :mod:`parcels.rng`

  * Simple ``print`` statements, such as::

      print("Some print")
      print(particle.lon)
      print("particle id: %d" % particle.id)
      print("lon: %f, lat: %f" % (particle.lon, particle.lat))

* Local variables can be used in Kernels, and these variables will be accessible in all concatenated Kernels. Note that these local variables are not shared between particles, and also not between time steps.

* Note that one has to be careful with writing kernels for vector fields on Curvilinear grids. While Parcels automatically rotates the `U` and `V` field when necessary, this is not the case for for example wind data. In that case, one will have to write their own rotation function.

All other functions and methods are not supported yet in Parcels Kernels. If there is a functionality that can not be programmed with this limited set of commands, please create an `Issue ticket <https://github.com/OceanParcels/parcels/issues>`_.

Parcels references
===================================

The following peer-reviewed articles have used Parcels

* McAdam R and E van Sebille (2018) Surface connectivity and inter-ocean exchanges from drifter-based transition matrices. *Journal of Geophysical Research*, *in press*. doi:`10.1002/2017JC013363 <https://dx.doi.org/10.1002/2017JC013363>`_

* van Sebille, E, SM Griffies, R Abernathey, TP Adams, P Berloff, A Biastoch, B Blanke, EP Chassignet, Y Cheng, CJ Cotter, E Deleersnijder, K Döös, HF Drake, S Drijfhout, SF Gary, AW Heemink, J Kjellsson, IM Koszalka, M Lange, C Lique, GA MacGilchrist, R Marsh, CG Mayorga Adame, R McAdam, F Nencioli, CB Paris, MD Piggott, JA Polton, S Rühs, SHAM Shah, MD Thomas, J Wang, PJ Wolfram, L Zanna, and JD Zika (2018) Lagrangian ocean analysis: fundamentals and practices. *Ocean Modelling*, *121*, 49-75. doi:`10.1016/j.ocemod.2017.11.008 <http://dx.doi.org/10.1016/j.ocemod.2017.11.008>`_

* Lange, M and E van Sebille (2017) Parcels v0.9: prototyping a Lagrangian Ocean Analysis framework for the petascale age. *Geoscientific Model Development*, *10*, 4175-4186. doi:`10.5194/gmd-2017-167 <https://www.geosci-model-dev.net/10/4175/2017/gmd-10-4175-2017.html>`_

Parcels funding and support
===================================

Parcels development has been supported by the following organisations:

.. figure:: funderlogos.png
   :align: center

* The `European Research Council under the H2020 Starting Grant <https://erc.europa.eu/sites/default/files/press_release/files/erc_press_release_stg2016_results.pdf>`_ `TOPIOS (grant agreement No 715386) <http://erik.vansebille.com/science/topios.html>`_.

* `Imperial College London <https://www.imperial.ac.uk/>`_ and specifically the `Grantham Institute <https://www.imperial.ac.uk/grantham/>`_.

* `Utrecht University <https://www.uu.nl/>`_ and specifically the `Institute for Marine and Atmospheric Research <https://www.uu.nl/en/research/institute-for-marine-and-atmospheric-research-imau>`_.

* The `EPSRC <https://www.epsrc.ac.uk/>`_ through an Institutional Sponsorship grant to Erik van Sebille under reference number EP/N50869X/1.

Parcels documentation
===================================

See below for the technical documentation on the different Parcels modules

.. toctree::
   :maxdepth: 0

   parcels


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

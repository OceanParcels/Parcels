.. parcels documentation master file, created by
   sphinx-quickstart on Tue Oct 20 09:58:20 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Parcels
===================================

**Parcels** (**P**\ robably **A** **R**\ eally **C**\ omputationally **E**\ fficient **L**\ agrangian **S**\ imulator) is an experimental prototype code aimed at exploring novel approaches for Lagrangian tracking of virtual ocean particles in the petascale age. 

Its code is licensed under an `open source MIT license <https://github.com/OceanPARCELS/parcels/blob/master/LICENSE.md>`_ and can be downloaded from https://github.com/OceanPARCELS/parcels.

.. figure:: http://oceanparcels.org/animated-gifs/globcurrent_fullyseeded.gif
   :align: center
   
   *Animation of virtual particles carried by ocean surface flow in the* `Agulhas Current <https://en.wikipedia.org/wiki/Agulhas_Current>`_ *off South Africa. The particles are advected with* `Parcels <http://oceanparcels.org/>`_ *in data from the* `GlobCurrent Project <http://globcurrent.ifremer.fr/products-data/products-overview>`_. *See* `this tutorial <http://nbviewer.jupyter.org/github/OceanPARCELS/parcels/blob/master/examples/tutorial_Agulhasparticles.ipynb>`_ *for the Parcels code behind this animated gif.*

Parcels development status
===================================

Parcels is currently close to what we will release at version 0.9. 
This v0.9 will be a fully-functional, feature-complete code for offline 
Lagrangian ocean analysis. See below for a list of features, or keep an eye 
on the `Github Development Timeline page
<https://github.com/OceanPARCELS/parcels/projects/1>`_

**Currently implemented**

* Advection of particles using inbuilt kernels for Runge-Kutta4, Runge-Kutta45 and Euler Forward (see :mod:`parcels.kernels.advection`)
* Ability to define and execute custom kernels (see `this part of the Tutorial <http://nbviewer.jupyter.org/github/OceanPARCELS/parcels/blob/master/examples/PARCELStutorial.ipynb#Adding-a-custom-behaviour-kernel>`_)
* Ability to add custom Variables to Particles (see `this part of the Tutorial <http://nbviewer.jupyter.org/github/OceanPARCELS/parcels/blob/master/examples/PARCELStutorial.ipynb#A-second-example-kernel:-calculating-distance-travelled>`_)
* Ability to add and remove Particles (see :func:`parcels.particleset.ParticleSet.add` and :func:`parcels.particleset.ParticleSet.remove`)
* Ability to run in both Scipy and JIT (Just-In-Time compilation) mode. The former is easier to debug, but the latter can be a factor 1,000 faster (see the `JIT-vs-Scipy tutorial <http://nbviewer.jupyter.org/github/OceanPARCELS/parcels/blob/master/examples/tutorial_jit_vs_scipy.ipynb>`_)
* Ability to read in any type of hydrodynamic field in NetCDF format, as long as the grid is rectangular (i.e. grid axes are aligned with longitude and latitude; see :mod:`parcels.grid.Grid.from_netcdf` and `this part of the Tutorial <http://nbviewer.jupyter.org/github/OceanPARCELS/parcels/blob/master/examples/PARCELStutorial.ipynb#Reading-in-data-from-arbritrary-NetCDF-files>`_)
* Output particles in NetCDF format (see :mod:`parcels.particlefile`)
* Basic plotting of particles, both on the fly and from netcdf output files (see the `plotting tutorial <http://nbviewer.jupyter.org/github/OceanPARCELS/parcels/blob/master/examples/tutorial_plotting.ipynb>`_)

**Will be implemented for v0.9**

* Three-dimensional particles, which can change depth and be advected with vertical velocities
* Diffusion of particles using suite of inbuilt kernels

**Major developed goals beyond v0.9**

* Support for non-rectangular grids, including unstructured meshes
* Implementation of parallel execution using tiling of the domain
* Faster and more efficient code
* Advanced control of particles near land boundaries


Parcels Tutorials 
===================================

The best way to get started with Parcels is to have a look at the Jupyter notebooks below:

* `Parcels tutorial <http://nbviewer.jupyter.org/github/OceanPARCELS/parcels/blob/master/examples/PARCELStutorial.ipynb>`_ for a general overview

* `Plotting tutorial <http://nbviewer.jupyter.org/github/OceanPARCELS/parcels/blob/master/examples/tutorial_plotting.ipynb>`_ for further explanation on the plotting capabilities of Parcels

* `Periodic boundaries tutorial <https://github.com/OceanPARCELS/parcels/blob/master/examples/tutorial_periodic_boundaries.ipynb>`_ for a tutorial on how to implement periodic boundary conditions

* `Animated Gif tutorial <http://nbviewer.jupyter.org/github/OceanPARCELS/parcels/blob/master/examples/tutorial_Agulhasparticles.ipynb>`_ for a  tutorial on how to created the Agulhas region animated gif on the top of this page. This tutorial also features a brief explanation of how to handle `OutOfBounds` errors.

Installing Parcels
===================================

The latest version of Parcels, including tests and examples, 
can be obtained directly from github via::
    git clone https://github.com/OceanPARCELS/parcels.git
    cd parcels; pip install -r requirements.txt
    python scripts/pull_data.py
    export PYTHONPATH="$PYTHONPATH:$PWD"

In order for Parcels to work from any directory, add the following line to 
your ~/.bash_profile::
    export PYTHONPATH="$PYTHONPATH:$PWD"

Note that a functional NetCDF install is required.

Getting involved
===================================

Parcels development is supported by Imperial College London, with contributions 
from the people listed on the `Contributors page 
<https://github.com/OceanPARCELS/parcels/graphs/contributors>`_. 

If you want to help out with developing, testing or get involved in another way, 
please join the `mailing list 
<https://mailman.ic.ac.uk/mailman/listinfo/oceanparcels>`_.


Python code documentation
===================================

.. figure:: ParcelsDesign.png
   :align: center
   
   *The figure above gives a brief overview of how the most important classes and methods in Parcels are related.*


See below for links to the full documentation of the python code for Parcels

.. toctree::
   :maxdepth: 4

   parcels


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


## PARCELS

PARCELS (acronym to be decided) is an experimental prototype code aimed at exploring novel approaches for Lagrangian tracking of virtual ocean particles in the petascale age.

### Motivation

In the last two decades, Lagrangian tracking of virtual particles in Ocean General Circulation Models has vastly increased our understanding of ocean dynamics and how currents move stuff around.

However, we are now facing a situation where our Lagrangian codes severely lag the next generation of these ocean circulation models. These ocean models are so big and massively parallel, and they produce so much data, that in a few years we may face a situation where many of the Lagrangian frameworks cannot be used on the latest data anymore.

In this project, we will scope out and develop a new generic, open-source community prototype code for Lagrangian tracking of water particles through any type of ocean circulation models. 

### Installation

The easiest way to install the latest realease of PARCELS is through pip:
```
pip install git+https://github.com/OceanPARCELS/parcels.git
```
Alternatively, the latest version of Parcels can be checked out locally with:
```
git clone https://github.com/OceanPARCELS/PARCELScode.git parcels
cd parcels; pip install -r requirements.txt
export PYTHONPATH="$PYTHONPATH:$PWD"
```

### Example
A basic example of particle advection around an idealised peninsula
(based on North et al., 2009, section 2.2.2) using 4th order
Runge-Kutta scheme is provided in the `tests` directory. The necessary
grid files are generated (using NEMO conventions) with:
```
python tests/grid_peninsula.py <xdim> <ydim>
```
where `xdim` and `ydim` are the numbers of grid cells in each
dimension. The particle advection example can then be run with:
```
python tests/test_peninsula.py -p <npart> --output
```
where `npart` is the number of evenly initialised particles. The
resulting particle trajectories can be visualised using Parcel's
utility plotting script:
```
python scripts/plotParticles.py 2d -p MyParticle.nc
```
An alternative execution mode that utilises SciPy's interpolator
functions for spatial interpolation can be utilised with:
```
python tests/test_peninsula.py scipy -p <npart> --degree <deg> --output
```
where `deg` is the degree of spatial interpoaltion to use.

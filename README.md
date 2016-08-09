## Parcels

**Parcels** (**P**robably **A** **R**eally **C**omputationally
**E**fficient **L**agrangian **S**imulator) is an experimental
prototype code aimed at exploring novel approaches for Lagrangian
tracking of virtual ocean particles in the petascale age.

### Motivation

In the last two decades, Lagrangian tracking of virtual particles in Ocean General Circulation Models has vastly increased our understanding of ocean dynamics and how currents move stuff around.

However, we are now facing a situation where our Lagrangian codes severely lag the next generation of these ocean circulation models. These ocean models are so big and massively parallel, and they produce so much data, that in a few years we may face a situation where many of the Lagrangian frameworks cannot be used on the latest data anymore.

In this project, we will scope out and develop a new generic, open-source community prototype code for Lagrangian tracking of water particles through any type of ocean circulation models. 

### Tutorial

For a brief guide to running PARCELS and some sample output, have a look at the interactive tutorial [here](http://nbviewer.jupyter.org/github/OceanPARCELS/parcels/blob/master/examples/PARCELStutorial.ipynb). 
This covers the basics of a particle simulation within PARCELS and a sample of its key features, including custom kernels.

### Installation

The latest version of Parcels, including tests and examples, can be
obtained directly from github via:
```
git clone --recursive https://github.com/OceanPARCELS/parcels.git
cd parcels; pip install -r requirements.txt
export PYTHONPATH="$PYTHONPATH:$PWD"
```
For a lighter checkout that does not include example data please omit
the `--recursive` flag from the above command. As an alternatively,
Parcels can also be installed directly via pip:
```
pip install git+https://github.com/OceanPARCELS/parcels.git
```
The above assumes that all dependencies are met, which can be achieved with:
```
curl -O https://raw.githubusercontent.com/OceanPARCELS/parcels/master/requirements.txt
pip install -r requirements.txt
```
In both cases a functional netCDF install is required.

### Example
A basic example of particle advection around an idealised peninsula
(based on North et al., 2009, section 2.2.2) using 4th order
Runge-Kutta scheme is provided in the `examples` directory. The necessary
grid files are generated (using NEMO conventions) with:
```
python examples/example_peninsula.py --grid <xdim> <ydim> -p <npart>
```
where `xdim` and `ydim` are the numbers of grid cells in each
dimension and `npart` is the number of evenly initialised
particles. The resulting particle trajectories can be visualised using
Parcel's utility plotting script:
```
python scripts/plotParticles.py 2d -p MyParticle.nc
```
An alternative execution mode that utilises SciPy's interpolator
functions for spatial interpolation can be utilised with:
```
python examples/example_peninsula.py scipy -p <npart> --degree <deg>
```
where `deg` is the degree of spatial interpoaltion to use.

Alternatively, there is also the `example_moving_eddies.py` example, in which particles move under the influence of two Gaussian eddies on an idealised grid.
To run the file, the prerequisite files must be downloaded. This is done automatically be calling the `pull_data.py` file in the `scripts` directory:

```
python scripts/pull_data.py
```

Now run the file as above (with default values taken):

```
python examples/example_moving_eddies.py
```

## Parcels

**Parcels** (**P**robably **A** **R**eally **C**omputationally
**E**fficient **L**agrangian **S**imulator) is an experimental
prototype code aimed at exploring novel approaches for Lagrangian
tracking of virtual ocean particles in the petascale age.

![AgulhasParticles](http://oceanparcels.org/animated-gifs/globcurrent_fullyseeded.gif)

*Animation of virtual particles carried by ocean surface flow in the [Agulhas Current](https://en.wikipedia.org/wiki/Agulhas_Current) off South Africa. The particles are advected with [PARCELS](http://oceanparcels.org/) in data from the [GlobCurrent Project](http://globcurrent.ifremer.fr/products-data/products-overview).*

### Motivation

In the last two decades, Lagrangian tracking of virtual particles in Ocean General Circulation Models has vastly increased our understanding of ocean dynamics and how currents move stuff around.

However, we are now facing a situation where our Lagrangian codes severely lag the next generation of these ocean circulation models. These ocean models are so big and massively parallel, and they produce so much data, that in a few years we may face a situation where many of the Lagrangian frameworks cannot be used on the latest data anymore.

In this project, we will scope out and develop a new generic, open-source community prototype code for Lagrangian tracking of water particles through any type of ocean circulation models. 

### Installation

The latest version of PARCELS, including tests and examples, can be
obtained directly from github via:
```
git clone https://github.com/OceanPARCELS/parcels.git
cd parcels; pip install -r requirements.txt
python scripts/pull_data.py
export PYTHONPATH="$PYTHONPATH:$PWD"
```
In order for PARCELS to work from any directory, add the line 
`export PYTHONPATH="$PYTHONPATH:$PWD"` to your `~/.bash_profile`

Note that a functional NetCDF install is required.

### Tutorial

For a brief guide to running PARCELS and some sample output, have a look at the [interactive tutorial](http://nbviewer.jupyter.org/github/OceanPARCELS/parcels/blob/master/examples/PARCELStutorial.ipynb).

[This tutorial](http://nbviewer.jupyter.org/github/OceanPARCELS/parcels/blob/master/examples/PARCELStutorial.ipynb) covers the basics of a particle simulation within PARCELS and a sample of its key features, including custom kernels.

There are also a number of worked-out examples in the `examples/` directory. Run these with e.g.
```
python examples/example_peninsula.py --grid 100 50 -p 10
```
And then plot the resulting particles using PARCELS simple plotting script
```
python scripts/plotParticles.py 2d -p MyParticle.nc
```
Showing the trajectories of 10 particles around an idealised peninsula (see also page 18 of the report [here](http://archimer.ifremer.fr/doc/00157/26792/24888.pdf)).
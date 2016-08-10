from parcels import *
from scripts import *
import numpy as np
import math
import datetime
from datetime import timedelta, datetime

filenames = {'U': "examples/GlobCurrent_example_data/20*.nc",
             'V': "examples/GlobCurrent_example_data/20*.nc"}
variables = {'U': 'eastward_eulerian_current_velocity',
             'V': 'northward_eulerian_current_velocity'}
dimensions = {'lat': 'lat',
              'lon': 'lon',
              'time': 'time'}
grid = Grid.from_netcdf(filenames, variables, dimensions)

pset = grid.ParticleSet(size=10, pclass=Particle, start=(31, -31), finish=(34, -31))
pset.show_velocity(t=datetime(2002, 1, 2), land=True, vmax=2)
pset.execute(AdvectionRK4, starttime=datetime(2002, 1, 2), runtime=timedelta(days=1),
             dt=timedelta(minutes=5), interval=timedelta(hours=6))
pset.show_velocity(land=True, latN=-30, latS=-36, lonE=33, lonW=18, vmax=2)

grid = Grid.from_nemo("examples/MovingEddies_data/moving_eddies*")
pset = grid.ParticleSet(size=2, pclass=Particle, lon=[3.3,  3.3], lat=[46.0, 47.8])
pset.show_velocity(lonW=2.8, lonE=3.5, latS=47.5, latN=48.3, vmax=3)

# -*- coding: utf-8 -*-
"""
test run for debugging the small dt/tolerance bug

Created on Thu Feb 27 14:44:03 2020

@author: reint fischer
"""

# Local Parcels path for Reint
# import sys
#
# sys.path.insert(0, "\\Users\\Gebruiker\\Documents\\GitHub\\parcels\\")  # Set path to find the newest parcels code

from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, ErrorCode
import numpy as np
from datetime import timedelta
from os import path


### Functions ###

def deleteparticle(particle, fieldset, time):
    """ This function deletes particles as they exit the domain and prints a message about their attributes at that moment
    """

    print('Particle '+str(particle.id)+' has died at t = '+str(time)+' at lon, lat, depth = '+str(particle.lon)+', '+str(particle.lat)+', '+str(particle.depth))
    particle.delete()

def run(outputdt=0.1,dt=0.004,runtime=1.5):
    datafile = path.join(path.dirname(__file__), 'test_data', 'dt_field')

    fieldset = FieldSet.from_parcels(datafile, allow_time_extrapolation=True)
    lon = fieldset.U.lon
    lat = fieldset.U.lat

    lons, lats = np.meshgrid(lon, lat)  # meshgrid at all gridpoints in the flow data
    lons = lons.flatten()
    lats = lats.flatten()
    inittime = np.asarray([0] * len(lons))

    pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=lons, lat=lats, time=inittime)

    outputdt = timedelta(seconds=outputdt)  # timesteps to create output at
    dt = timedelta(seconds=dt)  # timesteps to calculate particle trajectories
    runtime = timedelta(seconds=runtime)

    output_file = pset.ParticleFile(name='TEST1', outputdt=outputdt)

    pset.execute(AdvectionRK4,
                 runtime=runtime,
                 dt=dt,
                 recovery={ErrorCode.ErrorOutOfBounds: deleteparticle},
                 output_file=output_file)

    output_file.close()
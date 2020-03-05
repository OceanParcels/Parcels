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

from parcels import FieldSet, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4, ErrorCode, plotTrajectoriesFile
import numpy as np
from datetime import timedelta
from os import path
import threading


### Functions ###

def deleteparticle(particle, fieldset, time):
    """ This function deletes particles as they exit the domain and prints a message about their attributes at that moment
    """

    # print('Particle '+str(particle.id)+' has died at t = '+str(time))
    particle.delete()

class Abortion():
    abort_object = None
    aborted = False
    def __init__(self, object):
        self.abort_object = object

    def abort(self):
        self.abort_object.aborted = True
        self.aborted = True

def run(ioutputdt=0.1,idt=0.004,iruntime=1.5):
    datafile = path.join(path.dirname(__file__), 'test_data', 'dt_field')

    fieldset = FieldSet.from_parcels(datafile, allow_time_extrapolation=True)
    lon = fieldset.U.lon
    lat = fieldset.U.lat

    lons, lats = np.meshgrid(lon, lat)  # meshgrid at all gridpoints in the flow data
    lons = lons.flatten()
    lats = lats.flatten()
    inittime = np.asarray([0] * len(lons))

    pset = ParticleSet(fieldset=fieldset, pclass=ScipyParticle, lon=lons, lat=lats, time=inittime)
    abort_object = Abortion(pset)

    timer = threading.Timer(iruntime/ioutputdt+1000.,abort_object.abort)
    outputdt = timedelta(seconds=ioutputdt)  # timesteps to create output at
    dt = timedelta(seconds=idt)  # timesteps to calculate particle trajectories
    runtime = timedelta(seconds=iruntime)

    output_file = pset.ParticleFile(name='test_data/TEST2', outputdt=outputdt)
    timer.start()
    pset.execute(AdvectionRK4,
                 runtime=runtime,
                 dt=dt,
                 recovery={ErrorCode.ErrorOutOfBounds: deleteparticle}, output_file=output_file)
    timer.cancel()
    output_file.close()
    assert abort_object.aborted == False
    # particles = pset.particles
    # result = []
    # for i in range(len(particles)):
    #     result += [particles[i].time]
    # result = np.asarray(result)
    # assert np.allclose(result,iruntime)

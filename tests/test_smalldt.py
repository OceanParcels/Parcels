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

from parcels import FieldSet, ParticleSet, JITParticle, ScipyParticle, AdvectionRK4, ErrorCode
import numpy as np
from datetime import timedelta
from os import path
import threading
import pytest

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


# Functions #
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('dt', [0.004, -0.004, 0.01, -0.01, 0.1, -0.1])
def test_consistent_time_accumulation(mode, dt):
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

    ioutputdt = 0.1
    iruntime = 1.5
    outputdt = timedelta(seconds=ioutputdt)  # make this a parameter if the test result differs depending on this parameter value
    runtime = timedelta(seconds=iruntime)    # make this a parameter if the test result differs depending on this parameter value
    datafile = path.join(path.dirname(__file__), 'test_data', 'dt_field')

    fieldset = FieldSet.from_parcels(datafile, allow_time_extrapolation=True)
    lon = fieldset.U.lon
    lat = fieldset.U.lat

    lons, lats = np.meshgrid(lon[::4], lat[::4])  # meshgrid at all gridpoints in the flow data
    lons = lons.flatten()
    lats = lats.flatten()
    inittime = np.asarray([0] * len(lons))

    pset = ParticleSet(fieldset=fieldset, pclass=ptype[mode], lon=lons, lat=lats, time=inittime)
    abort_object = Abortion(pset)
    timer = threading.Timer(iruntime/ioutputdt+1000., abort_object.abort)

    output_file = pset.ParticleFile(name='test_data/TEST2', outputdt=outputdt)
    timer.start()
    pset.execute(AdvectionRK4,
                 runtime=runtime,
                 dt=timedelta(seconds=dt),
                 recovery={ErrorCode.ErrorOutOfBounds: deleteparticle}, output_file=output_file)
    timer.cancel()
    output_file.close()
    assert abort_object.aborted is False

    target_t = np.sign(dt) * iruntime
    particles = pset.particles
    result = []
    for i in range(len(particles)):
        result.append(particles[i].time)
    result = np.asarray(result)
    assert np.allclose(result, target_t)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('dt', [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
def test_numerical_stability(mode, dt):
    # [1e-8, 1e-7] are inherently unstable { ~ 100 picoseconds and below)
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

    ioutputdt = 2.0*dt
    iruntime = 100.0*dt
    outputdt = timedelta(seconds=ioutputdt)  # make this a parameter if the test result differs depending on this parameter value
    runtime = timedelta(seconds=iruntime)    # make this a parameter if the test result differs depending on this parameter value
    datafile = path.join(path.dirname(__file__), 'test_data', 'dt_field')

    fieldset = FieldSet.from_parcels(datafile, allow_time_extrapolation=True)
    lon = fieldset.U.lon
    lat = fieldset.U.lat

    lons, lats = np.meshgrid(lon[::4], lat[::4])  # meshgrid at all gridpoints in the flow data
    lons = lons.flatten()
    lats = lats.flatten()
    inittime = np.asarray([0] * len(lons))

    pset = ParticleSet(fieldset=fieldset, pclass=ptype[mode], lon=lons, lat=lats, time=inittime)
    abort_object = Abortion(pset)
    timer = threading.Timer(len(lats) * len(lons) * 0.5, abort_object.abort)

    output_file = pset.ParticleFile(name='test_data/TEST2', outputdt=outputdt)
    timer.start()
    pset.execute(AdvectionRK4,
                 runtime=runtime,
                 dt=timedelta(seconds=dt),
                 recovery={ErrorCode.ErrorOutOfBounds: deleteparticle}, output_file=output_file)
    timer.cancel()
    output_file.close()
    assert abort_object.aborted is False

    if not np.isclose(dt, 0, atol=1e-7):
        target_t = np.sign(dt) * iruntime
    else:
        target_t = 0
    particles = pset.particles
    result = []
    for i in range(len(particles)):
        result.append(particles[i].time)
    result = np.asarray(result)
    assert np.allclose(result, target_t)

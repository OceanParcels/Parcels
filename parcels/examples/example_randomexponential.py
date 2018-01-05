from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle
import numpy as np
from datetime import timedelta as delta
from parcels import random
import pytest

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def vertical_randomexponential(particle, fieldset, time, dt):
    # Kernel for random exponential variable in depth direction

    particle.depth = random.expovariate(fieldset.lambd)


def zero_fieldset(zdim=20):    # Define a flat field set with only one point in horizontal directions
    dimensions = {'lon': np.array([0.]),
                  'lat': np.array([0.]),
                  'depth': np.linspace(0, 50, zdim, dtype=np.float32)}

    data = {'U': np.zeros((1, 1, zdim), dtype=np.float32),
            'V': np.zeros((1, 1, zdim), dtype=np.float32)}

    return FieldSet.from_data(data, dimensions, mesh='flat')


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_randomexponential_example(mode, npart=2000):
    fieldset = zero_fieldset()

    # Rate parameter
    fieldset.lambd = 1.

    # Set random seed
    random.seed(123456)

    pset = ParticleSet(fieldset=fieldset, pclass=ptype[mode], lon=np.zeros(npart), lat=np.zeros(npart), depth=np.zeros(npart))

    endtime = delta(hours=1)
    dt = delta(hours=1)
    interval = delta(hours=1)

    pset.execute(vertical_randomexponential, endtime=endtime, dt=dt, interval=interval)

    depth = np.array([particle.depth for particle in pset.particles])
    expected_mean = 1./fieldset.lambd
    assert np.allclose(np.mean(depth), expected_mean, rtol=.1)

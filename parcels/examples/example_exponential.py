from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle
import numpy as np
from datetime import timedelta as delta
from parcels import random
import pytest

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def vertical_exponential(particle, fieldset, time, dt):
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
def test_expo_example(mode, npart=2000):
    fieldset = zero_fieldset()

    # Rate parameter
    fieldset.lambd = 1.

    # Set random seed
    random.seed(123456)

    pset = ParticleSet.from_line(fieldset=fieldset, size=npart, pclass=ptype[mode],
                                 start=(0., 0., 0.),
                                 finish=(0., 0., 0.))

    endtime = delta(hours=1)
    dt = delta(hours=1)
    interval = delta(hours=1)

    k_expo = pset.Kernel(vertical_exponential)

    pset.execute(k_expo, endtime=endtime, dt=dt, interval=interval,
                 output_file=pset.ParticleFile(name="ExpoParticle"),
                 show_movie=False)

    depth = np.array([particle.depth for particle in pset.particles])
    expected_mean = 1./fieldset.lambd
    assert np.allclose(np.mean(depth), expected_mean, rtol=.1)


if __name__ == "__main__":
    test_expo_example('jit')

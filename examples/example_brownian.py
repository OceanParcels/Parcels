from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle
import numpy as np
from datetime import timedelta as delta
import math
from parcels import random
import pytest

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def two_dim_brownian_flat(particle, fieldset, time, dt):
    # Kernel for simple Brownian particle diffusion in zonal and meridional direction.

    particle.lat += random.normalvariate(0, 1)*math.sqrt(2*dt*fieldset.Kh_meridional)
    particle.lon += random.normalvariate(0, 1)*math.sqrt(2*dt*fieldset.Kh_zonal)


def brownian_fieldset(xdim=200, ydim=200):     # Define a flat fieldset of zeros, for simplicity.
    dimensions = {'lon': np.linspace(0, 600000, xdim, dtype=np.float32),
                  'lat': np.linspace(0, 600000, ydim, dtype=np.float32)}

    data = {'U': np.zeros((xdim, ydim), dtype=np.float32),
            'V': np.zeros((xdim, ydim), dtype=np.float32)}

    return FieldSet.from_data(data, dimensions, mesh='flat')


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_brownian_example(mode, npart=3000):
    fieldset = brownian_fieldset()

    # Set diffusion constants.
    fieldset.Kh_meridional = 100.
    fieldset.Kh_zonal = 100.

    # Set random seed
    random.seed(123456)

    ptcls_start = 300000.  # Start all particles at same location in middle of grid.
    pset = ParticleSet.from_line(fieldset=fieldset, size=npart, pclass=ptype[mode],
                                 start=(ptcls_start, ptcls_start),
                                 finish=(ptcls_start, ptcls_start))

    endtime = delta(days=1)
    dt = delta(hours=1)
    interval = delta(hours=1)

    k_brownian = pset.Kernel(two_dim_brownian_flat)

    pset.execute(k_brownian, endtime=endtime, dt=dt, interval=interval,
                 output_file=pset.ParticleFile(name="BrownianParticle"),
                 show_movie=False)

    lats = np.array([particle.lat for particle in pset.particles])
    lons = np.array([particle.lon for particle in pset.particles])
    expected_std_lat = np.sqrt(2*fieldset.Kh_meridional*endtime.total_seconds())
    expected_std_lon = np.sqrt(2*fieldset.Kh_zonal*endtime.total_seconds())

    assert np.allclose(np.std(lats), expected_std_lat, rtol=.1)
    assert np.allclose(np.std(lons), expected_std_lon, rtol=.1)
    assert np.allclose(np.mean(lons), ptcls_start, rtol=.1)
    assert np.allclose(np.mean(lats), ptcls_start, rtol=.1)


if __name__ == "__main__":
    test_brownian_example('jit', npart=2000)

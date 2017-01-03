from parcels import Grid, ScipyParticle, JITParticle
import numpy as np
from datetime import timedelta as delta
import math
from parcels import random
import pytest

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def two_dim_brownian_flat(particle, grid, time, dt):

    # random.seed() - should a seed be included for reproducibility/testing purposes?
    # Use equation for particle diffusion.
    particle.lat += random.normalvariate(0, 1)*math.sqrt(2*dt*grid.Kh_meridional)
    particle.lon += random.normalvariate(0, 1)*math.sqrt(2*dt*grid.Kh_zonal)


def brownian_grid(xdim=200, ydim=200):     # Define a flat grid of zeros, for simplicity.
    lon = np.linspace(0, 600000, xdim, dtype=np.float32)
    lat = np.linspace(0, 600000, ydim, dtype=np.float32)

    U = np.zeros((lon.size, lat.size), dtype=np.float32)
    V = np.zeros((lon.size, lat.size), dtype=np.float32)

    return Grid.from_data(U, lon, lat, V, lon, lat, mesh='flat')


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_brownian_example(mode, npart=3000):

    grid = brownian_grid()

    # Set diffusion constants.
    grid.Kh_meridional = 100.
    grid.Kh_zonal = 100.

    ptcls_start = 300000.  # Start all particles at same location in middle of grid.
    pset = grid.ParticleSet(size=npart, pclass=ptype[mode],
                            start=(ptcls_start, ptcls_start),
                            finish=(ptcls_start, ptcls_start))

    endtime = delta(days=1)
    dt = delta(minutes=5)
    interval = delta(hours=1)

    k_brownian = pset.Kernel(two_dim_brownian_flat)

    pset.execute(k_brownian, endtime=endtime, dt=dt, interval=interval,
                 output_file=pset.ParticleFile(name="BrownianParticle"),
                 show_movie=False)

    lats = np.array([particle.lat for particle in pset.particles])
    lons = np.array([particle.lon for particle in pset.particles])
    expected_std_lat = np.sqrt(2*grid.Kh_meridional*endtime.total_seconds())
    expected_std_lon = np.sqrt(2*grid.Kh_zonal*endtime.total_seconds())

    assert np.allclose(np.std(lats), expected_std_lat, rtol=.1)
    assert np.allclose(np.std(lons), expected_std_lon, rtol=.1)
    assert np.allclose(np.mean(lons), ptcls_start, rtol=.1)
    assert np.allclose(np.mean(lats), ptcls_start, rtol=.1)


if __name__ == "__main__":
    test_brownian_example('jit', npart=2000)

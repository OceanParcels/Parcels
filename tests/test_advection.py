from parcels import Grid, Particle, JITParticle, AdvectionRK4, Geographic, GeographicPolar
import numpy as np
import pytest
from datetime import timedelta as delta


ptype = {'scipy': Particle, 'jit': JITParticle}


@pytest.fixture
def lon(xdim=200):
    return np.linspace(-170, 170, xdim, dtype=np.float32)


@pytest.fixture
def lat(ydim=100):
    return np.linspace(-80, 80, ydim, dtype=np.float32)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_advection_zonal(lon, lat, mode, npart=10):
    """ Particles at high latitude move geographically faster due to
        the pole correction in `GeographicPolar`.
    """
    U = np.ones((lon.size, lat.size), dtype=np.float32)
    V = np.zeros((lon.size, lat.size), dtype=np.float32)
    grid = Grid.from_data(U, lon, lat, V, lon, lat,
                          u_units=GeographicPolar(), v_units=Geographic())

    pset = grid.ParticleSet(npart, pclass=ptype[mode],
                            lon=np.zeros(npart, dtype=np.float32) + 20.,
                            lat=np.linspace(0, 80, npart, dtype=np.float32))
    pset.execute(AdvectionRK4, endtime=delta(hours=2), dt=delta(seconds=30))
    assert (np.diff(np.array([p.lon for p in pset])) > 1.e-4).all()


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_advection_meridional(lon, lat, mode, npart=10):
    """ Particles at high latitude move geographically faster due to
        the pole correction in `GeographicPolar`.
    """
    U = np.zeros((lon.size, lat.size), dtype=np.float32)
    V = np.ones((lon.size, lat.size), dtype=np.float32)
    grid = Grid.from_data(U, lon, lat, V, lon, lat,
                          u_units=GeographicPolar(), v_units=Geographic())

    pset = grid.ParticleSet(npart, pclass=ptype[mode],
                            lon=np.linspace(-60, 60, npart, dtype=np.float32),
                            lat=np.linspace(0, 30, npart, dtype=np.float32))
    delta_lat = np.diff(np.array([p.lat for p in pset]))
    pset.execute(AdvectionRK4, endtime=delta(hours=2), dt=delta(seconds=30))
    assert np.allclose(np.diff(np.array([p.lat for p in pset])), delta_lat, rtol=1.e-4)

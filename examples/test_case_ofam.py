from parcels import Grid, Particle, JITParticle, AdvectionRK4
from datetime import timedelta as delta
import pytest


def set_ofam_grid():
    filenames = {'U': "examples/OFAM_example_data/OFAM_simple_U.nc",
                 'V': "examples/OFAM_example_data/OFAM_simple_V.nc"}
    variables = {'U': 'u', 'V': 'v'}
    dimensions = {'lat': 'yu_ocean', 'lon': 'xu_ocean', 'depth': 'st_ocean',
                  'time': 'Time'}
    return Grid.from_netcdf(filenames, variables, dimensions)


def test_ofam_grid():
    grid = set_ofam_grid()
    assert(grid.U.lon.size == 2001)
    assert(grid.U.lat.size == 601)
    assert(grid.U.data.shape == (4, 601, 2001))
    assert(grid.V.lon.size == 2001)
    assert(grid.V.lat.size == 601)
    assert(grid.V.data.shape == (4, 601, 2001))


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_ofam_particles(mode):
    grid = set_ofam_grid()

    lonstart = [180]
    latstart = [10]

    ParticleClass = JITParticle if mode == 'jit' else Particle
    pset = grid.ParticleSet(len(lonstart), pclass=ParticleClass, lon=lonstart, lat=latstart)

    pset.execute(AdvectionRK4, runtime=delta(days=10), dt=delta(minutes=5),
                 interval=delta(hours=6))

    assert(abs(pset[0].lon - 173) < 1)
    assert(abs(pset[0].lat - 11) < 1)

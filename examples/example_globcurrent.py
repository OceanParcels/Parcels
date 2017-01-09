from parcels import Grid, ScipyParticle, JITParticle, AdvectionRK4
from datetime import timedelta as delta
import numpy as np
import pytest


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def set_globcurrent_grid(indices={}):
    filenames = {'U': "examples/GlobCurrent_example_data/20*-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc",
                 'V': "examples/GlobCurrent_example_data/20*-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc"}
    variables = {'U': 'eastward_eulerian_current_velocity', 'V': 'northward_eulerian_current_velocity'}
    dimensions = {'lat': 'lat', 'lon': 'lon',
                  'time': 'time'}
    return Grid.from_netcdf(filenames, variables, dimensions, indices)


def test_globcurrent_grid():
    grid = set_globcurrent_grid()
    assert(grid.U.lon.size == 81)
    assert(grid.U.lat.size == 41)
    assert(grid.U.data.shape == (365, 41, 81))
    assert(grid.V.lon.size == 81)
    assert(grid.V.lat.size == 41)
    assert(grid.V.data.shape == (365, 41, 81))

    indices = {'lon': [5], 'lat': range(20, 30)}
    gridsub = set_globcurrent_grid(indices=indices)
    assert np.allclose(gridsub.U.lon, grid.U.lon[indices['lon']])
    assert np.allclose(gridsub.U.lat, grid.U.lat[indices['lat']])
    assert np.allclose(gridsub.V.lon, grid.V.lon[indices['lon']])
    assert np.allclose(gridsub.V.lat, grid.V.lat[indices['lat']])


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_globcurrent_particles(mode):
    grid = set_globcurrent_grid()

    lonstart = [25]
    latstart = [-35]

    pset = grid.ParticleSet(pclass=ptype[mode], lon=lonstart, lat=latstart)

    pset.execute(AdvectionRK4, runtime=delta(days=1), dt=delta(minutes=5),
                 interval=delta(hours=1))

    assert(abs(pset[0].lon - 23.8) < 1)
    assert(abs(pset[0].lat - -35.3) < 1)

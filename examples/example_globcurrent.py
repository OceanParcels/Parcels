from parcels import Grid, ParticleSet, ScipyParticle, JITParticle, AdvectionRK4
from datetime import timedelta as delta
from os import path
from glob import glob
import numpy as np
import pytest


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def set_globcurrent_grid(filename=None, indices={}):
    if filename is None:
        filename = path.join(path.dirname(__file__), 'GlobCurrent_example_data',
                             '20*-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc')
    filenames = {'U': filename, 'V': filename}
    variables = {'U': 'eastward_eulerian_current_velocity', 'V': 'northward_eulerian_current_velocity'}
    dimensions = {'lat': 'lat', 'lon': 'lon', 'time': 'time'}
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
@pytest.mark.parametrize('dt, substart, subend, lonstart, latstart, irange', [
    (300., 0, 3, 25, -35, range(3, 9, 1)),
    (-300., 8, 10, 20, -39, range(7, 2, -1))
])
def test_globcurrent_grid_advancetime(mode, dt, substart, subend, lonstart, latstart, irange):
    basepath = path.join(path.dirname(__file__), 'GlobCurrent_example_data',
                         '20*-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc')
    files = glob(str(basepath))

    gridsub = set_globcurrent_grid(files[substart:subend])
    psetsub = ParticleSet.from_list(grid=gridsub, pclass=ptype[mode], lon=[lonstart], lat=[latstart])

    gridall = set_globcurrent_grid(files[0:10])
    psetall = ParticleSet.from_list(grid=gridall, pclass=ptype[mode], lon=[lonstart], lat=[latstart])
    if dt < 0:
        psetsub[0].time = gridsub.U.time[-1]
        psetall[0].time = gridall.U.time[-1]

    for i in irange:
        psetsub.execute(AdvectionRK4, starttime=psetsub[0].time, runtime=delta(days=1), dt=dt)
        gridsub.advancetime(set_globcurrent_grid(files[i]))

        psetall.execute(AdvectionRK4, starttime=psetall[0].time, runtime=delta(days=1), dt=dt)

    assert abs(psetsub[0].lon - psetall[0].lon) < 1e-4


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_globcurrent_particles(mode):
    grid = set_globcurrent_grid()

    lonstart = [25]
    latstart = [-35]

    pset = ParticleSet(grid, pclass=ptype[mode], lon=lonstart, lat=latstart)

    pset.execute(AdvectionRK4, runtime=delta(days=1), dt=delta(minutes=5),
                 interval=delta(hours=1))

    assert(abs(pset[0].lon - 23.8) < 1)
    assert(abs(pset[0].lat - -35.3) < 1)

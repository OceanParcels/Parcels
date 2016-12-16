from parcels import Grid, ScipyParticle, JITParticle, AdvectionRK4
from datetime import timedelta as delta
from py import path
from glob import glob
import pytest


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def set_globcurrent_grid(filename="examples/GlobCurrent_example_data/20*-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc"):
    filenames = {'U': filename,
                 'V': filename}
    variables = {'U': 'eastward_eulerian_current_velocity', 'V': 'northward_eulerian_current_velocity'}
    dimensions = {'lat': 'lat', 'lon': 'lon',
                  'time': 'time'}
    return Grid.from_netcdf(filenames, variables, dimensions)


def test_globcurrent_grid():
    grid = set_globcurrent_grid()
    assert(grid.U.lon.size == 81)
    assert(grid.U.lat.size == 41)
    assert(grid.U.data.shape == (365, 41, 81))
    assert(grid.V.lon.size == 81)
    assert(grid.V.lat.size == 41)
    assert(grid.V.data.shape == (365, 41, 81))


def test_globcurrent_grid_advancetime():
    basepath = path.local("examples/GlobCurrent_example_data/20*-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc")
    files = [path.local(fp) for fp in glob(str(basepath))]

    gridsub = set_globcurrent_grid(files[0:4])
    psetsub = gridsub.ParticleSet(1, pclass=ScipyParticle, lon=[25], lat=[-35])

    gridall = set_globcurrent_grid(files[0:10])
    psetall = gridall.ParticleSet(1, pclass=ScipyParticle, lon=[25], lat=[-35])

    for i in range(4, 9, 1):
        psetsub.execute(AdvectionRK4, starttime=psetsub[0].time, runtime=delta(days=1), dt=delta(minutes=5))
        gridsub.advancetime(set_globcurrent_grid(files[i]))

        psetall.execute(AdvectionRK4, starttime=psetall[0].time, runtime=delta(days=1), dt=delta(minutes=5))

    assert abs(psetsub[0].lon - psetall[0].lon) < 1e-4


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_globcurrent_particles(mode):
    grid = set_globcurrent_grid()

    lonstart = [25]
    latstart = [-35]

    pset = grid.ParticleSet(len(lonstart), pclass=ptype[mode], lon=lonstart, lat=latstart)

    pset.execute(AdvectionRK4, runtime=delta(days=1), dt=delta(minutes=5),
                 interval=delta(hours=1))

    assert(abs(pset[0].lon - 23.8) < 1)
    assert(abs(pset[0].lat - -35.3) < 1)

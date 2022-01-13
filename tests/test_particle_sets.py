from parcels import (FieldSet, Field, ScipyParticle, JITParticle,
                     Variable, StateCode, OperationCode, CurvilinearZGrid)
from parcels import ParticleSetSOA, ParticleFileSOA, KernelSOA  # noqa
from parcels import ParticleSetAOS, ParticleFileAOS, KernelAOS  # noqa
import numpy as np
import pytest

pset_modes = ['soa', 'aos']
ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
pset_type = {'soa': {'pset': ParticleSetSOA, 'pfile': ParticleFileSOA, 'kernel': KernelSOA},
             'aos': {'pset': ParticleSetAOS, 'pfile': ParticleFileAOS, 'kernel': KernelAOS}}


def fieldset(xdim=40, ydim=100):
    U = np.zeros((ydim, xdim), dtype=np.float32)
    V = np.zeros((ydim, xdim), dtype=np.float32)
    lon = np.linspace(0, 1, xdim, dtype=np.float32)
    lat = np.linspace(-60, 60, ydim, dtype=np.float32)
    depth = np.zeros(1, dtype=np.float32)
    data = {'U': np.array(U, dtype=np.float32), 'V': np.array(V, dtype=np.float32)}
    dimensions = {'lat': lat, 'lon': lon, 'depth': depth}
    return FieldSet.from_data(data, dimensions)


@pytest.fixture(name="fieldset")
def fieldset_fixture(xdim=40, ydim=100):
    return fieldset(xdim=xdim, ydim=ydim)


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_create_lon_lat(fieldset, pset_mode, mode, npart=100):
    lon = np.linspace(0, 1, npart, dtype=np.float32)
    lat = np.linspace(1, 0, npart, dtype=np.float32)
    pset = pset_type[pset_mode]['pset'](fieldset, lon=lon, lat=lat, pclass=ptype[mode])
    assert np.allclose([p.lon for p in pset], lon, rtol=1e-12)
    assert np.allclose([p.lat for p in pset], lat, rtol=1e-12)


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('lonlatdepth_dtype', [np.float64, np.float32])
def test_pset_create_line(fieldset, pset_mode, mode, lonlatdepth_dtype, npart=100):
    lon = np.linspace(0, 1, npart, dtype=lonlatdepth_dtype)
    lat = np.linspace(1, 0, npart, dtype=lonlatdepth_dtype)
    pset = pset_type[pset_mode]['pset'].from_line(fieldset, size=npart, start=(0, 1), finish=(1, 0),
                                                  pclass=ptype[mode], lonlatdepth_dtype=lonlatdepth_dtype)
    assert np.allclose([p.lon for p in pset], lon, rtol=1e-12)
    assert np.allclose([p.lat for p in pset], lat, rtol=1e-12)
    assert isinstance(pset[0].lat, lonlatdepth_dtype)


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_create_list_with_customvariable(fieldset, pset_mode, mode, npart=100):
    lon = np.linspace(0, 1, npart, dtype=np.float32)
    lat = np.linspace(1, 0, npart, dtype=np.float32)

    class MyParticle(ptype[mode]):
        v = Variable('v')

    v_vals = np.arange(npart)
    pset = pset_type[pset_mode]['pset'].from_list(fieldset, lon=lon, lat=lat, v=v_vals, pclass=MyParticle)
    assert np.allclose([p.lon for p in pset], lon, rtol=1e-12)
    assert np.allclose([p.lat for p in pset], lat, rtol=1e-12)
    assert np.allclose([p.v for p in pset], v_vals, rtol=1e-12)


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('restart', [True, False])
def test_pset_create_fromparticlefile(fieldset, pset_mode, mode, restart, tmpdir):
    filename = tmpdir.join("pset_fromparticlefile.nc")
    lon = np.linspace(0, 1, 10, dtype=np.float32)
    lat = np.linspace(1, 0, 10, dtype=np.float32)

    class TestParticle(ptype[mode]):
        p = Variable('p', np.float32, initial=0.33)
        p2 = Variable('p2', np.float32, initial=1, to_write=False)
        p3 = Variable('p3', np.float32, to_write='once')

    pset = pset_type[pset_mode]['pset'](fieldset, lon=lon, lat=lat, depth=[4]*len(lon), pclass=TestParticle, p3=np.arange(len(lon)))
    pfile = pset.ParticleFile(filename, outputdt=1)

    def Kernel(particle, fieldset, time):
        particle.p = 2.
        if particle.lon == 1.:
            particle.delete()

    pset.execute(Kernel, runtime=2, dt=1, output_file=pfile)
    pfile.close()

    pset_new = pset_type[pset_mode]['pset'].from_particlefile(fieldset, pclass=TestParticle, filename=filename,
                                                              restart=restart, repeatdt=1)

    for var in ['lon', 'lat', 'depth', 'time', 'p', 'p2', 'p3']:
        assert np.allclose([getattr(p, var) for p in pset], [getattr(p, var) for p in pset_new])

    if restart:
        assert np.allclose([p.id for p in pset], [p.id for p in pset_new])
    pset_new.execute(Kernel, runtime=2, dt=1)
    assert len(pset_new) == 3*len(pset)


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy'])
@pytest.mark.parametrize('lonlatdepth_dtype', [np.float64, np.float32])
def test_pset_create_field(fieldset, pset_mode, mode, lonlatdepth_dtype, npart=100):
    np.random.seed(123456)
    shape = (fieldset.U.lon.size, fieldset.U.lat.size)
    K = Field('K', lon=fieldset.U.lon, lat=fieldset.U.lat,
              data=np.ones(shape, dtype=np.float32), transpose=True)
    pset = pset_type[pset_mode]['pset'].from_field(fieldset, size=npart, pclass=ptype[mode],
                                                   start_field=K, lonlatdepth_dtype=lonlatdepth_dtype)
    assert (np.array([p.lon for p in pset]) <= K.lon[-1]).all()
    assert (np.array([p.lon for p in pset]) >= K.lon[0]).all()
    assert (np.array([p.lat for p in pset]) <= K.lat[-1]).all()
    assert (np.array([p.lat for p in pset]) >= K.lat[0]).all()
    assert isinstance(pset[0].lat, lonlatdepth_dtype)


@pytest.mark.parametrize('pset_mode', pset_modes)
def test_pset_create_field_curvi(pset_mode, npart=100):
    np.random.seed(123456)
    r_v = np.linspace(.25, 2, 20)
    theta_v = np.linspace(0, np.pi/2, 200)
    dtheta = theta_v[1]-theta_v[0]
    dr = r_v[1]-r_v[0]
    (r, theta) = np.meshgrid(r_v, theta_v)

    x = -1 + r * np.cos(theta)
    y = -1 + r * np.sin(theta)
    grid = CurvilinearZGrid(x, y)

    u = np.ones(x.shape)
    v = np.where(np.logical_and(theta > np.pi/4, theta < np.pi/3), 1, 0)

    ufield = Field('U', u, grid=grid)
    vfield = Field('V', v, grid=grid)
    fieldset = FieldSet(ufield, vfield)
    pset = pset_type[pset_mode]['pset'].from_field(fieldset, size=npart, pclass=ptype['scipy'], start_field=fieldset.V)

    lons = np.array([p.lon+1 for p in pset])
    lats = np.array([p.lat+1 for p in pset])
    thetas = np.arctan2(lats, lons)
    rs = np.sqrt(lons*lons + lats*lats)

    test = np.pi/4-dtheta < thetas
    test *= thetas < np.pi/3+dtheta
    test *= rs > .25-dr
    test *= rs < 2+dr
    assert np.all(test)


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_create_with_time(fieldset, pset_mode, mode, npart=100):
    lon = np.linspace(0, 1, npart)
    lat = np.linspace(1, 0, npart)
    time = 5.
    pset = pset_type[pset_mode]['pset'](fieldset, lon=lon, lat=lat, pclass=ptype[mode], time=time)
    assert np.allclose([p.time for p in pset], time, rtol=1e-12)
    pset = pset_type[pset_mode]['pset'].from_list(fieldset, lon=lon, lat=lat, pclass=ptype[mode],
                                                  time=[time]*npart)
    assert np.allclose([p.time for p in pset], time, rtol=1e-12)
    pset = pset_type[pset_mode]['pset'].from_line(fieldset, size=npart, start=(0, 1), finish=(1, 0),
                                                  pclass=ptype[mode], time=time)
    assert np.allclose([p.time for p in pset], time, rtol=1e-12)


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_not_multipldt_time(fieldset, pset_mode, mode):
    times = [0, 1.1]
    pset = pset_type[pset_mode]['pset'](fieldset, lon=[0]*2, lat=[0]*2, pclass=ptype[mode], time=times)

    def Addlon(particle, fieldset, time):
        particle.lon += particle.dt

    pset.execute(Addlon, dt=1, runtime=2)
    assert np.allclose([p.lon for p in pset], [2 - t for t in times])


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_repeated_release(fieldset, pset_mode, mode, npart=10):
    time = np.arange(0, npart, 1)  # release 1 particle every second
    pset = pset_type[pset_mode]['pset'](fieldset, lon=np.zeros(npart), lat=np.zeros(npart),
                                        pclass=ptype[mode], time=time)
    assert np.allclose([p.time for p in pset], time)

    def IncrLon(particle, fieldset, time):
        particle.lon += 1.
    pset.execute(IncrLon, dt=1., runtime=npart)
    assert np.allclose([p.lon for p in pset], np.arange(npart, 0, -1))


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_dt0(fieldset, pset_mode, mode, npart=10):
    pset = pset_type[pset_mode]['pset'](fieldset, lon=np.zeros(npart), lat=np.zeros(npart),
                                        pclass=ptype[mode])

    def IncrLon(particle, fieldset, time):
        particle.lon += 1
    pset.execute(IncrLon, dt=0., runtime=npart)
    assert np.allclose([p.lon for p in pset], 1.)
    assert np.allclose([p.time for p in pset], 0.)


@pytest.mark.parametrize('pset_mode', pset_modes)
def test_pset_repeatdt_check_dt(pset_mode, fieldset):
    pset = pset_type[pset_mode]['pset'](fieldset, lon=[0], lat=[0], pclass=ScipyParticle, repeatdt=5)

    def IncrLon(particle, fieldset, time):
        particle.lon = 1.
    pset.execute(IncrLon, dt=2, runtime=21)
    assert np.allclose([p.lon for p in pset], 1)  # if p.dt is nan, it won't be executed so p.lon will be 0


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_repeatdt_custominit(fieldset, pset_mode, mode):
    class MyParticle(ptype[mode]):
        sample_var = Variable('sample_var')

    pset = pset_type[pset_mode]['pset'](fieldset, lon=0, lat=0, pclass=MyParticle, repeatdt=1, sample_var=5)

    def DoNothing(particle, fieldset, time):
        return StateCode.Success

    pset.execute(DoNothing, dt=1, runtime=21)
    assert np.allclose([p.sample_var for p in pset], 5.)


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_stop_simulation(fieldset, pset_mode, mode):
    pset = pset_type[pset_mode]['pset'](fieldset, lon=0, lat=0, pclass=ptype[mode])

    def Delete(particle, fieldset, time):
        if time == 4:
            return OperationCode.StopExecution

    pset.execute(Delete, dt=1, runtime=21)
    assert pset[0].time == 4


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_access(fieldset, pset_mode, mode, npart=100):
    lon = np.linspace(0, 1, npart, dtype=np.float32)
    lat = np.linspace(1, 0, npart, dtype=np.float32)
    pset = pset_type[pset_mode]['pset'](fieldset, lon=lon, lat=lat, pclass=ptype[mode])
    assert(pset.size == 100)
    assert np.allclose([pset[i].lon for i in range(pset.size)], lon, rtol=1e-12)
    assert np.allclose([pset[i].lat for i in range(pset.size)], lat, rtol=1e-12)


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_custom_ptype(fieldset, pset_mode, mode, npart=100):
    class TestParticle(ptype[mode]):
        p = Variable('p', np.float32, initial=0.33)
        n = Variable('n', np.int32, initial=2)

    pset = pset_type[pset_mode]['pset'](fieldset, pclass=TestParticle,
                                        lon=np.linspace(0, 1, npart),
                                        lat=np.linspace(1, 0, npart))
    assert(pset.size == npart)
    assert np.allclose([p.p - 0.33 for p in pset], np.zeros(npart), atol=1e-5)
    assert np.allclose([p.n - 2 for p in pset], np.zeros(npart), rtol=1e-12)


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_add_explicit(fieldset, pset_mode, mode, npart=100):
    lon = np.linspace(0, 1, npart)
    lat = np.linspace(1, 0, npart)
    pset = pset_type[pset_mode]['pset'](fieldset, lon=[], lat=[], pclass=ptype[mode], lonlatdepth_dtype=np.float64)
    for i in range(npart):
        particle = pset_type[pset_mode]['pset'](pclass=ptype[mode], lon=lon[i], lat=lat[i],
                                                fieldset=fieldset, lonlatdepth_dtype=np.float64)
        pset.add(particle)
    assert(pset.size == 100)
    assert np.allclose([p.lon for p in pset], lon, rtol=1e-12)
    assert np.allclose([p.lat for p in pset], lat, rtol=1e-12)


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_add_shorthand(fieldset, pset_mode, mode, npart=100):
    lon = np.linspace(0, 1, npart, dtype=np.float32)
    lat = np.linspace(1, 0, npart, dtype=np.float32)
    pset = pset_type[pset_mode]['pset'](fieldset, lon=[], lat=[], pclass=ptype[mode])
    for i in range(npart):
        pset += pset_type[pset_mode]['pset'](pclass=ptype[mode], lon=lon[i], lat=lat[i], fieldset=fieldset)
    assert(pset.size == 100)
    assert np.allclose([p.lon for p in pset], lon, rtol=1e-12)
    assert np.allclose([p.lat for p in pset], lat, rtol=1e-12)


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_add_execute(fieldset, pset_mode, mode, npart=10):
    def AddLat(particle, fieldset, time):
        particle.lat += 0.1

    pset = pset_type[pset_mode]['pset'](fieldset, lon=[], lat=[], pclass=ptype[mode])
    for i in range(npart):
        pset += pset_type[pset_mode]['pset'](pclass=ptype[mode], lon=0.1, lat=0.1, fieldset=fieldset)
    for _ in range(3):
        pset.execute(pset.Kernel(AddLat), runtime=1., dt=1.0)
    assert np.allclose(np.array([p.lat for p in pset]), 0.4, rtol=1e-12)


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_merge_inplace(fieldset, pset_mode, mode, npart=100):
    pset1 = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode],
                                         lon=np.linspace(0, 1, npart),
                                         lat=np.linspace(1, 0, npart))
    pset2 = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode],
                                         lon=np.linspace(0, 1, npart),
                                         lat=np.linspace(0, 1, npart))
    assert(pset1.size == 100)
    assert(pset2.size == 100)
    pset1.add(pset2)
    assert(pset1.size == 200)


@pytest.mark.xfail(reason="ParticleSet duplication has not been implemented yet")
@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_merge_duplicate(fieldset, pset_mode, mode, npart=100):
    pset1 = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode],
                                         lon=np.linspace(0, 1, npart),
                                         lat=np.linspace(1, 0, npart))
    pset2 = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode],
                                         lon=np.linspace(0, 1, npart),
                                         lat=np.linspace(0, 1, npart))
    pset3 = pset1 + pset2
    assert(pset1.size == 100)
    assert(pset2.size == 100)
    assert(pset3.size == 200)


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_remove_index(fieldset, pset_mode, mode, npart=100):
    lon = np.linspace(0, 1, npart)
    lat = np.linspace(1, 0, npart)
    pset = pset_type[pset_mode]['pset'](fieldset, lon=lon, lat=lat, pclass=ptype[mode], lonlatdepth_dtype=np.float64)
    for ilon, ilat in zip(lon[::-1], lat[::-1]):
        assert(pset[-1].lon == ilon)
        assert(pset[-1].lat == ilat)
        pset.remove_indices(-1)
    assert(pset.size == 0)


@pytest.mark.xfail(reason="Particle removal has not been implemented yet")
@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_remove_particle(fieldset, pset_mode, mode, npart=100):
    lon = np.linspace(0, 1, npart)
    lat = np.linspace(1, 0, npart)
    pset = pset_type[pset_mode]['pset'](fieldset, lon=lon, lat=lat, pclass=ptype[mode])
    for ilon, ilat in zip(lon[::-1], lat[::-1]):
        assert(pset.lon[-1] == ilon)
        assert(pset.lat[-1] == ilat)
        pset.remove_indices(pset[-1])
    assert(pset.size == 0)


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_remove_kernel(fieldset, pset_mode, mode, npart=100):
    def DeleteKernel(particle, fieldset, time):
        if particle.lon >= .4:
            particle.delete()

    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode],
                                        lon=np.linspace(0, 1, npart),
                                        lat=np.linspace(1, 0, npart))
    pset.execute(pset.Kernel(DeleteKernel), endtime=1., dt=1.0)
    assert(pset.size == 40)


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_multi_execute(fieldset, pset_mode, mode, npart=10, n=5):
    def AddLat(particle, fieldset, time):
        particle.lat += 0.1

    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode],
                                        lon=np.linspace(0, 1, npart),
                                        lat=np.zeros(npart))
    k_add = pset.Kernel(AddLat)
    for _ in range(n):
        pset.execute(k_add, runtime=1., dt=1.0)
    assert np.allclose([p.lat - n*0.1 for p in pset], np.zeros(npart), rtol=1e-12)


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_multi_execute_delete(fieldset, pset_mode, mode, npart=10, n=5):
    def AddLat(particle, fieldset, time):
        particle.lat += 0.1

    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode],
                                        lon=np.linspace(0, 1, npart),
                                        lat=np.zeros(npart))
    k_add = pset.Kernel(AddLat)
    for _ in range(n):
        pset.execute(k_add, runtime=1., dt=1.0)
        pset.remove_indices(-1)
    assert np.allclose([p.lat - n*0.1 for p in pset], np.zeros(npart - n), rtol=1e-12)


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('area_scale', [True, False])
def test_density(fieldset, pset_mode, mode, area_scale):
    lons, lats = np.meshgrid(np.linspace(0.05, 0.95, 10), np.linspace(-30, 30, 20))
    pset = pset_type[pset_mode]['pset'](fieldset, pclass=ptype[mode], lon=lons, lat=lats)
    arr = pset.density(area_scale=area_scale)
    if area_scale:
        assert np.allclose(arr, 1 / fieldset.U.cell_areas(), rtol=1e-3)  # check that density equals 1/area
    else:
        assert(np.sum(arr) == lons.size)  # check conservation of particles
        inds = np.where(arr)
        for i in range(len(inds[0])):  # check locations (low atol because of coarse grid)
            assert np.allclose(fieldset.U.lon[inds[1][i]], pset[i].lon, atol=fieldset.U.lon[1]-fieldset.U.lon[0])
            assert np.allclose(fieldset.U.lat[inds[0][i]], pset[i].lat, atol=fieldset.U.lat[1]-fieldset.U.lat[0])


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('staggered_grid', ['Agrid', 'Cgrid'])
def test_from_field_exact_val(pset_mode, staggered_grid):
    xdim = 4
    ydim = 3

    lon = np.linspace(-1, 2, xdim, dtype=np.float32)
    lat = np.linspace(50, 52, ydim, dtype=np.float32)

    dimensions = {'lat': lat, 'lon': lon}
    if staggered_grid == 'Agrid':
        U = np.zeros((ydim, xdim), dtype=np.float32)
        V = np.zeros((ydim, xdim), dtype=np.float32)
        data = {'U': np.array(U, dtype=np.float32), 'V': np.array(V, dtype=np.float32)}
        mask = np.array([[1, 1, 0, 0],
                         [1, 1, 1, 0],
                         [1, 1, 1, 1]])
        fieldset = FieldSet.from_data(data, dimensions, mesh='flat')

        FMask = Field('mask', mask, lon, lat)
        fieldset.add_field(FMask)
    elif staggered_grid == 'Cgrid':
        U = np.array([[0, 0, 0, 0],
                      [1, 0, 0, 0],
                      [1, 1, 0, 0]])
        V = np.array([[0, 1, 0, 0],
                      [0, 1, 0, 0],
                      [0, 1, 1, 0]])
        data = {'U': np.array(U, dtype=np.float32), 'V': np.array(V, dtype=np.float32)}
        mask = np.array([[-1, -1, -1, -1],
                         [-1, 1, 0, 0],
                         [-1, 1, 1, 0]])
        fieldset = FieldSet.from_data(data, dimensions, mesh='flat')
        fieldset.U.interp_method = 'cgrid_velocity'
        fieldset.V.interp_method = 'cgrid_velocity'

        FMask = Field('mask', mask, lon, lat, interp_method='cgrid_tracer')
        fieldset.add_field(FMask)

    class SampleParticle(ptype['scipy']):
        mask = Variable('mask', initial=fieldset.mask)

    pset = pset_type[pset_mode]['pset'].from_field(fieldset, size=400, pclass=SampleParticle, start_field=FMask, time=0)
    # pset.show(field=FMask)
    assert np.allclose([p.mask for p in pset], 1)
    assert (np.array([p.lon for p in pset]) <= 1).all()
    test = np.logical_or(np.array([p.lon for p in pset]) <= 0, np.array([p.lat for p in pset]) >= 51)
    assert test.all()

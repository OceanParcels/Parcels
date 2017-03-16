from parcels import FieldSet, ParticleSet, Field, ScipyParticle, JITParticle
import numpy as np
import pytest


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


@pytest.fixture
def fieldset(xdim=100, ydim=100):
    U = np.zeros((xdim, ydim), dtype=np.float32)
    V = np.zeros((xdim, ydim), dtype=np.float32)
    lon = np.linspace(0, 1, xdim, dtype=np.float32)
    lat = np.linspace(0, 1, ydim, dtype=np.float32)
    depth = np.zeros(1, dtype=np.float32)
    time = np.zeros(1, dtype=np.float64)
    data = {'U': np.array(U, dtype=np.float32), 'V': np.array(V, dtype=np.float32)}
    dimensions = {'lat': lat, 'lon': lon, 'depth': depth, 'time': time}
    return FieldSet.from_data(data, dimensions)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_create_lon_lat(fieldset, mode, npart=100):
    lon = np.linspace(0, 1, npart, dtype=np.float32)
    lat = np.linspace(1, 0, npart, dtype=np.float32)
    pset = ParticleSet(fieldset, lon=lon, lat=lat, pclass=ptype[mode])
    assert np.allclose([p.lon for p in pset], lon, rtol=1e-12)
    assert np.allclose([p.lat for p in pset], lat, rtol=1e-12)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_create_line(fieldset, mode, npart=100):
    lon = np.linspace(0, 1, npart, dtype=np.float32)
    lat = np.linspace(1, 0, npart, dtype=np.float32)
    pset = ParticleSet.from_line(fieldset, size=npart, start=(0, 1), finish=(1, 0), pclass=ptype[mode])
    assert np.allclose([p.lon for p in pset], lon, rtol=1e-12)
    assert np.allclose([p.lat for p in pset], lat, rtol=1e-12)


@pytest.mark.parametrize('mode', ['scipy'])
def test_pset_create_field(fieldset, mode, npart=100):
    np.random.seed(123456)
    shape = (fieldset.U.lon.size, fieldset.U.lat.size)
    K = Field('K', lon=fieldset.U.lon, lat=fieldset.U.lat,
              data=np.ones(shape, dtype=np.float32))
    pset = ParticleSet.from_field(fieldset, size=npart, pclass=ptype[mode], start_field=K)
    assert (np.array([p.lon for p in pset]) <= 1.).all()
    assert (np.array([p.lon for p in pset]) >= 0.).all()
    assert (np.array([p.lat for p in pset]) <= 1.).all()
    assert (np.array([p.lat for p in pset]) >= 0.).all()


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_create_with_time(fieldset, mode, npart=100):
    lon = np.linspace(0, 1, npart, dtype=np.float32)
    lat = np.linspace(1, 0, npart, dtype=np.float32)
    time = 5.
    pset = ParticleSet(fieldset, lon=lon, lat=lat, pclass=ptype[mode], time=time)
    assert np.allclose([p.time for p in pset], time, rtol=1e-12)
    pset = ParticleSet.from_list(fieldset, lon=lon, lat=lat, pclass=ptype[mode],
                                 time=[time]*npart)
    assert np.allclose([p.time for p in pset], time, rtol=1e-12)
    pset = ParticleSet.from_line(fieldset, size=npart, start=(0, 1), finish=(1, 0),
                                 pclass=ptype[mode], time=time)
    assert np.allclose([p.time for p in pset], time, rtol=1e-12)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_access(fieldset, mode, npart=100):
    lon = np.linspace(0, 1, npart, dtype=np.float32)
    lat = np.linspace(1, 0, npart, dtype=np.float32)
    pset = ParticleSet(fieldset, lon=lon, lat=lat, pclass=ptype[mode])
    assert(pset.size == 100)
    assert np.allclose([pset[i].lon for i in range(pset.size)], lon, rtol=1e-12)
    assert np.allclose([pset[i].lat for i in range(pset.size)], lat, rtol=1e-12)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_custom_ptype(fieldset, mode, npart=100):
    class TestParticle(ptype[mode]):
        user_vars = {'p': np.float32, 'n': np.int32}

        def __init__(self, *args, **kwargs):
            super(TestParticle, self).__init__(*args, **kwargs)
            self.p = 0.33
            self.n = 2

    pset = ParticleSet(fieldset, pclass=TestParticle,
                       lon=np.linspace(0, 1, npart, dtype=np.float32),
                       lat=np.linspace(1, 0, npart, dtype=np.float32))
    assert(pset.size == 100)
    # FIXME: The float test fails with a conversion error of 1.e-8
    # assert np.allclose([p.p - 0.33 for p in pset], np.zeros(npart), rtol=1e-12)
    assert np.allclose([p.n - 2 for p in pset], np.zeros(npart), rtol=1e-12)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_add_explicit(fieldset, mode, npart=100):
    lon = np.linspace(0, 1, npart, dtype=np.float32)
    lat = np.linspace(1, 0, npart, dtype=np.float32)
    pset = ParticleSet(fieldset, lon=[], lat=[], pclass=ptype[mode])
    for i in range(npart):
        particle = ptype[mode](lon=lon[i], lat=lat[i], fieldset=fieldset)
        pset.add(particle)
    assert(pset.size == 100)
    assert np.allclose([p.lon for p in pset], lon, rtol=1e-12)
    assert np.allclose([p.lat for p in pset], lat, rtol=1e-12)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_add_shorthand(fieldset, mode, npart=100):
    lon = np.linspace(0, 1, npart, dtype=np.float32)
    lat = np.linspace(1, 0, npart, dtype=np.float32)
    pset = ParticleSet(fieldset, lon=[], lat=[], pclass=ptype[mode])
    for i in range(npart):
        pset += ptype[mode](lon=lon[i], lat=lat[i], fieldset=fieldset)
    assert(pset.size == 100)
    assert np.allclose([p.lon for p in pset], lon, rtol=1e-12)
    assert np.allclose([p.lat for p in pset], lat, rtol=1e-12)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_add_execute(fieldset, mode, npart=10):
    def AddLat(particle, fieldset, time, dt):
        particle.lat += 0.1

    pset = ParticleSet(fieldset, lon=[], lat=[], pclass=ptype[mode])
    for i in range(npart):
        pset += ptype[mode](lon=0.1, lat=0.1, fieldset=fieldset)
    for _ in range(3):
        pset.execute(pset.Kernel(AddLat), starttime=0., endtime=1., dt=1.0)
    assert np.allclose(np.array([p.lat for p in pset]), 0.4, rtol=1e-12)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_merge_inplace(fieldset, mode, npart=100):
    pset1 = ParticleSet(fieldset, pclass=ptype[mode],
                        lon=np.linspace(0, 1, npart, dtype=np.float32),
                        lat=np.linspace(1, 0, npart, dtype=np.float32))
    pset2 = ParticleSet(fieldset, pclass=ptype[mode],
                        lon=np.linspace(0, 1, npart, dtype=np.float32),
                        lat=np.linspace(0, 1, npart, dtype=np.float32))
    assert(pset1.size == 100)
    assert(pset2.size == 100)
    pset1.add(pset2)
    assert(pset1.size == 200)


@pytest.mark.xfail(reason="ParticleSet duplication has not been implemented yet")
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_merge_duplicate(fieldset, mode, npart=100):
    pset1 = ParticleSet(fieldset, pclass=ptype[mode],
                        lon=np.linspace(0, 1, npart, dtype=np.float32),
                        lat=np.linspace(1, 0, npart, dtype=np.float32))
    pset2 = ParticleSet(fieldset, pclass=ptype[mode],
                        lon=np.linspace(0, 1, npart, dtype=np.float32),
                        lat=np.linspace(0, 1, npart, dtype=np.float32))
    pset3 = pset1 + pset2
    assert(pset1.size == 100)
    assert(pset2.size == 100)
    assert(pset3.size == 200)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_remove_index(fieldset, mode, npart=100):
    lon = np.linspace(0, 1, npart, dtype=np.float32)
    lat = np.linspace(1, 0, npart, dtype=np.float32)
    pset = ParticleSet(fieldset, lon=lon, lat=lat, pclass=ptype[mode])
    for ilon, ilat in zip(lon[::-1], lat[::-1]):
        p = pset.remove(-1)
        assert(p.lon == ilon)
        assert(p.lat == ilat)
    assert(pset.size == 0)


@pytest.mark.xfail(reason="Particle removal has not been implemented yet")
@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_remove_particle(fieldset, mode, npart=100):
    lon = np.linspace(0, 1, npart, dtype=np.float32)
    lat = np.linspace(1, 0, npart, dtype=np.float32)
    pset = ParticleSet(fieldset, lon=lon, lat=lat, pclass=ptype[mode])
    for ilon, ilat in zip(lon[::-1], lat[::-1]):
        p = pset.remove(pset[-1])
        assert(p.lon == ilon)
        assert(p.lat == ilat)
    assert(pset.size == 0)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_remove_kernel(fieldset, mode, npart=100):
    def DeleteKernel(particle, fieldset, time, dt):
        if particle.lon >= .4:
            particle.delete()

    pset = ParticleSet(fieldset, pclass=ptype[mode],
                       lon=np.linspace(0, 1, npart, dtype=np.float32),
                       lat=np.linspace(1, 0, npart, dtype=np.float32))
    pset.execute(pset.Kernel(DeleteKernel), starttime=0., endtime=1., dt=1.0)
    assert(pset.size == 40)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_multi_execute(fieldset, mode, npart=10, n=5):
    def AddLat(particle, fieldset, time, dt):
        particle.lat += 0.1

    pset = ParticleSet(fieldset, pclass=ptype[mode],
                       lon=np.linspace(0, 1, npart, dtype=np.float32),
                       lat=np.zeros(npart, dtype=np.float32))
    k_add = pset.Kernel(AddLat)
    for _ in range(n):
        pset.execute(k_add, starttime=0., endtime=1., dt=1.0)
    assert np.allclose([p.lat - n*0.1 for p in pset], np.zeros(npart), rtol=1e-12)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pset_multi_execute_delete(fieldset, mode, npart=10, n=5):
    def AddLat(particle, fieldset, time, dt):
        particle.lat += 0.1

    pset = ParticleSet(fieldset, pclass=ptype[mode],
                       lon=np.linspace(0, 1, npart, dtype=np.float32),
                       lat=np.zeros(npart, dtype=np.float32))
    k_add = pset.Kernel(AddLat)
    for _ in range(n):
        pset.execute(k_add, starttime=0., endtime=1., dt=1.0)
        pset.remove(-1)
    assert np.allclose([p.lat - n*0.1 for p in pset], np.zeros(npart - n), rtol=1e-12)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_density(fieldset, mode, npart=10):
    pset = ParticleSet(fieldset, pclass=ptype[mode],
                       lon=np.linspace(0, 1, npart, dtype=np.float32),
                       lat=0.5*np.ones(npart, dtype=np.float32))
    arr = pset.density(area_scale=False)
    assert(np.sum(arr) == npart)  # check conservation of particles
    inds = zip(*np.where(arr))
    for i in range(len(inds)):  # check locations (low rtol because of coarse grid)
        assert np.allclose(fieldset.U.lon[inds[i][0]], pset[i].lon, rtol=1e-1)
        assert np.allclose(fieldset.U.lat[inds[i][1]], pset[i].lat, rtol=1e-1)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pfile_array_remove_particles(fieldset, mode, tmpdir, npart=10):
    filepath = tmpdir.join("pfile_array_remove_particles")
    pset = ParticleSet(fieldset, pclass=ptype[mode],
                       lon=np.linspace(0, 1, npart, dtype=np.float32),
                       lat=0.5*np.ones(npart, dtype=np.float32))
    pfile = pset.ParticleFile(filepath)
    pfile.write(pset, 0)
    pset.remove(3)
    pfile.write(pset, 1)

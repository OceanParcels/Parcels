import numpy as np
import pytest

from parcels import (
    CurvilinearZGrid,
    Field,
    FieldSet,
    Particle,
    ParticleSet,
    ParticleSetWarning,
    StatusCode,
    Variable,
)
from tests.common_kernels import DoNothing
from tests.utils import create_fieldset_zeros_simple


@pytest.fixture
def fieldset():
    return create_fieldset_zeros_simple()


@pytest.fixture
def pset(fieldset):
    npart = 10
    pset = ParticleSet(fieldset, pclass=Particle, lon=np.linspace(0, 1, npart), lat=np.zeros(npart))
    return pset


def test_pset_create_lon_lat(fieldset):
    npart = 100
    lon = np.linspace(0, 1, npart, dtype=np.float32)
    lat = np.linspace(1, 0, npart, dtype=np.float32)
    pset = ParticleSet(fieldset, lon=lon, lat=lat, pclass=Particle)
    assert np.allclose([p.lon for p in pset], lon, rtol=1e-12)
    assert np.allclose([p.lat for p in pset], lat, rtol=1e-12)


@pytest.mark.parametrize("lonlatdepth_dtype", [np.float64, np.float32])
def test_pset_create_line(fieldset, lonlatdepth_dtype):
    npart = 100
    lon = np.linspace(0, 1, npart, dtype=lonlatdepth_dtype)
    lat = np.linspace(1, 0, npart, dtype=lonlatdepth_dtype)
    pset = ParticleSet.from_line(
        fieldset, size=npart, start=(0, 1), finish=(1, 0), pclass=Particle, lonlatdepth_dtype=lonlatdepth_dtype
    )
    assert np.allclose([p.lon for p in pset], lon, rtol=1e-12)
    assert np.allclose([p.lat for p in pset], lat, rtol=1e-12)
    assert isinstance(pset[0].lat, lonlatdepth_dtype)


def test_create_empty_pset(fieldset):
    pset = ParticleSet(fieldset, pclass=Particle)
    assert pset.size == 0

    pset.execute(DoNothing, endtime=1.0, dt=1.0)
    assert pset.size == 0


def test_pset_create_list_with_customvariable(fieldset):
    npart = 100
    lon = np.linspace(0, 1, npart, dtype=np.float32)
    lat = np.linspace(1, 0, npart, dtype=np.float32)

    MyParticle = Particle.add_variable("v")

    v_vals = np.arange(npart)
    pset = ParticleSet.from_list(fieldset, lon=lon, lat=lat, v=v_vals, pclass=MyParticle)
    assert np.allclose([p.lon for p in pset], lon, rtol=1e-12)
    assert np.allclose([p.lat for p in pset], lat, rtol=1e-12)
    assert np.allclose([p.v for p in pset], v_vals, rtol=1e-12)


@pytest.mark.parametrize("restart", [True, False])
def test_pset_create_fromparticlefile(fieldset, restart, tmp_zarrfile):
    lon = np.linspace(0, 1, 10, dtype=np.float32)
    lat = np.linspace(1, 0, 10, dtype=np.float32)

    TestParticle = Particle.add_variable("p", np.float32, initial=0.33)
    TestParticle = TestParticle.add_variable("p2", np.float32, initial=1, to_write=False)
    TestParticle = TestParticle.add_variable("p3", np.float64, to_write="once")

    pset = ParticleSet(fieldset, lon=lon, lat=lat, depth=[4] * len(lon), pclass=TestParticle, p3=np.arange(len(lon)))
    pfile = pset.ParticleFile(tmp_zarrfile, outputdt=1)

    def Kernel(particle, fieldset, time):  # pragma: no cover
        particle.p = 2.0
        if particle.lon == 1.0:
            particle.delete()

    pset.execute(Kernel, runtime=2, dt=1, output_file=pfile)

    pset_new = ParticleSet.from_particlefile(
        fieldset, pclass=TestParticle, filename=tmp_zarrfile, restart=restart, repeatdt=1
    )

    for var in ["lon", "lat", "depth", "time", "p", "p2", "p3"]:
        assert np.allclose([getattr(p, var) for p in pset], [getattr(p, var) for p in pset_new])

    if restart:
        assert np.allclose([p.id for p in pset], [p.id for p in pset_new])
    pset_new.execute(Kernel, runtime=2, dt=1)
    assert len(pset_new) == 3 * len(pset)
    assert pset[0].p3.dtype == np.float64


@pytest.mark.parametrize("lonlatdepth_dtype", [np.float64, np.float32])
def test_pset_create_field(fieldset, lonlatdepth_dtype):
    npart = 100
    np.random.seed(123456)
    shape = (fieldset.U.lon.size, fieldset.U.lat.size)
    K = Field("K", lon=fieldset.U.lon, lat=fieldset.U.lat, data=np.ones(shape, dtype=np.float32), transpose=True)
    pset = ParticleSet.from_field(
        fieldset, size=npart, pclass=Particle, start_field=K, lonlatdepth_dtype=lonlatdepth_dtype
    )
    assert (np.array([p.lon for p in pset]) <= K.lon[-1]).all()
    assert (np.array([p.lon for p in pset]) >= K.lon[0]).all()
    assert (np.array([p.lat for p in pset]) <= K.lat[-1]).all()
    assert (np.array([p.lat for p in pset]) >= K.lat[0]).all()
    assert isinstance(pset[0].lat, lonlatdepth_dtype)


def test_pset_create_field_curvi():
    npart = 100
    np.random.seed(123456)
    r_v = np.linspace(0.25, 2, 20)
    theta_v = np.linspace(0, np.pi / 2, 200)
    dtheta = theta_v[1] - theta_v[0]
    dr = r_v[1] - r_v[0]
    (r, theta) = np.meshgrid(r_v, theta_v)

    x = -1 + r * np.cos(theta)
    y = -1 + r * np.sin(theta)
    grid = CurvilinearZGrid(x, y)

    u = np.ones(x.shape)
    v = np.where(np.logical_and(theta > np.pi / 4, theta < np.pi / 3), 1, 0)

    ufield = Field("U", u, grid=grid)
    vfield = Field("V", v, grid=grid)
    fieldset = FieldSet(ufield, vfield)
    pset = ParticleSet.from_field(fieldset, size=npart, pclass=Particle, start_field=fieldset.V)

    lons = np.array([p.lon + 1 for p in pset])
    lats = np.array([p.lat + 1 for p in pset])
    thetas = np.arctan2(lats, lons)
    rs = np.sqrt(lons * lons + lats * lats)

    test = np.pi / 4 - dtheta < thetas
    test *= thetas < np.pi / 3 + dtheta
    test *= rs > 0.25 - dr
    test *= rs < 2 + dr
    assert np.all(test)


def test_pset_create_with_time(fieldset):
    npart = 100
    lon = np.linspace(0, 1, npart)
    lat = np.linspace(1, 0, npart)
    time = 5.0
    pset = ParticleSet(fieldset, lon=lon, lat=lat, pclass=Particle, time=time)
    assert np.allclose([p.time for p in pset], time, rtol=1e-12)
    pset = ParticleSet.from_list(fieldset, lon=lon, lat=lat, pclass=Particle, time=[time] * npart)
    assert np.allclose([p.time for p in pset], time, rtol=1e-12)
    pset = ParticleSet.from_line(fieldset, size=npart, start=(0, 1), finish=(1, 0), pclass=Particle, time=time)
    assert np.allclose([p.time for p in pset], time, rtol=1e-12)


def test_pset_create_outside_time():
    fieldset = create_fieldset_zeros_simple(withtime=True)
    time = [-1, 0, 1, 20 * 86400]
    with pytest.warns(ParticleSetWarning, match="Some particles are set to be released*"):
        ParticleSet(fieldset, pclass=Particle, lon=[0] * len(time), lat=[0] * len(time), time=time)


def test_pset_not_multipldt_time(fieldset):
    times = [0, 1.1]
    pset = ParticleSet(fieldset, lon=[0] * 2, lat=[0] * 2, pclass=Particle, time=times)

    def Addlon(particle, fieldset, time):  # pragma: no cover
        particle_dlon += particle.dt  # noqa

    pset.execute(Addlon, dt=1, runtime=2)
    assert np.allclose([p.lon_nextloop for p in pset], [2 - t for t in times])


def test_pset_repeated_release(fieldset):
    npart = 10
    time = np.arange(0, npart, 1)  # release 1 particle every second
    pset = ParticleSet(fieldset, lon=np.zeros(npart), lat=np.zeros(npart), pclass=Particle, time=time)
    assert np.allclose([p.time for p in pset], time)

    def IncrLon(particle, fieldset, time):  # pragma: no cover
        particle_dlon += 1.0  # noqa

    pset.execute(IncrLon, dt=1.0, runtime=npart + 1)
    assert np.allclose([p.lon for p in pset], np.arange(npart, 0, -1))


def test_pset_repeatdt_check_dt(fieldset):
    pset = ParticleSet(fieldset, lon=[0], lat=[0], pclass=Particle, repeatdt=5)

    def IncrLon(particle, fieldset, time):  # pragma: no cover
        particle.lon = 1.0

    pset.execute(IncrLon, dt=2, runtime=21)
    assert np.allclose([p.lon for p in pset], 1)  # if p.dt is nan, it won't be executed so p.lon will be 0


def test_pset_repeatdt_custominit(fieldset):
    MyParticle = Particle.add_variable("sample_var")

    pset = ParticleSet(fieldset, lon=0, lat=0, pclass=MyParticle, repeatdt=1, sample_var=5)

    pset.execute(DoNothing, dt=1, runtime=21)
    assert np.allclose([p.sample_var for p in pset], 5.0)


def test_pset_stop_simulation(fieldset):
    pset = ParticleSet(fieldset, lon=0, lat=0, pclass=Particle)

    def Delete(particle, fieldset, time):  # pragma: no cover
        if time == 4:
            return StatusCode.StopExecution

    pset.execute(Delete, dt=1, runtime=21)
    assert pset[0].time == 4


def test_pset_access(fieldset):
    npart = 100
    lon = np.linspace(0, 1, npart, dtype=np.float32)
    lat = np.linspace(1, 0, npart, dtype=np.float32)
    pset = ParticleSet(fieldset, lon=lon, lat=lat, pclass=Particle)
    assert pset.size == 100
    assert np.allclose([pset[i].lon for i in range(pset.size)], lon, rtol=1e-12)
    assert np.allclose([pset[i].lat for i in range(pset.size)], lat, rtol=1e-12)


def test_pset_custom_ptype(fieldset):
    npart = 100
    TestParticle = Particle.add_variable([Variable("p", np.float32, initial=0.33), Variable("n", np.int32, initial=2)])

    pset = ParticleSet(fieldset, pclass=TestParticle, lon=np.linspace(0, 1, npart), lat=np.linspace(1, 0, npart))
    assert pset.size == npart
    assert np.allclose([p.p - 0.33 for p in pset], np.zeros(npart), atol=1e-5)
    assert np.allclose([p.n - 2 for p in pset], np.zeros(npart), rtol=1e-12)


def test_pset_add_explicit(fieldset):
    npart = 100
    lon = np.linspace(0, 1, npart)
    lat = np.linspace(1, 0, npart)
    pset = ParticleSet(fieldset, lon=[], lat=[], pclass=Particle, lonlatdepth_dtype=np.float64)
    for i in range(npart):
        particle = ParticleSet(pclass=Particle, lon=lon[i], lat=lat[i], fieldset=fieldset, lonlatdepth_dtype=np.float64)
        pset.add(particle)
    assert pset.size == npart
    assert np.allclose([p.lon for p in pset], lon, rtol=1e-12)
    assert np.allclose([p.lat for p in pset], lat, rtol=1e-12)


def test_pset_add_shorthand(fieldset):
    npart = 100
    lon = np.linspace(0, 1, npart, dtype=np.float32)
    lat = np.linspace(1, 0, npart, dtype=np.float32)
    pset = ParticleSet(fieldset, lon=[], lat=[], pclass=Particle)
    for i in range(npart):
        pset += ParticleSet(pclass=Particle, lon=lon[i], lat=lat[i], fieldset=fieldset)
    assert pset.size == npart
    assert np.allclose([p.lon for p in pset], lon, rtol=1e-12)
    assert np.allclose([p.lat for p in pset], lat, rtol=1e-12)


def test_pset_add_execute(fieldset):
    npart = 10

    def AddLat(particle, fieldset, time):  # pragma: no cover
        particle_dlat += 0.1  # noqa

    pset = ParticleSet(fieldset, lon=[], lat=[], pclass=Particle)
    for _ in range(npart):
        pset += ParticleSet(pclass=Particle, lon=0.1, lat=0.1, fieldset=fieldset)
    for _ in range(4):
        pset.execute(pset.Kernel(AddLat), runtime=1.0, dt=1.0)
    assert np.allclose(np.array([p.lat for p in pset]), 0.4, rtol=1e-12)


def test_pset_merge_inplace(fieldset):
    npart = 100
    pset1 = ParticleSet(fieldset, pclass=Particle, lon=np.linspace(0, 1, npart), lat=np.linspace(1, 0, npart))
    pset2 = ParticleSet(fieldset, pclass=Particle, lon=np.linspace(0, 1, npart), lat=np.linspace(0, 1, npart))
    assert pset1.size == npart
    assert pset2.size == npart
    pset1.add(pset2)
    assert pset1.size == 2 * npart


@pytest.mark.xfail(reason="ParticleSet duplication has not been implemented yet")
def test_pset_merge_duplicate(fieldset):
    npart = 100
    pset1 = ParticleSet(fieldset, pclass=Particle, lon=np.linspace(0, 1, npart), lat=np.linspace(1, 0, npart))
    pset2 = ParticleSet(fieldset, pclass=Particle, lon=np.linspace(0, 1, npart), lat=np.linspace(0, 1, npart))
    pset3 = pset1 + pset2
    assert pset1.size == npart
    assert pset2.size == npart
    assert pset3.size == 2 * npart


def test_pset_remove_index(fieldset):
    npart = 100
    lon = np.linspace(0, 1, npart)
    lat = np.linspace(1, 0, npart)
    pset = ParticleSet(fieldset, lon=lon, lat=lat, pclass=Particle, lonlatdepth_dtype=np.float64)
    for ilon, ilat in zip(lon[::-1], lat[::-1], strict=True):
        assert pset[-1].lon == ilon
        assert pset[-1].lat == ilat
        pset.remove_indices(-1)
    assert pset.size == 0


@pytest.mark.xfail(reason="Particle removal has not been implemented yet")
def test_pset_remove_particle(fieldset):
    npart = 100
    lon = np.linspace(0, 1, npart)
    lat = np.linspace(1, 0, npart)
    pset = ParticleSet(fieldset, lon=lon, lat=lat, pclass=Particle)
    for ilon, ilat in zip(lon[::-1], lat[::-1], strict=True):
        assert pset.lon[-1] == ilon
        assert pset.lat[-1] == ilat
        pset.remove_indices(pset[-1])
    assert pset.size == 0


def test_pset_remove_kernel(fieldset):
    npart = 100

    def DeleteKernel(particle, fieldset, time):  # pragma: no cover
        if particle.lon >= 0.4:
            particle.delete()

    pset = ParticleSet(fieldset, pclass=Particle, lon=np.linspace(0, 1, npart), lat=np.linspace(1, 0, npart))
    pset.execute(pset.Kernel(DeleteKernel), endtime=1.0, dt=1.0)
    assert pset.size == 40


def test_pset_multi_execute(fieldset):
    npart = 10
    n = 5

    def AddLat(particle, fieldset, time):  # pragma: no cover
        particle_dlat += 0.1  # noqa

    pset = ParticleSet(fieldset, pclass=Particle, lon=np.linspace(0, 1, npart), lat=np.zeros(npart))
    k_add = pset.Kernel(AddLat)
    for _ in range(n + 1):
        pset.execute(k_add, runtime=1.0, dt=1.0)
    assert np.allclose([p.lat - n * 0.1 for p in pset], np.zeros(npart), rtol=1e-12)


def test_pset_multi_execute_delete(fieldset):
    npart = 10
    n = 5

    def AddLat(particle, fieldset, time):  # pragma: no cover
        particle_dlat += 0.1  # noqa

    pset = ParticleSet(fieldset, pclass=Particle, lon=np.linspace(0, 1, npart), lat=np.zeros(npart))
    k_add = pset.Kernel(AddLat)
    for _ in range(n + 1):
        pset.execute(k_add, runtime=1.0, dt=1.0)
        pset.remove_indices(-1)
    assert np.allclose(pset.lat, n * 0.1, atol=1e-12)


@pytest.mark.parametrize("staggered_grid", ["Agrid", "Cgrid"])
def test_from_field_exact_val(staggered_grid):
    xdim = 4
    ydim = 3

    lon = np.linspace(-1, 2, xdim, dtype=np.float32)
    lat = np.linspace(50, 52, ydim, dtype=np.float32)

    dimensions = {"lat": lat, "lon": lon}
    if staggered_grid == "Agrid":
        U = np.zeros((ydim, xdim), dtype=np.float32)
        V = np.zeros((ydim, xdim), dtype=np.float32)
        data = {"U": np.array(U, dtype=np.float32), "V": np.array(V, dtype=np.float32)}
        mask = np.array([[1, 1, 0, 0],
                         [1, 1, 1, 0],
                         [1, 1, 1, 1]])  # fmt: skip
        fieldset = FieldSet.from_data(data, dimensions, mesh="flat")

        FMask = Field("mask", mask, lon, lat)
        fieldset.add_field(FMask)
    elif staggered_grid == "Cgrid":
        U = np.array([[0, 0, 0, 0],
                      [1, 0, 0, 0],
                      [1, 1, 0, 0]])  # fmt: skip
        V = np.array([[0, 1, 0, 0],
                      [0, 1, 0, 0],
                      [0, 1, 1, 0]])  # fmt: skip
        data = {"U": np.array(U, dtype=np.float32), "V": np.array(V, dtype=np.float32)}
        mask = np.array([[-1, -1, -1, -1], [-1, 1, 0, 0], [-1, 1, 1, 0]])
        fieldset = FieldSet.from_data(data, dimensions, mesh="flat")
        fieldset.U.interp_method = "cgrid_velocity"
        fieldset.V.interp_method = "cgrid_velocity"

        FMask = Field("mask", mask, lon, lat, interp_method="cgrid_tracer")
        fieldset.add_field(FMask)

    SampleParticle = Particle.add_variable("mask", initial=0)

    def SampleMask(particle, fieldset, time):  # pragma: no cover
        particle.mask = fieldset.mask[particle]

    pset = ParticleSet.from_field(fieldset, size=400, pclass=SampleParticle, start_field=FMask, time=0)
    pset.execute(SampleMask, dt=1, runtime=1)
    assert np.allclose([p.mask for p in pset], 1)
    assert (np.array([p.lon for p in pset]) <= 1).all()
    test = np.logical_or(np.array([p.lon for p in pset]) <= 0, np.array([p.lat for p in pset]) >= 51)
    assert test.all()

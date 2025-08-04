from contextlib import nullcontext as does_not_raise
from datetime import datetime, timedelta
from operator import attrgetter

import numpy as np
import pytest
import xarray as xr

from parcels import (
    Field,
    FieldSet,
    Particle,
    ParticleSet,
    ParticleSetWarning,
    Variable,
)
from parcels._datasets.structured.generic import datasets as datasets_structured
from parcels.xgrid import XGrid
from tests.common_kernels import DoNothing


@pytest.fixture
def fieldset() -> FieldSet:
    ds = datasets_structured["ds_2d_left"]
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U (A grid)"], grid, mesh_type="flat")
    V = Field("V", ds["V (A grid)"], grid, mesh_type="flat")
    return FieldSet([U, V])


def test_pset_create_lon_lat(fieldset):
    npart = 100
    lon = np.linspace(0, 1, npart, dtype=np.float32)
    lat = np.linspace(1, 0, npart, dtype=np.float32)
    pset = ParticleSet(fieldset, lon=lon, lat=lat, pclass=Particle)
    assert np.allclose([p.lon for p in pset], lon, rtol=1e-12)
    assert np.allclose([p.lat for p in pset], lat, rtol=1e-12)


def test_create_empty_pset(fieldset):
    pset = ParticleSet(fieldset, pclass=Particle)
    assert pset.size == 0

    pset.execute(DoNothing, endtime=1.0, dt=1.0)
    assert pset.size == 0


@pytest.mark.parametrize("offset", [0, 1, 200])
def test_pset_with_pids(fieldset, offset, npart=100):
    lon = np.linspace(0, 1, npart)
    lat = np.linspace(1, 0, npart)
    trajectory_ids = np.arange(offset, npart + offset)
    pset = ParticleSet(fieldset, lon=lon, lat=lat, trajectory_ids=trajectory_ids)
    assert np.allclose([p.trajectory for p in pset], trajectory_ids, atol=1e-12)


@pytest.mark.parametrize("aslist", [True, False])
def test_pset_customvars_on_pset(fieldset, aslist):
    if aslist:
        MyParticle = Particle.add_variable([Variable("sample_var"), Variable("sample_var2")])
        pset = ParticleSet(fieldset, lon=0, lat=0, pclass=MyParticle, sample_var=5.0, sample_var2=10.0)
    else:
        MyParticle = Particle.add_variable(Variable("sample_var"))
        pset = ParticleSet(fieldset, lon=0, lat=0, pclass=MyParticle, sample_var=5.0)

    pset.execute(DoNothing, dt=np.timedelta64(1, "s"), runtime=np.timedelta64(21, "s"))
    assert np.allclose([p.sample_var for p in pset], 5.0)
    if aslist:
        assert np.allclose([p.sample_var2 for p in pset], 10.0)


def test_pset_custominit_on_pset_attrgetter(fieldset):
    MyParticle = Particle.add_variable(Variable("sample_var", initial=attrgetter("lon")))

    pset = ParticleSet(fieldset, lon=3, lat=0, pclass=MyParticle)

    pset.execute(DoNothing, dt=np.timedelta64(1, "s"), runtime=np.timedelta64(21, "s"))
    assert np.allclose([p.sample_var for p in pset], 3.0)


@pytest.mark.parametrize("pset_override", [True, False])
def test_pset_custominit_on_pclass(fieldset, pset_override):
    MyParticle = Particle.add_variable(Variable("sample_var", initial=4))

    if pset_override:
        pset = ParticleSet(fieldset, lon=0, lat=0, pclass=MyParticle, sample_var=5)
    else:
        pset = ParticleSet(fieldset, lon=0, lat=0, pclass=MyParticle)

    pset.execute(DoNothing, dt=np.timedelta64(1, "s"), runtime=np.timedelta64(21, "s"))

    check_val = 5.0 if pset_override else 4.0
    assert np.allclose([p.sample_var for p in pset], check_val)


@pytest.mark.parametrize(
    "time, expectation",
    [
        (np.timedelta64(0, "s"), does_not_raise()),
        (np.datetime64("2000-01-02T00:00:00"), does_not_raise()),
        (0.0, pytest.raises(TypeError)),
        (timedelta(seconds=0), pytest.raises(TypeError)),
        (datetime(2023, 1, 1, 0, 0, 0), pytest.raises(TypeError)),
    ],
)
def test_particleset_init_time_type(fieldset, time, expectation):
    with expectation:
        ParticleSet(fieldset, lon=[0.2], lat=[5.0], time=[time], pclass=Particle)


def test_pset_create_outside_time(fieldset):
    time = xr.date_range("1999", "2001", 20)
    with pytest.warns(ParticleSetWarning, match="Some particles are set to be released*"):
        ParticleSet(fieldset, pclass=Particle, lon=[0] * len(time), lat=[0] * len(time), time=time)


@pytest.mark.parametrize(
    "dt, expectation",
    [
        (np.timedelta64(5, "s"), does_not_raise()),
        (5.0, pytest.raises(TypeError)),
        (np.datetime64("2000-01-02T00:00:00"), pytest.raises(TypeError)),
        (timedelta(seconds=2), pytest.raises(TypeError)),
    ],
)
def test_particleset_dt_type(fieldset, dt, expectation):
    pset = ParticleSet(fieldset, lon=[0.2], lat=[5.0], depth=[50.0], pclass=Particle)
    with expectation:
        pset.execute(runtime=np.timedelta64(10, "s"), dt=dt, pyfunc=DoNothing)


def test_pset_starttime_not_multiple_dt(fieldset):
    times = [0, 1, 2]
    datetimes = [fieldset.time_interval.left + np.timedelta64(t, "s") for t in times]
    pset = ParticleSet(fieldset, lon=[0] * len(times), lat=[0] * len(times), pclass=Particle, time=datetimes)

    def Addlon(particle, fieldset, time):  # pragma: no cover
        particle.dlon += particle.dt / np.timedelta64(1, "s")

    pset.execute(Addlon, dt=np.timedelta64(2, "s"), runtime=np.timedelta64(8, "s"), verbose_progress=False)
    assert np.allclose([p.lon_nextloop for p in pset], [8 - t for t in times])


@pytest.mark.parametrize(
    "runtime, expectation",
    [
        (np.timedelta64(5, "s"), does_not_raise()),
        (5.0, pytest.raises(TypeError)),
        (timedelta(seconds=2), pytest.raises(TypeError)),
        (np.datetime64("2001-01-02T00:00:00"), pytest.raises(TypeError)),
        (datetime(2000, 1, 2, 0, 0, 0), pytest.raises(TypeError)),
    ],
)
def test_particleset_runtime_type(fieldset, runtime, expectation):
    pset = ParticleSet(fieldset, lon=[0.2], lat=[5.0], depth=[50.0], pclass=Particle)
    with expectation:
        pset.execute(runtime=runtime, dt=np.timedelta64(10, "s"), pyfunc=DoNothing)


@pytest.mark.parametrize(
    "endtime, expectation",
    [
        (np.datetime64("2000-01-02T00:00:00"), does_not_raise()),
        (5.0, pytest.raises(TypeError)),
        (np.timedelta64(5, "s"), pytest.raises(TypeError)),
        (timedelta(seconds=2), pytest.raises(TypeError)),
        (datetime(2000, 1, 2, 0, 0, 0), pytest.raises(TypeError)),
    ],
)
def test_particleset_endtime_type(fieldset, endtime, expectation):
    pset = ParticleSet(fieldset, lon=[0.2], lat=[5.0], depth=[50.0], pclass=Particle)
    with expectation:
        pset.execute(endtime=endtime, dt=np.timedelta64(10, "m"), pyfunc=DoNothing)


def test_pset_add_explicit(fieldset):
    npart = 11
    lon = np.linspace(0, 1, npart)
    lat = np.linspace(1, 0, npart)
    pset = ParticleSet(fieldset, lon=lon[0], lat=lat[0], pclass=Particle)
    for i in range(1, npart):
        particle = ParticleSet(pclass=Particle, lon=lon[i], lat=lat[i], fieldset=fieldset)
        pset.add(particle)
    assert len(pset) == npart
    assert np.allclose([p.lon for p in pset], lon, atol=1e-12)
    assert np.allclose([p.lat for p in pset], lat, atol=1e-12)
    assert np.allclose(np.diff(pset._data["trajectory"]), np.ones(pset._data["trajectory"].size - 1), atol=1e-12)


def test_pset_add_implicit(fieldset):
    pset = ParticleSet(fieldset, lon=np.zeros(3), lat=np.ones(3), pclass=Particle)
    pset += ParticleSet(fieldset, lon=np.ones(4), lat=np.zeros(4), pclass=Particle)
    assert len(pset) == 7
    assert np.allclose(np.diff(pset._data["trajectory"]), np.ones(6), atol=1e-12)


def test_pset_add_implicit_in_loop(fieldset, npart=10):
    pset = ParticleSet(fieldset, lon=[], lat=[])
    for _ in range(npart):
        pset += ParticleSet(pclass=Particle, lon=0.1, lat=0.1, fieldset=fieldset)
    assert pset.size == npart


def test_pset_merge_inplace(fieldset, npart=100):
    pset1 = ParticleSet(fieldset, lon=np.linspace(0, 1, npart), lat=np.linspace(1, 0, npart))
    pset2 = ParticleSet(fieldset, lon=np.linspace(0, 1, npart), lat=np.linspace(0, 1, npart))
    assert pset1.size == npart
    assert pset2.size == npart
    pset1.add(pset2)
    assert pset1.size == 2 * npart


def test_pset_remove_index(fieldset, npart=100):
    lon = np.linspace(0, 1, npart)
    lat = np.linspace(1, 0, npart)
    pset = ParticleSet(fieldset, lon=lon, lat=lat)
    indices_to_remove = [0, 10, 20]
    pset.remove_indices(indices_to_remove)
    assert pset.size == 97
    assert not np.any(np.in1d(pset.trajectory, indices_to_remove))


def test_pset_iterator(fieldset):
    npart = 10
    pset = ParticleSet(fieldset, lon=np.zeros(npart), lat=np.ones(npart))
    for i, particle in enumerate(pset):
        assert particle.trajectory == i
    assert i == npart - 1

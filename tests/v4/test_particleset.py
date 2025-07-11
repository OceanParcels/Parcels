from contextlib import nullcontext as does_not_raise
from datetime import datetime, timedelta

import numpy as np
import pytest
import xarray as xr

from parcels import (
    AdvectionEE,
    Field,
    FieldSet,
    Particle,
    ParticleSet,
    ParticleSetWarning,
    StatusCode,
    UXPiecewiseConstantFace,
    VectorField,
    xgcm,
)
from parcels._datasets.structured.generic import datasets as datasets_structured
from parcels._datasets.unstructured.generic import datasets as datasets_unstructured
from parcels.uxgrid import UxGrid
from parcels.xgrid import XGrid


@pytest.fixture
def fieldset() -> FieldSet:
    ds = datasets_structured["ds_2d_left"]
    grid = XGrid(xgcm.Grid(ds))
    U = Field("U", ds["U (A grid)"], grid, mesh_type="flat")
    V = Field("V", ds["V (A grid)"], grid, mesh_type="flat")
    return FieldSet([U, V])


def DoNothing(particle, fieldset, time):
    pass


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
    assert np.allclose(np.diff(pset._data.trajectory), np.ones(pset._data.trajectory.size - 1), atol=1e-12)


def test_pset_add_implicit(fieldset):
    pset = ParticleSet(fieldset, lon=np.zeros(3), lat=np.ones(3), pclass=Particle)
    pset += ParticleSet(fieldset, lon=np.ones(4), lat=np.zeros(4), pclass=Particle)
    assert len(pset) == 7
    assert np.allclose(np.diff(pset._data.trajectory), np.ones(6), atol=1e-12)


def test_pset_iterator(fieldset):
    npart = 10
    pset = ParticleSet(fieldset, lon=np.zeros(npart), lat=np.ones(npart))
    for i, particle in enumerate(pset):
        assert particle.trajectory == i
    assert i == npart - 1


def test_pset_stop_simulation(fieldset):
    pset = ParticleSet(fieldset, lon=0, lat=0, pclass=Particle)

    def Delete(particle, fieldset, time):  # pragma: no cover
        if time >= fieldset.U.time[0].values + np.timedelta64(4, "s"):
            return StatusCode.StopExecution

    pset.execute(Delete, dt=np.timedelta64(1, "s"), runtime=np.timedelta64(21, "s"))
    assert pset[0].time == fieldset.U.time[0].values + np.timedelta64(4, "s")


@pytest.mark.parametrize("verbose_progress", [True, False])
def test_uxstommelgyre_pset_execute(verbose_progress):
    ds = datasets_unstructured["stommel_gyre_delaunay"]
    grid = UxGrid(grid=ds.uxgrid, z=ds.coords["nz"])
    U = Field(
        name="U",
        data=ds.U,
        grid=grid,
        mesh_type="spherical",
        interp_method=UXPiecewiseConstantFace,
    )
    V = Field(
        name="V",
        data=ds.V,
        grid=grid,
        mesh_type="spherical",
        interp_method=UXPiecewiseConstantFace,
    )
    P = Field(
        name="P",
        data=ds.p,
        grid=grid,
        mesh_type="spherical",
        interp_method=UXPiecewiseConstantFace,
    )
    UV = VectorField(name="UV", U=U, V=V)
    fieldset = FieldSet([UV, UV.U, UV.V, P])
    pset = ParticleSet(
        fieldset,
        lon=[30.0],
        lat=[5.0],
        depth=[50.0],
        time=[np.timedelta64(0, "s")],
        pclass=Particle,
    )
    pset.execute(
        runtime=np.timedelta64(10, "m"),
        dt=np.timedelta64(60, "s"),
        pyfunc=AdvectionEE,
        verbose_progress=verbose_progress,
    )


@pytest.mark.xfail(reason="Output file not implemented yet")
def test_uxstommelgyre_pset_execute_output():
    ds = datasets_unstructured["stommel_gyre_delaunay"]
    grid = UxGrid(grid=ds.uxgrid, z=ds.coords["nz"])
    U = Field(
        name="U",
        data=ds.U,
        grid=grid,
        mesh_type="spherical",
        interp_method=UXPiecewiseConstantFace,
    )
    V = Field(
        name="V",
        data=ds.V,
        grid=grid,
        mesh_type="spherical",
        interp_method=UXPiecewiseConstantFace,
    )
    P = Field(
        name="P",
        data=ds.p,
        grid=grid,
        mesh_type="spherical",
        interp_method=UXPiecewiseConstantFace,
    )
    UV = VectorField(name="UV", U=U, V=V)
    fieldset = FieldSet([UV, UV.U, UV.V, P])
    pset = ParticleSet(
        fieldset,
        lon=[30.0],
        lat=[5.0],
        depth=[50.0],
        time=[0.0],
        pclass=Particle,
    )
    output_file = pset.ParticleFile(
        name="stommel_uxarray_particles.zarr",  # the file name
        outputdt=timedelta(minutes=5),  # the time step of the outputs
    )
    pset.execute(
        runtime=np.timedelta64(10, "m"), dt=np.timedelta64(60, "s"), pyfunc=AdvectionEE, output_file=output_file
    )

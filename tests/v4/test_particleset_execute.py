import numpy as np
import pytest

from parcels import (
    AdvectionEE,
    Field,
    FieldSet,
    Particle,
    ParticleSet,
    StatusCode,
    UXPiecewiseConstantFace,
    VectorField,
    xgcm,
)
from parcels._datasets.structured.generic import datasets as datasets_structured
from parcels._datasets.unstructured.generic import datasets as datasets_unstructured
from parcels.uxgrid import UxGrid
from parcels.xgrid import XGrid
from tests.common_kernels import DoNothing


@pytest.fixture
def fieldset() -> FieldSet:
    ds = datasets_structured["ds_2d_left"]
    grid = XGrid(xgcm.Grid(ds))
    U = Field("U", ds["U (A grid)"], grid, mesh_type="flat")
    V = Field("V", ds["V (A grid)"], grid, mesh_type="flat")
    return FieldSet([U, V])


def test_pset_remove_particle_in_kernel(fieldset):
    npart = 100
    pset = ParticleSet(fieldset, lon=np.linspace(0, 1, npart), lat=np.linspace(1, 0, npart))

    def DeleteKernel(particle, fieldset, time):  # pragma: no cover
        if particle.lon >= 0.4 and particle.lon <= 0.6:
            particle.delete()

    pset.execute(pset.Kernel(DeleteKernel), runtime=np.timedelta64(1, "s"), dt=np.timedelta64(1, "s"))
    indices = [i for i in range(npart) if not (40 <= i < 60)]
    assert [p.trajectory for p in pset] == indices
    assert pset[70].trajectory == 90
    assert pset[-1].trajectory == npart - 1
    assert pset.size == 80


def test_pset_stop_simulation(fieldset):
    pset = ParticleSet(fieldset, lon=0, lat=0, pclass=Particle)

    def Delete(particle, fieldset, time):  # pragma: no cover
        if time >= fieldset.time_interval.left + np.timedelta64(4, "s"):
            return StatusCode.StopExecution

    pset.execute(Delete, dt=np.timedelta64(1, "s"), runtime=np.timedelta64(21, "s"))
    assert pset[0].time == fieldset.time_interval.left + np.timedelta64(4, "s")


@pytest.mark.parametrize("with_delete", [True, False])
def test_pset_multi_execute(fieldset, with_delete, npart=10, n=5):
    pset = ParticleSet(fieldset, lon=np.linspace(0, 1, npart), lat=np.zeros(npart))

    def AddLat(particle, fieldset, time):  # pragma: no cover
        particle_dlat += 0.1  # noqa

    k_add = pset.Kernel(AddLat)
    for _ in range(n + 1):
        pset.execute(k_add, runtime=np.timedelta64(1, "s"), dt=np.timedelta64(1, "s"))
        if with_delete:
            pset.remove_indices(len(pset) - 1)
    if with_delete:
        assert np.allclose(pset.lat, n * 0.1, atol=1e-12)
    else:
        assert np.allclose([p.lat - n * 0.1 for p in pset], np.zeros(npart), rtol=1e-12)


@pytest.mark.parametrize(
    "starttime, endtime, dt",
    [(0, 10, 1), (0, 10, 3), (2, 16, 3), (20, 10, -1), (20, 0, -2), (5, 15, None)],
)
def test_execution_endtime(fieldset, starttime, endtime, dt):
    starttime = fieldset.time_interval.left + np.timedelta64(starttime, "s")
    endtime = fieldset.time_interval.left + np.timedelta64(endtime, "s")
    dt = np.timedelta64(dt, "s")
    pset = ParticleSet(fieldset, time=starttime, lon=0, lat=0)
    pset.execute(DoNothing, endtime=endtime, dt=dt)
    assert abs(pset.time_nextloop - endtime) < np.timedelta64(1, "ms")


@pytest.mark.parametrize(
    "starttime, runtime, dt",
    [(0, 10, 1), (0, 10, 3), (2, 16, 3), (20, 10, -1), (20, 0, -2), (5, 15, None)],
)
def test_execution_runtime(fieldset, starttime, runtime, dt):
    starttime = fieldset.time_interval.left + np.timedelta64(starttime, "s")
    runtime = np.timedelta64(runtime, "s")
    sign_dt = 1 if dt is None else np.sign(dt)
    dt = np.timedelta64(dt, "s")
    pset = ParticleSet(fieldset, time=starttime, lon=0, lat=0)
    pset.execute(DoNothing, runtime=runtime, dt=dt)
    assert abs(pset.time_nextloop - starttime - runtime * sign_dt) < np.timedelta64(1, "ms")


def test_execution_fail_python_exception(fieldset, npart=10):
    pset = ParticleSet(fieldset, lon=np.linspace(0, 1, npart), lat=np.linspace(1, 0, npart))

    def PythonFail(particle, fieldset, time):  # pragma: no cover
        if particle.time >= fieldset.time_interval.left + np.timedelta64(10, "s"):
            raise RuntimeError("Enough is enough!")
        else:
            pass

    with pytest.raises(RuntimeError):
        pset.execute(PythonFail, runtime=np.timedelta64(20, "s"), dt=np.timedelta64(2, "s"))
    assert len(pset) == npart
    assert pset.time[0] == fieldset.time_interval.left + np.timedelta64(10, "s")
    assert all([time == fieldset.time_interval.left + np.timedelta64(0, "s") for time in pset.time[1:]])


def test_pset_update_particle(fieldset, npart=10):
    lon_start = np.linspace(0, 1, npart)
    lat_start = np.linspace(1, 0, npart)
    pset = ParticleSet(fieldset, lon=np.linspace(0, 1, npart), lat=np.linspace(1, 0, npart))

    def UpdateParticle(particle, fieldset, time):  # pragma: no cover
        particle.lon += 0.1
        particle.lat -= 0.1

    pset.execute(pset.Kernel(UpdateParticle), runtime=np.timedelta64(10, "s"), dt=np.timedelta64(1, "s"))
    assert np.allclose(pset.lon, lon_start + 1, atol=1e-5)
    assert np.allclose(pset.lat, lat_start - 1, atol=1e-5)


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
        outputdt=np.timedelta64(5, "m"),  # the time step of the outputs
    )
    pset.execute(
        runtime=np.timedelta64(10, "m"), dt=np.timedelta64(60, "s"), pyfunc=AdvectionEE, output_file=output_file
    )

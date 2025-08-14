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
)
from parcels._datasets.structured.generated import simple_UV_dataset
from parcels._datasets.structured.generic import datasets as datasets_structured
from parcels._datasets.unstructured.generic import datasets as datasets_unstructured
from parcels.tools.statuscodes import FieldOutOfBoundError, TimeExtrapolationError
from parcels.uxgrid import UxGrid
from parcels.xgrid import XGrid
from tests import utils
from tests.common_kernels import DoNothing


@pytest.fixture
def fieldset() -> FieldSet:
    ds = datasets_structured["ds_2d_left"]
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U (A grid)"], grid, mesh_type="flat")
    V = Field("V", ds["V (A grid)"], grid, mesh_type="flat")
    return FieldSet([U, V])


@pytest.fixture
def zonal_flow_fieldset() -> FieldSet:
    ds = simple_UV_dataset(mesh_type="flat")
    ds["U"].data[:] = 1.0
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, mesh_type="flat")
    V = Field("V", ds["V"], grid, mesh_type="flat")
    UV = VectorField("UV", U, V)
    return FieldSet([U, V, UV])


def test_pset_remove_particle_in_kernel(fieldset):
    npart = 100
    pset = ParticleSet(fieldset, lon=np.linspace(0, 1, npart), lat=np.linspace(1, 0, npart))

    def DeleteKernel(particle, fieldset, time):  # pragma: no cover
        particle.state = np.where((particle.lon >= 0.4) & (particle.lon <= 0.6), StatusCode.Delete, particle.state)

    pset.execute(pset.Kernel(DeleteKernel), runtime=np.timedelta64(1, "s"), dt=np.timedelta64(1, "s"))
    indices = [i for i in range(npart) if not (40 <= i < 60)]
    assert [p.trajectory for p in pset] == indices
    assert pset[70].trajectory == 90
    assert pset[-1].trajectory == npart - 1
    assert pset.size == 80


@pytest.mark.parametrize("npart", [1, 100])
def test_pset_stop_simulation(fieldset, npart):
    pset = ParticleSet(fieldset, lon=np.zeros(npart), lat=np.zeros(npart), pclass=Particle)

    def Delete(particle, fieldset, time):  # pragma: no cover
        particle[particle.time >= fieldset.time_interval.left + np.timedelta64(4, "s")].state = StatusCode.StopExecution

    pset.execute(Delete, dt=np.timedelta64(1, "s"), runtime=np.timedelta64(21, "s"))
    assert pset[0].time == fieldset.time_interval.left + np.timedelta64(4, "s")


@pytest.mark.parametrize("with_delete", [True, False])
def test_pset_multi_execute(fieldset, with_delete, npart=10, n=5):
    pset = ParticleSet(fieldset, lon=np.linspace(0, 1, npart), lat=np.zeros(npart))

    def AddLat(particle, fieldset, time):  # pragma: no cover
        particle.dlat += 0.1

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


def test_dont_run_particles_outside_starttime(fieldset):
    # Test forward in time (note third particle is outside endtime)
    start_times = [fieldset.time_interval.left + np.timedelta64(t, "s") for t in [0, 2, 10]]
    endtime = fieldset.time_interval.left + np.timedelta64(8, "s")

    def AddLon(particle, fieldset, time):  # pragma: no cover
        particle.lon += 1

    pset = ParticleSet(fieldset, lon=np.zeros(len(start_times)), lat=np.zeros(len(start_times)), time=start_times)
    pset.execute(AddLon, dt=np.timedelta64(1, "s"), endtime=endtime)

    np.testing.assert_array_equal(pset.lon, [8, 6, 0])
    assert pset.time_nextloop[0:1] == endtime
    assert pset.time_nextloop[2] == start_times[2]  # this particle has not been executed

    # Test backward in time (note third particle is outside endtime)
    start_times = [fieldset.time_interval.right - np.timedelta64(t, "s") for t in [0, 2, 10]]
    endtime = fieldset.time_interval.right - np.timedelta64(8, "s")

    pset = ParticleSet(fieldset, lon=np.zeros(len(start_times)), lat=np.zeros(len(start_times)), time=start_times)
    pset.execute(AddLon, dt=-np.timedelta64(1, "s"), endtime=endtime)

    np.testing.assert_array_equal(pset.lon, [8, 6, 0])
    assert pset.time_nextloop[0:1] == endtime
    assert pset.time_nextloop[2] == start_times[2]  # this particle has not been executed


def test_some_particles_throw_outofbounds(zonal_flow_fieldset):
    npart = 100
    lon = np.linspace(0, 9e5, npart)
    pset = ParticleSet(zonal_flow_fieldset, lon=lon, lat=np.zeros_like(lon))

    with pytest.raises(FieldOutOfBoundError):
        pset.execute(AdvectionEE, runtime=np.timedelta64(1_000_000, "s"), dt=np.timedelta64(10_000, "s"))


def test_delete_on_all_errors(fieldset):
    def MoveRight(particle, fieldset, time):  # pragma: no cover
        particle.dlon += 1
        fieldset.U[particle.time, particle.depth, particle.lat, particle.lon, particle]

    def DeleteAllErrorParticles(particle, fieldset, time):  # pragma: no cover
        particle[particle.state > 20].state = StatusCode.Delete

    pset = ParticleSet(fieldset, lon=[1e5, 2], lat=[0, 0])
    pset.execute([MoveRight, DeleteAllErrorParticles], runtime=np.timedelta64(10, "s"), dt=np.timedelta64(1, "s"))
    assert len(pset) == 0


def test_some_particles_throw_outoftime(fieldset):
    time = [fieldset.time_interval.left + np.timedelta64(t, "D") for t in [0, 350]]
    pset = ParticleSet(fieldset, lon=np.zeros_like(time), lat=np.zeros_like(time), time=time)

    def FieldAccessOutsideTime(particle, fieldset, time):  # pragma: no cover
        fieldset.U[particle.time + np.timedelta64(1, "D"), particle.depth, particle.lat, particle.lon, particle]

    with pytest.raises(TimeExtrapolationError):
        pset.execute(FieldAccessOutsideTime, runtime=np.timedelta64(400, "D"), dt=np.timedelta64(10, "D"))


def test_execution_check_stopallexecution(fieldset):
    def addoneLon(particle, fieldset, time):  # pragma: no cover
        particle.dlon += 1
        particle[particle.lon + particle.dlon >= 10].state = StatusCode.StopAllExecution

    pset = ParticleSet(fieldset, lon=[0, 0], lat=[0, 0])
    pset.execute(addoneLon, runtime=np.timedelta64(20, "s"), dt=np.timedelta64(1, "s"))
    np.testing.assert_allclose(pset.lon, 9)
    np.testing.assert_allclose(pset.time - fieldset.time_interval.left, np.timedelta64(9, "s"))


def test_execution_recover_out_of_bounds(fieldset):
    npart = 2

    def MoveRight(particle, fieldset, time):  # pragma: no cover
        fieldset.U[particle.time, particle.depth, particle.lat, particle.lon + 0.1, particle]
        particle.dlon += 0.1

    def MoveLeft(particle, fieldset, time):  # pragma: no cover
        inds = np.where(particle.state == StatusCode.ErrorOutOfBounds)
        print(inds, particle.state)
        particle[inds].dlon -= 1.0
        particle[inds].state = StatusCode.Success

    lon = np.linspace(0.05, 6.95, npart)
    lat = np.linspace(1, 0, npart)
    pset = ParticleSet(fieldset, lon=lon, lat=lat)
    pset.execute([MoveRight, MoveLeft], runtime=np.timedelta64(61, "s"), dt=np.timedelta64(1, "s"))
    assert len(pset) == npart
    np.testing.assert_allclose(pset.lon, [6.05, 5.95], rtol=1e-5)
    np.testing.assert_allclose(pset.lat, lat, rtol=1e-5)


@pytest.mark.parametrize(
    "starttime, runtime, dt",
    [(0, 10, 1), (0, 10, 3), (2, 16, 3), (20, 10, -1), (20, 0, -2), (5, 15, None)],
)
@pytest.mark.parametrize("npart", [1, 10])
def test_execution_runtime(fieldset, starttime, runtime, dt, npart):
    starttime = fieldset.time_interval.left + np.timedelta64(starttime, "s")
    runtime = np.timedelta64(runtime, "s")
    sign_dt = 1 if dt is None else np.sign(dt)
    dt = np.timedelta64(dt, "s")
    pset = ParticleSet(fieldset, time=starttime, lon=np.zeros(npart), lat=np.zeros(npart))
    pset.execute(DoNothing, runtime=runtime, dt=dt)
    assert all([abs(p.time_nextloop - starttime - runtime * sign_dt) < np.timedelta64(1, "ms") for p in pset])


def test_changing_dt_in_kernel(fieldset):
    def KernelCounter(particle, fieldset, time):  # pragma: no cover
        particle.lon += 1

    pset = ParticleSet(fieldset, lon=np.zeros(1), lat=np.zeros(1))
    pset.execute(KernelCounter, dt=np.timedelta64(2, "s"), runtime=np.timedelta64(5, "s"))
    assert pset.lon == 3
    print(pset.dt)
    assert pset.dt == np.timedelta64(2, "s")


@pytest.mark.parametrize("npart", [1, 100])
def test_execution_fail_python_exception(fieldset, npart):
    pset = ParticleSet(fieldset, lon=np.linspace(0, 1, npart), lat=np.linspace(1, 0, npart))

    def PythonFail(particle, fieldset, time):  # pragma: no cover
        inds = np.argwhere(particle.time >= fieldset.time_interval.left + np.timedelta64(10, "s"))
        if inds.size > 0:
            raise RuntimeError("Enough is enough!")

    with pytest.raises(RuntimeError):
        pset.execute(PythonFail, runtime=np.timedelta64(20, "s"), dt=np.timedelta64(2, "s"))
    assert len(pset) == npart
    assert all(pset.time == fieldset.time_interval.left + np.timedelta64(10, "s"))


def test_uxstommelgyre_pset_execute():
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
    )
    assert utils.round_and_hash_float_array([p.lon for p in pset]) == 1165396086
    assert utils.round_and_hash_float_array([p.lat for p in pset]) == 1142124776


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

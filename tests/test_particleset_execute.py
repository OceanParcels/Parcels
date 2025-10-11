from contextlib import nullcontext as does_not_raise
from datetime import datetime, timedelta

import numpy as np
import pytest

from parcels import (
    Field,
    FieldInterpolationError,
    FieldOutOfBoundError,
    FieldSet,
    Particle,
    ParticleFile,
    ParticleSet,
    StatusCode,
    UxGrid,
    Variable,
    VectorField,
    XGrid,
)
from parcels._datasets.structured.generated import simple_UV_dataset
from parcels._datasets.structured.generic import datasets as datasets_structured
from parcels._datasets.unstructured.generic import datasets as datasets_unstructured
from parcels.interpolators import UXPiecewiseConstantFace, UXPiecewiseLinearNode
from parcels.kernels import AdvectionEE, AdvectionRK4, AdvectionRK4_3D
from parcels.particlefile import ParticleFile
from parcels.tools.statuscodes import FieldInterpolationError, FieldOutOfBoundError, OutsideTimeInterval
from parcels.uxgrid import UxGrid
from parcels.xgrid import XGrid
from tests import utils
from tests.common_kernels import DoNothing


@pytest.fixture
def fieldset() -> FieldSet:
    ds = datasets_structured["ds_2d_left"]
    grid = XGrid.from_dataset(ds, mesh="flat")
    U = Field("U", ds["U (A grid)"], grid)
    V = Field("V", ds["V (A grid)"], grid)
    UV = VectorField("UV", U, V)
    return FieldSet([U, V, UV])


@pytest.fixture
def fieldset_no_time_interval() -> FieldSet:
    # i.e., no time variation
    ds = datasets_structured["ds_2d_left"].isel(time=0).drop_vars("time")

    grid = XGrid.from_dataset(ds, mesh="flat")
    U = Field("U", ds["U (A grid)"], grid)
    V = Field("V", ds["V (A grid)"], grid)
    UV = VectorField("UV", U, V)
    return FieldSet([U, V, UV])


@pytest.fixture
def zonal_flow_fieldset() -> FieldSet:
    ds = simple_UV_dataset(mesh="flat")
    ds["U"].data[:] = 1.0
    grid = XGrid.from_dataset(ds, mesh="flat")
    U = Field("U", ds["U"], grid)
    V = Field("V", ds["V"], grid)
    UV = VectorField("UV", U, V)
    return FieldSet([U, V, UV])


def test_pset_execute_invalid_arguments(fieldset, fieldset_no_time_interval):
    for dt in [1, np.timedelta64(0, "s"), np.timedelta64(None)]:
        with pytest.raises(
            ValueError,
            match="dt must be a non-zero datetime.timedelta or np.timedelta64 object, got .*",
        ):
            ParticleSet(fieldset, lon=[0.2], lat=[5.0], pclass=Particle).execute(AdvectionRK4, dt=dt)

    with pytest.raises(
        ValueError,
        match="runtime and endtime are mutually exclusive - provide one or the other. Got .*",
    ):
        ParticleSet(fieldset, lon=[0.2], lat=[5.0], pclass=Particle).execute(
            AdvectionRK4, runtime=np.timedelta64(1, "s"), endtime=np.datetime64("2100-01-01"), dt=np.timedelta64(1, "s")
        )

    with pytest.raises(
        ValueError,
        match="The runtime must be a datetime.timedelta or np.timedelta64 object. Got .*",
    ):
        ParticleSet(fieldset, lon=[0.2], lat=[5.0], pclass=Particle).execute(
            AdvectionRK4, runtime=1, dt=np.timedelta64(1, "s")
        )

    msg = """Calculated/provided end time of .* is not in fieldset time interval .* Either reduce your runtime, modify your provided endtime, or change your release timing.*"""
    with pytest.raises(
        ValueError,
        match=msg,
    ):
        ParticleSet(fieldset, lon=[0.2], lat=[5.0], pclass=Particle).execute(
            AdvectionRK4, endtime=np.datetime64("1990-01-01"), dt=np.timedelta64(1, "s")
        )

    with pytest.raises(
        ValueError,
        match=msg,
    ):
        ParticleSet(fieldset, lon=[0.2], lat=[5.0], pclass=Particle).execute(
            AdvectionRK4, endtime=np.datetime64("2100-01-01"), dt=np.timedelta64(-1, "s")
        )

    with pytest.raises(
        ValueError,
        match="The endtime must be of the same type as the fieldset.time_interval start time. Got .*",
    ):
        ParticleSet(fieldset, lon=[0.2], lat=[5.0], pclass=Particle).execute(
            AdvectionRK4, endtime=12345, dt=np.timedelta64(1, "s")
        )

    with pytest.raises(
        ValueError,
        match="The runtime must be provided when the time_interval is not defined for a fieldset.",
    ):
        ParticleSet(fieldset_no_time_interval, lon=[0.2], lat=[5.0], pclass=Particle).execute(
            AdvectionRK4, dt=np.timedelta64(1, "s")
        )


@pytest.mark.parametrize(
    "runtime, expectation",
    [
        (np.timedelta64(5, "s"), does_not_raise()),
        (timedelta(seconds=2), does_not_raise()),
        (5.0, pytest.raises(ValueError)),
        (np.datetime64("2001-01-02T00:00:00"), pytest.raises(ValueError)),
        (datetime(2000, 1, 2, 0, 0, 0), pytest.raises(ValueError)),
    ],
)
def test_particleset_runtime_type(fieldset, runtime, expectation):
    pset = ParticleSet(fieldset, lon=[0.2], lat=[5.0], z=[50.0], pclass=Particle)
    with expectation:
        pset.execute(runtime=runtime, dt=np.timedelta64(10, "s"), pyfunc=DoNothing)


@pytest.mark.parametrize(
    "endtime, expectation",
    [
        (np.datetime64("2000-01-02T00:00:00"), does_not_raise()),
        (5.0, pytest.raises(ValueError)),
        (np.timedelta64(5, "s"), pytest.raises(ValueError)),
        (timedelta(seconds=2), pytest.raises(ValueError)),
        (datetime(2000, 1, 2, 0, 0, 0), pytest.raises(ValueError)),
    ],
)
def test_particleset_endtime_type(fieldset, endtime, expectation):
    pset = ParticleSet(fieldset, lon=[0.2], lat=[5.0], z=[50.0], pclass=Particle)
    with expectation:
        pset.execute(endtime=endtime, dt=np.timedelta64(10, "m"), pyfunc=DoNothing)


@pytest.mark.parametrize(
    "dt", [np.timedelta64(1, "s"), np.timedelta64(1, "ms"), np.timedelta64(10, "ms"), np.timedelta64(1, "ns")]
)
def test_pset_execute_subsecond_dt(fieldset, dt):
    def AddDt(particles, fieldset):  # pragma: no cover
        dt = particles.dt / np.timedelta64(1, "s")
        particles.added_dt += dt

    pclass = Particle.add_variable(Variable("added_dt", dtype=np.float32, initial=0))
    pset = ParticleSet(fieldset, pclass=pclass, lon=0, lat=0)
    pset.update_dt_dtype(dt.dtype)
    pset.execute(AddDt, runtime=dt * 10, dt=dt)
    np.testing.assert_allclose(pset[0].added_dt, 10.0 * dt / np.timedelta64(1, "s"), atol=1e-5)


def test_pset_execute_subsecond_dt_error(fieldset):
    pset = ParticleSet(fieldset, lon=0, lat=0)
    with pytest.raises(ValueError, match="The dtype of dt"):
        pset.execute(DoNothing, runtime=np.timedelta64(10, "ms"), dt=np.timedelta64(1, "ms"))


def test_pset_remove_particle_in_kernel(fieldset):
    npart = 100
    pset = ParticleSet(fieldset, lon=np.linspace(0, 1, npart), lat=np.linspace(1, 0, npart))

    def DeleteKernel(particles, fieldset):  # pragma: no cover
        particles.state = np.where((particles.lon >= 0.4) & (particles.lon <= 0.6), StatusCode.Delete, particles.state)

    pset.execute(pset.Kernel(DeleteKernel), runtime=np.timedelta64(1, "s"), dt=np.timedelta64(1, "s"))
    indices = [i for i in range(npart) if not (40 <= i < 60)]
    assert [p.trajectory for p in pset] == indices
    assert pset[70].trajectory == 90
    assert pset[-1].trajectory == npart - 1
    assert pset.size == 80


@pytest.mark.parametrize("npart", [1, 100])
def test_pset_stop_simulation(fieldset, npart):
    pset = ParticleSet(fieldset, lon=np.zeros(npart), lat=np.zeros(npart), pclass=Particle)

    def Delete(particles, fieldset):  # pragma: no cover
        particles[
            particles.time >= fieldset.time_interval.left + np.timedelta64(4, "s")
        ].state = StatusCode.StopExecution

    pset.execute(Delete, dt=np.timedelta64(1, "s"), runtime=np.timedelta64(21, "s"))
    assert pset[0].time == fieldset.time_interval.left + np.timedelta64(4, "s")


@pytest.mark.parametrize("with_delete", [True, False])
def test_pset_multi_execute(fieldset, with_delete, npart=10, n=5):
    pset = ParticleSet(fieldset, lon=np.linspace(0, 1, npart), lat=np.zeros(npart))

    def AddLat(particles, fieldset):  # pragma: no cover
        particles.dlat += 0.1

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
    [(0, 10, 1), (0, 10, 3), (2, 16, 3), (20, 10, -1), (20, 0, -2), (5, 15, 1)],
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

    def AddLon(particles, fieldset):  # pragma: no cover
        particles.lon += 1

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
    def MoveRight(particles, fieldset):  # pragma: no cover
        particles.dlon += 1
        fieldset.U[particles.time, particles.z, particles.lat, particles.lon, particles]

    def DeleteAllErrorParticles(particles, fieldset):  # pragma: no cover
        particles[particles.state > 20].state = StatusCode.Delete

    pset = ParticleSet(fieldset, lon=[1e5, 2], lat=[0, 0])
    pset.execute([MoveRight, DeleteAllErrorParticles], runtime=np.timedelta64(10, "s"), dt=np.timedelta64(1, "s"))
    assert len(pset) == 0


def test_some_particles_throw_outoftime(fieldset):
    time = [fieldset.time_interval.left + np.timedelta64(t, "D") for t in [0, 350]]
    pset = ParticleSet(fieldset, lon=np.zeros_like(time), lat=np.zeros_like(time), time=time)

    def FieldAccessOutsideTime(particles, fieldset):  # pragma: no cover
        fieldset.U[particles.time + np.timedelta64(400, "D"), particles.z, particles.lat, particles.lon, particles]

    with pytest.raises(OutsideTimeInterval):
        pset.execute(FieldAccessOutsideTime, runtime=np.timedelta64(1, "D"), dt=np.timedelta64(10, "D"))


def test_raise_grid_searching_error(): ...


def test_raise_general_error(): ...


def test_errorinterpolation(fieldset):
    def NaNInterpolator(field, ti, position, tau, t, z, y, x):  # pragma: no cover
        return np.nan * np.zeros_like(x)

    def SampleU(particles, fieldset):  # pragma: no cover
        fieldset.U[particles.time, particles.z, particles.lat, particles.lon, particles]

    fieldset.U.interp_method = NaNInterpolator
    pset = ParticleSet(fieldset, lon=[0, 2], lat=[0, 0])
    with pytest.raises(FieldInterpolationError):
        pset.execute(SampleU, runtime=np.timedelta64(2, "s"), dt=np.timedelta64(1, "s"))


def test_execution_check_stopallexecution(fieldset):
    def addoneLon(particles, fieldset):  # pragma: no cover
        particles.dlon += 1
        particles[particles.lon + particles.dlon >= 10].state = StatusCode.StopAllExecution

    pset = ParticleSet(fieldset, lon=[0, 0], lat=[0, 0])
    pset.execute(addoneLon, runtime=np.timedelta64(20, "s"), dt=np.timedelta64(1, "s"))
    np.testing.assert_allclose(pset.lon, 9)
    np.testing.assert_allclose(pset.time - fieldset.time_interval.left, np.timedelta64(9, "s"))


def test_execution_recover_out_of_bounds(fieldset):
    npart = 2

    def MoveRight(particles, fieldset):  # pragma: no cover
        fieldset.U[particles.time, particles.z, particles.lat, particles.lon + 0.1, particles]
        particles.dlon += 0.1

    def MoveLeft(particles, fieldset):  # pragma: no cover
        inds = np.where(particles.state == StatusCode.ErrorOutOfBounds)
        print(inds, particles.state)
        particles[inds].dlon -= 1.0
        particles[inds].state = StatusCode.Success

    lon = np.linspace(0.05, 6.95, npart)
    lat = np.linspace(1, 0, npart)
    pset = ParticleSet(fieldset, lon=lon, lat=lat)
    pset.execute([MoveRight, MoveLeft], runtime=np.timedelta64(61, "s"), dt=np.timedelta64(1, "s"))
    assert len(pset) == npart
    np.testing.assert_allclose(pset.lon, [6.05, 5.95], rtol=1e-5)
    np.testing.assert_allclose(pset.lat, lat, rtol=1e-5)


@pytest.mark.parametrize(
    "starttime, runtime, dt",
    [(0, 10, 1), (0, 10, 3), (2, 16, 3), (20, 10, -1), (20, 0, -2), (5, 15, 1)],
)
@pytest.mark.parametrize("npart", [1, 10])
def test_execution_runtime(fieldset, starttime, runtime, dt, npart):
    starttime = fieldset.time_interval.left + np.timedelta64(starttime, "s")
    runtime = np.timedelta64(runtime, "s")
    sign_dt = np.sign(dt)
    dt = np.timedelta64(dt, "s")
    pset = ParticleSet(fieldset, time=starttime, lon=np.zeros(npart), lat=np.zeros(npart))
    pset.execute(DoNothing, runtime=runtime, dt=dt)
    assert all([abs(p.time_nextloop - starttime - runtime * sign_dt) < np.timedelta64(1, "ms") for p in pset])


def test_changing_dt_in_kernel(fieldset):
    def KernelCounter(particles, fieldset):  # pragma: no cover
        particles.lon += 1

    pset = ParticleSet(fieldset, lon=np.zeros(1), lat=np.zeros(1))
    pset.execute(KernelCounter, dt=np.timedelta64(2, "s"), runtime=np.timedelta64(5, "s"))
    assert pset.lon == 3
    print(pset.dt)
    assert pset.dt == np.timedelta64(2, "s")


@pytest.mark.parametrize("npart", [1, 100])
def test_execution_fail_python_exception(fieldset, npart):
    pset = ParticleSet(fieldset, lon=np.linspace(0, 1, npart), lat=np.linspace(1, 0, npart))

    def PythonFail(particles, fieldset):  # pragma: no cover
        inds = np.argwhere(particles.time >= fieldset.time_interval.left + np.timedelta64(10, "s"))
        if inds.size > 0:
            raise RuntimeError("Enough is enough!")

    with pytest.raises(RuntimeError):
        pset.execute(PythonFail, runtime=np.timedelta64(20, "s"), dt=np.timedelta64(2, "s"))
    assert len(pset) == npart
    assert all(pset.time == fieldset.time_interval.left + np.timedelta64(10, "s"))


@pytest.mark.parametrize(
    "kernel_names, expected",
    [
        ("Lat1", [0, 1]),
        ("Lat2", [2, 0]),
        pytest.param(
            "Lat1and2",
            [2, 1],
            marks=pytest.mark.xfail(
                reason="Will be fixed alongside GH #2143 . Failing due to https://github.com/OceanParcels/Parcels/pull/2199#issuecomment-3285278876."
            ),
        ),
        ("Lat1then2", [2, 1]),
    ],
)
def test_execution_update_particle_in_kernel_function(fieldset, kernel_names, expected):
    npart = 2

    pset = ParticleSet(fieldset, lon=np.linspace(0, 1, npart), lat=np.zeros(npart))

    def Lat1(particles, fieldset):  # pragma: no cover
        def SetLat1(p):
            p.lat = 1

        SetLat1(particles[(particles.lat == 0) & (particles.lon > 0.5)])

    def Lat2(particles, fieldset):  # pragma: no cover
        def SetLat2(p):
            p.lat = 2

        SetLat2(particles[(particles.lat == 0) & (particles.lon < 0.5)])

    def Lat1and2(particles, fieldset):  # pragma: no cover
        def SetLat1(p):
            p.lat = 1

        def SetLat2(p):
            p.lat = 2

        SetLat1(particles[(particles.lat == 0) & (particles.lon > 0.5)])
        SetLat2(particles[(particles.lat == 0) & (particles.lon < 0.5)])

    if kernel_names == "Lat1":
        kernels = [Lat1]
    elif kernel_names == "Lat2":
        kernels = [Lat2]
    elif kernel_names == "Lat1and2":
        kernels = [Lat1and2]
    elif kernel_names == "Lat1then2":
        kernels = [Lat1, Lat2]

    pset.execute(kernels, runtime=np.timedelta64(2, "s"), dt=np.timedelta64(1, "s"))
    np.testing.assert_allclose(pset.lat, expected, rtol=1e-5)


def test_uxstommelgyre_pset_execute():
    ds = datasets_unstructured["stommel_gyre_delaunay"]
    grid = UxGrid(grid=ds.uxgrid, z=ds.coords["nz"], mesh="spherical")
    U = Field(
        name="U",
        data=ds.U,
        grid=grid,
        interp_method=UXPiecewiseConstantFace,
    )
    V = Field(
        name="V",
        data=ds.V,
        grid=grid,
        interp_method=UXPiecewiseConstantFace,
    )
    P = Field(
        name="P",
        data=ds.p,
        grid=grid,
        interp_method=UXPiecewiseConstantFace,
    )
    UV = VectorField(name="UV", U=U, V=V)
    fieldset = FieldSet([UV, UV.U, UV.V, P])
    pset = ParticleSet(
        fieldset,
        lon=[30.0],
        lat=[5.0],
        z=[50.0],
        time=[np.timedelta64(0, "s")],
        pclass=Particle,
    )
    pset.execute(
        AdvectionEE,
        runtime=np.timedelta64(10, "m"),
        dt=np.timedelta64(60, "s"),
    )
    assert utils.round_and_hash_float_array([p.lon for p in pset]) == 1165396086
    assert utils.round_and_hash_float_array([p.lat for p in pset]) == 1142124776


def test_uxstommelgyre_multiparticle_pset_execute():
    ds = datasets_unstructured["stommel_gyre_delaunay"]
    grid = UxGrid(grid=ds.uxgrid, z=ds.coords["nz"], mesh="spherical")
    U = Field(
        name="U",
        data=ds.U,
        grid=grid,
        interp_method=UXPiecewiseConstantFace,
    )
    V = Field(
        name="V",
        data=ds.V,
        grid=grid,
        interp_method=UXPiecewiseConstantFace,
    )
    W = Field(
        name="W",
        data=ds.W,
        grid=grid,
        interp_method=UXPiecewiseLinearNode,
    )
    P = Field(
        name="P",
        data=ds.p,
        grid=grid,
        interp_method=UXPiecewiseConstantFace,
    )
    UVW = VectorField(name="UVW", U=U, V=V, W=W)
    fieldset = FieldSet([UVW, UVW.U, UVW.V, UVW.W, P])
    pset = ParticleSet(
        fieldset,
        lon=[30.0, 32.0],
        lat=[5.0, 5.0],
        z=[50.0, 50.0],
        time=[np.timedelta64(0, "s")],
        pclass=Particle,
    )
    pset.execute(
        runtime=np.timedelta64(10, "m"),
        dt=np.timedelta64(60, "s"),
        pyfunc=AdvectionRK4_3D,
    )


@pytest.mark.xfail(reason="Output file not implemented yet")
def test_uxstommelgyre_pset_execute_output():
    ds = datasets_unstructured["stommel_gyre_delaunay"]
    grid = UxGrid(grid=ds.uxgrid, z=ds.coords["nz"], mesh="spherical")
    U = Field(
        name="U",
        data=ds.U,
        grid=grid,
        interp_method=UXPiecewiseConstantFace,
    )
    V = Field(
        name="V",
        data=ds.V,
        grid=grid,
        interp_method=UXPiecewiseConstantFace,
    )
    P = Field(
        name="P",
        data=ds.p,
        grid=grid,
        interp_method=UXPiecewiseConstantFace,
    )
    UV = VectorField(name="UV", U=U, V=V)
    fieldset = FieldSet([UV, UV.U, UV.V, P])
    pset = ParticleSet(
        fieldset,
        lon=[30.0],
        lat=[5.0],
        z=[50.0],
        time=[0.0],
        pclass=Particle,
    )
    output_file = ParticleFile(
        name="stommel_uxarray_particles.zarr",  # the file name
        outputdt=np.timedelta64(5, "m"),  # the time step of the outputs
    )
    pset.execute(
        runtime=np.timedelta64(10, "m"), dt=np.timedelta64(60, "s"), pyfunc=AdvectionEE, output_file=output_file
    )

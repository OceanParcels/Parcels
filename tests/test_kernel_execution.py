import os
import sys

import numpy as np
import pytest

import parcels
from parcels import (
    AdvectionRK4,
    FieldOutOfBoundError,
    FieldSet,
    JITParticle,
    ParticleSet,
    ScipyParticle,
    StatusCode,
)
from tests.common_kernels import DeleteParticle, DoNothing, MoveEast, MoveNorth
from tests.utils import create_fieldset_unit_mesh

ptype = {"scipy": ScipyParticle, "jit": JITParticle}


@pytest.fixture
def fieldset_unit_mesh():
    return create_fieldset_unit_mesh()


@pytest.mark.parametrize("mode", ["scipy", "jit"])
@pytest.mark.parametrize("kernel_type", ["update_lon", "update_dlon"])
def test_execution_order(mode, kernel_type):
    fieldset = FieldSet.from_data(
        {"U": [[0, 1], [2, 3]], "V": np.ones((2, 2))}, {"lon": [0, 2], "lat": [0, 2]}, mesh="flat"
    )

    def MoveLon_Update_Lon(particle, fieldset, time):
        particle.lon += 0.2

    def MoveLon_Update_dlon(particle, fieldset, time):
        particle_dlon += 0.2  # noqa

    def SampleP(particle, fieldset, time):
        particle.p = fieldset.U[time, particle.depth, particle.lat, particle.lon]

    SampleParticle = ptype[mode].add_variable("p", dtype=np.float32, initial=0.0)

    MoveLon = MoveLon_Update_dlon if kernel_type == "update_dlon" else MoveLon_Update_Lon

    kernels = [MoveLon, SampleP]
    lons = []
    ps = []
    for dir in [1, -1]:
        pset = ParticleSet(fieldset, pclass=SampleParticle, lon=0, lat=0)
        pset.execute(kernels[::dir], endtime=1, dt=1)
        lons.append(pset.lon)
        ps.append(pset.p)

    if kernel_type == "update_dlon":
        assert np.isclose(lons[0], lons[1])
        assert np.isclose(ps[0], ps[1])
        assert np.allclose(lons[0], 0)
    else:
        assert np.isclose(ps[0] - ps[1], 0.1)
        assert np.allclose(lons[0], 0.2)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
@pytest.mark.parametrize(
    "start, end, substeps, dt",
    [
        (0.0, 10.0, 1, 1.0),
        (0.0, 10.0, 4, 1.0),
        (0.0, 10.0, 1, 3.0),
        (2.0, 16.0, 5, 3.0),
        (20.0, 10.0, 4, -1.0),
        (20.0, -10.0, 7, -2.0),
    ],
)
def test_execution_endtime(fieldset_unit_mesh, mode, start, end, substeps, dt):
    npart = 10
    pset = ParticleSet(
        fieldset_unit_mesh, pclass=ptype[mode], time=start, lon=np.linspace(0, 1, npart), lat=np.linspace(1, 0, npart)
    )
    pset.execute(DoNothing, endtime=end, dt=dt)
    assert np.allclose(pset.time_nextloop, end)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
@pytest.mark.parametrize(
    "start, end, substeps, dt",
    [
        (0.0, 10.0, 1, 1.0),
        (0.0, 10.0, 4, 1.0),
        (0.0, 10.0, 1, 3.0),
        (2.0, 16.0, 5, 3.0),
        (20.0, 10.0, 4, -1.0),
        (20.0, -10.0, 7, -2.0),
    ],
)
def test_execution_runtime(fieldset_unit_mesh, mode, start, end, substeps, dt):
    npart = 10
    pset = ParticleSet(
        fieldset_unit_mesh, pclass=ptype[mode], time=start, lon=np.linspace(0, 1, npart), lat=np.linspace(1, 0, npart)
    )
    t_step = abs(end - start) / substeps
    for _ in range(substeps):
        pset.execute(DoNothing, runtime=t_step, dt=dt)
    assert np.allclose(pset.time_nextloop, end)


@pytest.mark.parametrize("mode", ["scipy"])
def test_execution_fail_python_exception(fieldset_unit_mesh, mode):
    npart = 10

    def PythonFail(particle, fieldset, time):
        if particle.time >= 10.0:
            raise RuntimeError("Enough is enough!")
        else:
            pass

    pset = ParticleSet(
        fieldset_unit_mesh, pclass=ptype[mode], lon=np.linspace(0, 1, npart), lat=np.linspace(1, 0, npart)
    )
    with pytest.raises(RuntimeError):
        pset.execute(PythonFail, endtime=20.0, dt=2.0)
    assert len(pset) == npart
    assert np.isclose(pset.time[0], 10)
    assert np.allclose(pset.time[1:], 0.0)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_execution_fail_out_of_bounds(fieldset_unit_mesh, mode):
    npart = 10

    def MoveRight(particle, fieldset, time):
        tmp1, tmp2 = fieldset.UV[time, particle.depth, particle.lat, particle.lon + 0.1, particle]
        particle_dlon += 0.1  # noqa

    pset = ParticleSet(
        fieldset_unit_mesh, pclass=ptype[mode], lon=np.linspace(0, 1, npart), lat=np.linspace(1, 0, npart)
    )
    with pytest.raises(FieldOutOfBoundError):
        pset.execute(MoveRight, endtime=10.0, dt=1.0)
    assert len(pset) == npart
    assert (pset.lon - 1.0 > -1.0e12).all()


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_execution_recover_out_of_bounds(fieldset_unit_mesh, mode):
    npart = 2

    def MoveRight(particle, fieldset, time):
        tmp1, tmp2 = fieldset.UV[time, particle.depth, particle.lat, particle.lon + 0.1, particle]
        particle_dlon += 0.1  # noqa

    def MoveLeft(particle, fieldset, time):
        if particle.state == StatusCode.ErrorOutOfBounds:
            particle_dlon -= 1.0  # noqa
            particle.state = StatusCode.Success

    lon = np.linspace(0.05, 0.95, npart)
    lat = np.linspace(1, 0, npart)
    pset = ParticleSet(fieldset_unit_mesh, pclass=ptype[mode], lon=lon, lat=lat)
    pset.execute([MoveRight, MoveLeft], endtime=11.0, dt=1.0)
    assert len(pset) == npart
    assert np.allclose(pset.lon, lon, rtol=1e-5)
    assert np.allclose(pset.lat, lat, rtol=1e-5)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_execution_check_all_errors(fieldset_unit_mesh, mode):
    def MoveRight(particle, fieldset, time):
        tmp1, tmp2 = fieldset.UV[time, particle.depth, particle.lat, particle.lon, particle]

    def RecoverAllErrors(particle, fieldset, time):
        if particle.state > 4:
            particle.state = StatusCode.Delete

    pset = ParticleSet(fieldset_unit_mesh, pclass=ptype[mode], lon=10, lat=0)
    pset.execute([MoveRight, RecoverAllErrors], endtime=11.0, dt=1.0)
    assert len(pset) == 0


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_execution_check_stopallexecution(fieldset_unit_mesh, mode):
    def addoneLon(particle, fieldset, time):
        particle_dlon += 1  # noqa

        if particle.lon + particle_dlon >= 10:
            particle.state = StatusCode.StopAllExecution

    pset = ParticleSet(fieldset_unit_mesh, pclass=ptype[mode], lon=[0, 1], lat=[0, 0])
    pset.execute(addoneLon, endtime=20.0, dt=1.0)
    assert pset[0].lon == 9
    assert pset[0].time == 9
    assert pset[1].lon == 1
    assert pset[1].time == 0


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_execution_delete_out_of_bounds(fieldset_unit_mesh, mode):
    npart = 10

    def MoveRight(particle, fieldset, time):
        tmp1, tmp2 = fieldset.UV[time, particle.depth, particle.lat, particle.lon + 0.1, particle]
        particle_dlon += 0.1  # noqa

    lon = np.linspace(0.05, 0.95, npart)
    lat = np.linspace(1, 0, npart)
    pset = ParticleSet(fieldset_unit_mesh, pclass=ptype[mode], lon=lon, lat=lat)
    pset.execute([MoveRight, DeleteParticle], endtime=10.0, dt=1.0)
    assert len(pset) == 0


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_kernel_add_no_new_variables(fieldset_unit_mesh, mode):
    pset = ParticleSet(fieldset_unit_mesh, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute(pset.Kernel(MoveEast) + pset.Kernel(MoveNorth), endtime=2.0, dt=1.0)
    assert np.allclose(pset.lon, 0.6, rtol=1e-5)
    assert np.allclose(pset.lat, 0.6, rtol=1e-5)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_multi_kernel_duplicate_varnames(fieldset_unit_mesh, mode):
    # Testing for merging of two Kernels with the same variable declared
    # Should throw a warning, but go ahead regardless
    def Kernel1(particle, fieldset, time):
        add_lon = 0.1
        particle_dlon += add_lon  # noqa

    def Kernel2(particle, fieldset, time):
        add_lon = -0.3
        particle_dlon += add_lon  # noqa

    pset = ParticleSet(fieldset_unit_mesh, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute([Kernel1, Kernel2], endtime=2.0, dt=1.0)
    assert np.allclose(pset.lon, 0.3, rtol=1e-5)


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_multi_kernel_reuse_varnames(fieldset_unit_mesh, mode):
    # Testing for merging of two Kernels with the same variable declared
    # Should throw a warning, but go ahead regardless
    def MoveEast1(particle, fieldset, time):
        add_lon = 0.2
        particle_dlon += add_lon  # noqa

    def MoveEast2(particle, fieldset, time):
        particle_dlon += add_lon  # noqa

    pset = ParticleSet(fieldset_unit_mesh, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute(pset.Kernel(MoveEast1) + pset.Kernel(MoveEast2), endtime=2.0, dt=1.0)
    assert np.allclose(pset.lon, [0.9], rtol=1e-5)  # should be 0.5 + 0.2 + 0.2 = 0.9


def test_combined_kernel_from_list(fieldset_unit_mesh):
    """
    Test pset.Kernel(List[function])

    Tests that a Kernel can be created from a list functions, or a list of
    mixed functions and kernel objects.
    """

    def MoveEast(particle, fieldset, time):
        particle_dlon += 0.1  # noqa

    def MoveNorth(particle, fieldset, time):
        particle_dlat += 0.1  # noqa

    pset = ParticleSet(fieldset_unit_mesh, pclass=JITParticle, lon=[0.5], lat=[0.5])
    kernels_single = pset.Kernel([AdvectionRK4])
    kernels_functions = pset.Kernel([AdvectionRK4, MoveEast, MoveNorth])

    # Check if the kernels were combined correctly
    assert kernels_single.funcname == "AdvectionRK4"
    assert kernels_functions.funcname == "AdvectionRK4MoveEastMoveNorth"


def test_combined_kernel_from_list_error_checking(fieldset_unit_mesh):
    """
    Test pset.Kernel(List[function])

    Tests that various error cases raise appropriate messages.
    """
    pset = ParticleSet(fieldset_unit_mesh, pclass=JITParticle, lon=[0.5], lat=[0.5])

    # Test that list has to be non-empty
    with pytest.raises(ValueError):
        pset.Kernel([])

    # Test that list has to be all functions
    with pytest.raises(ValueError):
        pset.Kernel([AdvectionRK4, "something else"])

    # Can't mix kernel objects and functions in list
    with pytest.raises(ValueError):
        kernels_mixed = pset.Kernel([pset.Kernel(AdvectionRK4), MoveEast, MoveNorth])
        assert kernels_mixed.funcname == "AdvectionRK4MoveEastMoveNorth"


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_update_kernel_in_script(fieldset_unit_mesh, mode):
    # Testing what happens when kernels are updated during runtime of a script
    # Should throw a warning, but go ahead regardless
    def MoveEast(particle, fieldset, time):
        add_lon = 0.1
        particle_dlon += add_lon  # noqa

    def MoveWest(particle, fieldset, time):
        add_lon = -0.3
        particle_dlon += add_lon  # noqa

    pset = ParticleSet(fieldset_unit_mesh, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute(pset.Kernel(MoveEast), endtime=1.0, dt=1.0)
    pset.execute(pset.Kernel(MoveWest), endtime=3.0, dt=1.0)
    assert np.allclose(pset.lon, 0.3, rtol=1e-5)  # should be 0.5 + 0.1 - 0.3 = 0.3


@pytest.mark.parametrize("delete_cfiles", [True, False])
@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="skipping windows test as windows compiler generates warning"
)
def test_execution_keep_cfiles_and_nocompilation_warnings(fieldset_unit_mesh, delete_cfiles):
    pset = ParticleSet(fieldset_unit_mesh, pclass=JITParticle, lon=[0.0], lat=[0.0])
    pset.execute(AdvectionRK4, delete_cfiles=delete_cfiles, endtime=1.0, dt=1.0)
    cfile = pset.kernel.src_file
    logfile = pset.kernel.log_file
    del pset.kernel
    if delete_cfiles:
        assert not os.path.exists(cfile)
    else:
        assert os.path.exists(cfile)
        with open(logfile) as f:
            assert "warning" not in f.read(), "Compilation WARNING in log file"


def test_compilers():
    from parcels.compilation.codecompiler import (
        CCompiler_SS,
        Clang_parameters,
        MinGW_parameters,
        VS_parameters,
    )

    for param_class in [Clang_parameters, MinGW_parameters, VS_parameters]:
        params = param_class()  # noqa

    print(CCompiler_SS())


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_explicit_ParcelsRandom(fieldset_unit_mesh, mode):
    """Testing `from parcels import ParcelsRandom` in kernel code"""
    from parcels import ParcelsRandom

    def nudge_kernel(particle, fieldset, time):
        dlat = ParcelsRandom.uniform(2, 3)
        particle_dlat += dlat  # noqa

    pset = ParticleSet(fieldset_unit_mesh, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute(nudge_kernel, runtime=2, dt=1)
    assert 2.5 <= pset[0].lat <= 3.5


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_parcels_dot_ParcelsRandom(fieldset_unit_mesh, mode):
    """Testing `parcels.ParcelsRandom` in kernel code"""

    def nudge_kernel(particle, fieldset, time):
        particle_dlat += parcels.ParcelsRandom.uniform(2, 3)  # noqa

    pset = ParticleSet(fieldset_unit_mesh, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute(nudge_kernel, runtime=2, dt=1)
    assert 2.5 <= pset[0].lat <= 3.5


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_parcels_dot_rng(fieldset_unit_mesh, mode):
    """Testing `parcels.rng` in kernel code."""

    def nudge_kernel(particle, fieldset, time):
        dlat = parcels.rng.uniform(2, 3)
        particle_dlat += dlat  # noqa

    pset = ParticleSet(fieldset_unit_mesh, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute(nudge_kernel, runtime=2, dt=1)
    assert 2.5 <= pset[0].lat <= 3.5


@pytest.mark.parametrize("mode", ["scipy", "jit"])
def test_custom_ParcelsRandom_alias(fieldset_unit_mesh, mode):
    """Testing aliasing ParcelsRandom to another name."""
    from parcels import ParcelsRandom as my_custom_name

    def nudge_kernel(particle, fieldset, time):
        particle_dlat += my_custom_name.uniform(2, 3)  # noqa

    pset = ParticleSet(fieldset_unit_mesh, pclass=ptype[mode], lon=[0.5], lat=[0.5])

    try:
        pset.execute(nudge_kernel, runtime=2, dt=1)
    except Exception:
        pass  # This test is expected to fail
    else:
        pytest.fail(
            "Parcels uses function name to determine kernel support. "
            "Aliasing ParcelsRandom to another name is not supported."
        )


def test_outdated_kernel(fieldset_unit_mesh):
    """
    Make sure that if users try using a kernel from pre Parcels 2.0 they get an error.

    Prevents users from copy-pasting old kernels that are no longer supported.
    """
    pset = ParticleSet(fieldset_unit_mesh, pclass=JITParticle, lon=[0.5], lat=[0.5])

    def outdated_kernel(particle, fieldset, time, dt):
        particle.lon += 0.1

    with pytest.raises(ValueError) as e:
        pset.Kernel(outdated_kernel)

    assert "Since Parcels v2.0" in str(e.value)

    with pytest.raises(ValueError) as e:
        pset.execute(outdated_kernel, endtime=1.0, dt=1.0)

    assert "Since Parcels v2.0" in str(e.value)

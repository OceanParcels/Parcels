import sys
from os import path

import numpy as np
import pytest

from parcels import (
    AdvectionRK4,
    FieldOutOfBoundError,
    FieldSet,
    JITParticle,
    ParticleSet,
    ScipyParticle,
    StatusCode,
)

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def DoNothing(particle, fieldset, time):
    pass


def fieldset(xdim=20, ydim=20):
    """Standard unit mesh fieldset."""
    lon = np.linspace(0., 1., xdim, dtype=np.float32)
    lat = np.linspace(0., 1., ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    data = {'U': np.array(U, dtype=np.float32), 'V': np.array(V, dtype=np.float32)}
    dimensions = {'lat': lat, 'lon': lon}
    return FieldSet.from_data(data, dimensions, mesh='flat')


@pytest.fixture(name="fieldset")
def fieldset_fixture(xdim=20, ydim=20):
    return fieldset(xdim=xdim, ydim=ydim)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('kernel_type', ['update_lon', 'update_dlon'])
def test_execution_order(mode, kernel_type):
    fieldset = FieldSet.from_data({'U': [[0, 1], [2, 3]], 'V': np.ones((2, 2))}, {'lon': [0, 2], 'lat': [0, 2]}, mesh='flat')

    def MoveLon_Update_Lon(particle, fieldset, time):
       particle.lon += 0.2  # noqa

    def MoveLon_Update_dlon(particle, fieldset, time):
       particle_dlon += 0.2  # noqa

    def SampleP(particle, fieldset, time):
        particle.p = fieldset.U[time, particle.depth, particle.lat, particle.lon]

    SampleParticle = ptype[mode].add_variable('p', dtype=np.float32, initial=0.)

    MoveLon = MoveLon_Update_dlon if kernel_type == 'update_dlon' else MoveLon_Update_Lon

    kernels = [MoveLon, SampleP]
    lons = []
    ps = []
    for dir in [1, -1]:
        pset = ParticleSet(fieldset, pclass=SampleParticle, lon=0, lat=0)
        pset.execute(kernels[::dir], endtime=1, dt=1)
        lons.append(pset.lon)
        ps.append(pset.p)

    if kernel_type == 'update_dlon':
        assert np.isclose(lons[0], lons[1])
        assert np.isclose(ps[0], ps[1])
        assert np.allclose(lons[0], 0)
    else:
        assert np.isclose(ps[0] - ps[1], 0.1)
        assert np.allclose(lons[0], 0.2)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('start, end, substeps, dt', [
    (0., 10., 1, 1.),
    (0., 10., 4, 1.),
    (0., 10., 1, 3.),
    (2., 16., 5, 3.),
    (20., 10., 4, -1.),
    (20., -10., 7, -2.),
])
def test_execution_endtime(fieldset, mode, start, end, substeps, dt, npart=10):
    pset = ParticleSet(fieldset, pclass=ptype[mode], time=start,
                       lon=np.linspace(0, 1, npart),
                       lat=np.linspace(1, 0, npart))
    pset.execute(DoNothing, endtime=end, dt=dt)
    assert np.allclose(pset.time_nextloop, end)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('start, end, substeps, dt', [
    (0., 10., 1, 1.),
    (0., 10., 4, 1.),
    (0., 10., 1, 3.),
    (2., 16., 5, 3.),
    (20., 10., 4, -1.),
    (20., -10., 7, -2.),
])
def test_execution_runtime(fieldset, mode, start, end, substeps, dt, npart=10):
    pset = ParticleSet(fieldset, pclass=ptype[mode], time=start,
                       lon=np.linspace(0, 1, npart),
                       lat=np.linspace(1, 0, npart))
    t_step = abs(end - start) / substeps
    for _ in range(substeps):
        pset.execute(DoNothing, runtime=t_step, dt=dt)
    assert np.allclose(pset.time_nextloop, end)


@pytest.mark.parametrize('mode', ['scipy'])
def test_execution_fail_python_exception(fieldset, mode, npart=10):
    def PythonFail(particle, fieldset, time):
        if particle.time >= 10.:
            raise RuntimeError("Enough is enough!")
        else:
            pass

    pset = ParticleSet(fieldset, pclass=ptype[mode],
                       lon=np.linspace(0, 1, npart),
                       lat=np.linspace(1, 0, npart))
    error_thrown = False
    try:
        pset.execute(PythonFail, endtime=20., dt=2.)
    except RuntimeError:
        error_thrown = True
    assert error_thrown
    assert len(pset) == npart
    assert np.isclose(pset.time[0], 10)
    assert np.allclose(pset.time[1:], 0.)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_execution_fail_out_of_bounds(fieldset, mode, npart=10):
    def MoveRight(particle, fieldset, time):
        tmp1, tmp2 = fieldset.UV[time, particle.depth, particle.lat, particle.lon + 0.1, particle]  # noqa
        particle_dlon += 0.1  # noqa

    pset = ParticleSet(fieldset, pclass=ptype[mode],
                       lon=np.linspace(0, 1, npart),
                       lat=np.linspace(1, 0, npart))
    error_thrown = False
    try:
        pset.execute(MoveRight, endtime=10., dt=1.)
    except FieldOutOfBoundError:
        error_thrown = True
    assert error_thrown
    assert len(pset) == npart
    assert (pset.lon - 1. > -1.e12).all()


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_execution_recover_out_of_bounds(fieldset, mode, npart=2):
    def MoveRight(particle, fieldset, time):
        tmp1, tmp2 = fieldset.UV[time, particle.depth, particle.lat, particle.lon + 0.1, particle]  # noqa
        particle_dlon += 0.1  # noqa

    def MoveLeft(particle, fieldset, time):
        if particle.state == StatusCode.ErrorOutOfBounds:
            particle_dlon -= 1.  # noqa
            particle.state = StatusCode.Success

    lon = np.linspace(0.05, 0.95, npart)
    lat = np.linspace(1, 0, npart)
    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=lon, lat=lat)
    pset.execute([MoveRight, MoveLeft], endtime=11., dt=1.)
    assert len(pset) == npart
    assert np.allclose(pset.lon, lon, rtol=1e-5)
    assert np.allclose(pset.lat, lat, rtol=1e-5)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_execution_check_all_errors(fieldset, mode):
    def MoveRight(particle, fieldset, time):
        tmp1, tmp2 = fieldset.UV[time, particle.depth, particle.lat, particle.lon, particle]  # noqa

    def RecoverAllErrors(particle, fieldset, time):
        if particle.state > 4:
            particle.state = StatusCode.Delete

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=10, lat=0)
    pset.execute([MoveRight, RecoverAllErrors], endtime=11., dt=1.)
    assert len(pset) == 0


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_execution_check_stopallexecution(fieldset, mode):

    def addoneLon(particle, fieldset, time):
        particle_dlon += 1  # noqa

        if particle.lon + particle_dlon >= 10:
            particle.state = StatusCode.StopAllExecution

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0, 1], lat=[0, 0])
    pset.execute(addoneLon, endtime=20., dt=1.)
    assert pset[0].lon == 9
    assert pset[0].time == 9
    assert pset[1].lon == 1
    assert pset[1].time == 0


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_execution_delete_out_of_bounds(fieldset, mode, npart=10):
    def MoveRight(particle, fieldset, time):
        tmp1, tmp2 = fieldset.UV[time, particle.depth, particle.lat, particle.lon + 0.1, particle]  # noqa
        particle_dlon += 0.1  # noqa

    def DeleteMe(particle, fieldset, time):
        if particle.state == StatusCode.ErrorOutOfBounds:
            particle.delete()

    lon = np.linspace(0.05, 0.95, npart)
    lat = np.linspace(1, 0, npart)
    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=lon, lat=lat)
    pset.execute([MoveRight, DeleteMe], endtime=10., dt=1.)
    assert len(pset) == 0


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_kernel_add_no_new_variables(fieldset, mode):
    def MoveEast(particle, fieldset, time):
        particle_dlon += 0.1  # noqa

    def MoveNorth(particle, fieldset, time):
        particle_dlat += 0.1  # noqa

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute(pset.Kernel(MoveEast) + pset.Kernel(MoveNorth),
                 endtime=2., dt=1.)
    assert np.allclose(pset.lon, 0.6, rtol=1e-5)
    assert np.allclose(pset.lat, 0.6, rtol=1e-5)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_multi_kernel_duplicate_varnames(fieldset, mode):
    # Testing for merging of two Kernels with the same variable declared
    # Should throw a warning, but go ahead regardless
    def MoveEast(particle, fieldset, time):
        add_lon = 0.1
        particle_dlon += add_lon  # noqa

    def MoveWest(particle, fieldset, time):
        add_lon = -0.3
        particle_dlon += add_lon  # noqa

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute([MoveEast, MoveWest], endtime=2., dt=1.)
    assert np.allclose(pset.lon, 0.3, rtol=1e-5)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_multi_kernel_reuse_varnames(fieldset, mode):
    # Testing for merging of two Kernels with the same variable declared
    # Should throw a warning, but go ahead regardless
    def MoveEast1(particle, fieldset, time):
        add_lon = 0.2
        particle_dlon += add_lon  # noqa

    def MoveEast2(particle, fieldset, time):
        particle_dlon += add_lon  # noqa

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute(pset.Kernel(MoveEast1) + pset.Kernel(MoveEast2),
                 endtime=2., dt=1.)
    assert np.allclose(pset.lon, [0.9], rtol=1e-5)  # should be 0.5 + 0.2 + 0.2 = 0.9


def test_combined_kernel_from_list(fieldset):
    """
    Test pset.Kernel(List[function])

    Tests that a Kernel can be created from a list functions, or a list of
    mixed functions and kernel objects.
    """
    def MoveEast(particle, fieldset, time):
        particle_dlon += 0.1  # noqa

    def MoveNorth(particle, fieldset, time):
        particle_dlat += 0.1  # noqa

    pset = ParticleSet(fieldset, pclass=JITParticle, lon=[0.5], lat=[0.5])
    kernels_single = pset.Kernel([AdvectionRK4])
    kernels_functions = pset.Kernel([AdvectionRK4, MoveEast, MoveNorth])

    # Check if the kernels were combined correctly
    assert kernels_single.funcname == "AdvectionRK4"
    assert kernels_functions.funcname == "AdvectionRK4MoveEastMoveNorth"


def test_combined_kernel_from_list_error_checking(fieldset):
    """
    Test pset.Kernel(List[function])

    Tests that various error cases raise appropriate messages.
    """
    def MoveEast(particle, fieldset, time):
        particle_dlon += 0.1  # noqa

    def MoveNorth(particle, fieldset, time):
        particle_dlat += 0.1  # noqa

    pset = ParticleSet(fieldset, pclass=JITParticle, lon=[0.5], lat=[0.5])

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


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_update_kernel_in_script(fieldset, mode):
    # Testing what happens when kernels are updated during runtime of a script
    # Should throw a warning, but go ahead regardless
    def MoveEast(particle, fieldset, time):
        add_lon = 0.1
        particle_dlon += add_lon  # noqa

    def MoveWest(particle, fieldset, time):
        add_lon = -0.3
        particle_dlon += add_lon  # noqa

    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0.5], lat=[0.5])
    pset.execute(pset.Kernel(MoveEast), endtime=1., dt=1.)
    pset.execute(pset.Kernel(MoveWest), endtime=3., dt=1.)
    assert np.allclose(pset.lon, 0.3, rtol=1e-5)  # should be 0.5 + 0.1 - 0.3 = 0.3


@pytest.mark.parametrize('delete_cfiles', [True, False])
@pytest.mark.skipif(sys.platform.startswith("win"), reason="skipping windows test as windows compiler generates warning")
def test_execution_keep_cfiles_and_nocompilation_warnings(fieldset, delete_cfiles):
    pset = ParticleSet(fieldset, pclass=JITParticle, lon=[0.], lat=[0.])
    pset.execute(pset.Kernel(AdvectionRK4, delete_cfiles=delete_cfiles), endtime=1., dt=1.)
    cfile = pset.kernel.src_file
    logfile = pset.kernel.log_file
    del pset.kernel
    if delete_cfiles:
        assert not path.exists(cfile)
    else:
        assert path.exists(cfile)
        with open(logfile) as f:
            assert 'warning' not in f.read(), 'Compilation WARNING in log file'


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

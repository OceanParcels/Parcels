from operator import attrgetter

import numpy as np
import pytest

from parcels import (
    AdvectionRK4,
    FieldSet,
    JITParticle,
    ParticleSet,
    ScipyParticle,
    Variable,
)

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def fieldset(xdim=100, ydim=100):
    data = {'U': np.zeros((ydim, xdim), dtype=np.float32),
            'V': np.zeros((ydim, xdim), dtype=np.float32)}
    dimensions = {'lon': np.linspace(0, 1, xdim, dtype=np.float32),
                  'lat': np.linspace(0, 1, ydim, dtype=np.float32)}
    return FieldSet.from_data(data, dimensions, mesh='flat')


@pytest.fixture(name="fieldset")
def fieldset_fixture(xdim=100, ydim=100):
    return fieldset(xdim=xdim, ydim=ydim)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_print(fieldset, mode):
    TestParticle = ptype[mode].add_variable('p', to_write=True)
    pset = ParticleSet(fieldset, pclass=TestParticle, lon=[0, 1], lat=[0, 1])
    print(pset)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_variable_init(fieldset, mode, npart=10):
    """Test that checks correct initialisation of custom variables."""
    extra_vars = [Variable('p_float', dtype=np.float32, initial=10.),
                  Variable('p_double', dtype=np.float64, initial=11.)]
    TestParticle = ptype[mode].add_variables(extra_vars)
    TestParticle = TestParticle.add_variable('p_int', np.int32, initial=12.)
    pset = ParticleSet(fieldset, pclass=TestParticle,
                       lon=np.linspace(0, 1, npart),
                       lat=np.linspace(1, 0, npart))

    def addOne(particle, fieldset, time):
        particle.p_float += 1.
        particle.p_double += 1.
        particle.p_int += 1
    pset.execute(pset.Kernel(AdvectionRK4)+addOne, runtime=1., dt=1.)
    assert np.allclose([p.p_float for p in pset], 11., rtol=1e-12)
    assert np.allclose([p.p_double for p in pset], 12., rtol=1e-12)
    assert np.allclose([p.p_int for p in pset], 13, rtol=1e-12)


@pytest.mark.parametrize('mode', ['jit'])
@pytest.mark.parametrize('type', ['np.int8', 'mp.float', 'np.int16'])
def test_variable_unsupported_dtypes(fieldset, mode, type):
    """Test that checks errors thrown for unsupported dtypes in JIT mode."""
    TestParticle = ptype[mode].add_variable('p', dtype=type, initial=10.)
    error_thrown = False
    try:
        ParticleSet(fieldset, pclass=TestParticle, lon=[0], lat=[0])
    except (RuntimeError, TypeError):
        error_thrown = True
    assert error_thrown


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_variable_special_names(fieldset, mode):
    """Test that checks errors thrown for special names."""
    for vars in ['z', 'lon']:
        TestParticle = ptype[mode].add_variable(vars, dtype=np.float32, initial=10.)
        error_thrown = False
        try:
            ParticleSet(fieldset, pclass=TestParticle, lon=[0], lat=[0])
        except AttributeError:
            error_thrown = True
        assert error_thrown


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('coord_type', [np.float32, np.float64])
def test_variable_init_relative(fieldset, mode, coord_type, npart=10):
    """Test that checks relative initialisation of custom variables."""
    lonlat_type = np.float64 if coord_type == 'double' else np.float32

    TestParticle = ptype[mode].add_variables([
        Variable('p_base', dtype=lonlat_type, initial=10.),
        Variable('p_relative', dtype=lonlat_type, initial=attrgetter('p_base')),
        Variable('p_lon', dtype=lonlat_type, initial=attrgetter('lon')),
        Variable('p_lat', dtype=lonlat_type, initial=attrgetter('lat'))])

    lon = np.linspace(0, 1, npart, dtype=lonlat_type)
    lat = np.linspace(1, 0, npart, dtype=lonlat_type)
    pset = ParticleSet(fieldset, pclass=TestParticle, lon=lon, lat=lat, lonlatdepth_dtype=coord_type)
    # Adjust base variable to test for aliasing effects
    for p in pset:
        p.p_base += 3.
    assert np.allclose([p.p_base for p in pset], 13., rtol=1e-12)
    assert np.allclose([p.p_relative for p in pset], 10., rtol=1e-12)
    assert np.allclose([p.p_lon for p in pset], lon, rtol=1e-12)
    assert np.allclose([p.p_lat for p in pset], lat, rtol=1e-12)

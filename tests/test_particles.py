from parcels import Grid, ScipyParticle, JITParticle, Variable
import numpy as np
import pytest
from operator import attrgetter


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


@pytest.fixture
def grid(xdim=100, ydim=100):
    U = np.zeros((xdim, ydim), dtype=np.float32)
    V = np.zeros((xdim, ydim), dtype=np.float32)
    lon = np.linspace(0, 1, xdim, dtype=np.float32)
    lat = np.linspace(0, 1, ydim, dtype=np.float32)
    return Grid.from_data(U, lon, lat, V, lon, lat, mesh='flat')


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_variable_init(grid, mode, npart=10):
    """Test that checks correct initialisation of custom variables"""
    class TestParticle(ptype[mode]):
        p_float = Variable('p_float', dtype=np.float32, default=10.)
        p_double = Variable('p_double', dtype=np.float64, default=11.)
        p_int = Variable('p_int', dtype=np.int32, default=12.)
    pset = grid.ParticleSet(npart, pclass=TestParticle,
                            lon=np.linspace(0, 1, npart, dtype=np.float32),
                            lat=np.linspace(1, 0, npart, dtype=np.float32))
    assert np.array([isinstance(p.p_float, np.float32) for p in pset]).all()
    assert np.allclose([p.p_float for p in pset], 10., rtol=1e-12)
    assert np.array([isinstance(p.p_double, np.float64) for p in pset]).all()
    assert np.allclose([p.p_double for p in pset], 11., rtol=1e-12)
    assert np.array([isinstance(p.p_int, np.int32) for p in pset]).all()
    assert np.allclose([p.p_int for p in pset], 12, rtol=1e-12)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_variable_init_relative(grid, mode, npart=10):
    """Test that checks relative initialisation of custom variables"""
    class TestParticle(ptype[mode]):
        p_base = Variable('p_base', dtype=np.float32, default=10.)
        p_relative = Variable('p_relative', dtype=np.float32,
                              default=attrgetter('p_base'))
        p_offset = Variable('p_offset', dtype=np.float32,
                            default=attrgetter('p_base'))

        def __init__(self, *args, **kwargs):
            super(TestParticle, self).__init__(*args, **kwargs)
            self.p_offset += 2.
    pset = grid.ParticleSet(npart, pclass=TestParticle,
                            lon=np.linspace(0, 1, npart, dtype=np.float32),
                            lat=np.linspace(1, 0, npart, dtype=np.float32))
    # Adjust base variable to test for aliasing effects
    for p in pset:
        p.p_base += 3.
    assert np.allclose([p.p_base for p in pset], 13., rtol=1e-12)
    assert np.allclose([p.p_relative for p in pset], 10., rtol=1e-12)
    assert np.allclose([p.p_offset for p in pset], 12., rtol=1e-12)

from parcels import Grid, Particle, JITParticle, Kernel
from parcels import random as parcels_random
import numpy as np
import pytest
import random as py_random


ptype = {'scipy': Particle, 'jit': JITParticle}


def expr_kernel(name, pset, expr):
    pycode = """def %s(particle, grid, time, dt):
    particle.p = %s""" % (name, expr)
    return Kernel(pset.grid, pset.ptype, pyfunc=None,
                  funccode=pycode, funcname=name,
                  funcvars=['particle'])


@pytest.fixture
def grid(xdim=20, ydim=20):
    """ Standard unit mesh grid """
    lon = np.linspace(0., 1., xdim, dtype=np.float32)
    lat = np.linspace(0., 1., ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    return Grid.from_data(np.array(U, dtype=np.float32), lon, lat,
                          np.array(V, dtype=np.float32), lon, lat,
                          mesh='flat')


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('name, expr, result', [
    ('Add', '2 + 5', 7),
    ('Sub', '6 - 2', 4),
    ('Mul', '3 * 5', 15),
    ('Div', '24 / 4', 6),
])
def test_expression_int(grid, mode, name, expr, result, npart=10):
    """ Test basic arithmetic expressions """
    class TestParticle(ptype[mode]):
        user_vars = {'p': np.int32}
    pset = grid.ParticleSet(npart, pclass=TestParticle,
                            lon=np.linspace(0., 1., npart, dtype=np.float32),
                            lat=np.zeros(npart, dtype=np.float32) + 0.5)
    pset.execute(expr_kernel('Test%s' % name, pset, expr), endtime=1., dt=1.)
    assert(np.array([result == particle.p for particle in pset]).all())


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('name, expr, result', [
    ('Add', '2. + 5.', 7),
    ('Sub', '6. - 2.', 4),
    ('Mul', '3. * 5.', 15),
    ('Div', '24. / 4.', 6),
])
def test_expression_float(grid, mode, name, expr, result, npart=10):
    """ Test basic arithmetic expressions """
    class TestParticle(ptype[mode]):
        user_vars = {'p': np.int32}
    pset = grid.ParticleSet(npart, pclass=TestParticle,
                            lon=np.linspace(0., 1., npart, dtype=np.float32),
                            lat=np.zeros(npart, dtype=np.float32) + 0.5)
    pset.execute(expr_kernel('Test%s' % name, pset, expr), endtime=1., dt=1.)
    assert(np.array([result == particle.p for particle in pset]).all())


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('name, expr, result', [
    ('True', 'True', True),
    ('False', 'False', False),
    ('And', 'True and False', False),
    ('Or', 'True or False', True),
    ('Equal', '5 == 5', True),
    ('Lesser', '5 < 3', False),
    ('LesserEq', '3 <= 5', True),
    ('Greater', '4 > 2', True),
    ('GreaterEq', '2 >= 4', False),
])
def test_expression_bool(grid, mode, name, expr, result, npart=10):
    """ Test basic arithmetic expressions """
    class TestParticle(ptype[mode]):
        user_vars = {'p': np.int32}
    pset = grid.ParticleSet(npart, pclass=TestParticle,
                            lon=np.linspace(0., 1., npart, dtype=np.float32),
                            lat=np.zeros(npart, dtype=np.float32) + 0.5)
    pset.execute(expr_kernel('Test%s' % name, pset, expr), endtime=1., dt=1.)
    if mode == 'jit':
        assert(np.array([result == (particle.p == 1) for particle in pset]).all())
    else:
        assert(np.array([result == particle.p for particle in pset]).all())


def random_series(npart, rngfunc, rngargs, mode):
    random = parcels_random if mode == 'jit' else py_random
    random.seed(1234)
    func = getattr(random, rngfunc)
    series = [func(*rngargs) for _ in range(npart)]
    random.seed(1234)  # Reset the RNG seed
    return series


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('rngfunc, rngargs', [
    ('random', []),
    ('uniform', [0., 20.]),
    ('randint', [0, 20]),
])
def test_random_float(grid, mode, rngfunc, rngargs, npart=10):
    """ Test basic random number generation """
    class TestParticle(ptype[mode]):
        user_vars = {'p': np.int32 if rngfunc == 'randint' else np.float32}
    pset = grid.ParticleSet(npart, pclass=TestParticle,
                            lon=np.linspace(0., 1., npart, dtype=np.float32),
                            lat=np.zeros(npart, dtype=np.float32) + 0.5)
    series = random_series(npart, rngfunc, rngargs, mode)
    kernel = expr_kernel('TestRandom_%s' % rngfunc, pset,
                         'random.%s(%s)' % (rngfunc, ', '.join([str(a) for a in rngargs])))
    pset.execute(kernel, endtime=1., dt=1.)
    assert np.allclose(np.array([p.p for p in pset]), series, rtol=1e-12)

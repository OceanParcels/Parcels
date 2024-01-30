import random as py_random
from os import path

import numpy as np
import pytest

from parcels import (
    FieldSet,
    JITParticle,
    Kernel,
    ParcelsRandom,
    ParticleSet,
    ScipyParticle,
    Variable,
)
from parcels.application_kernels.EOSseawaterproperties import (
    PressureFromLatDepth,
    PtempFromTemp,
    TempFromPtemp,
    UNESCODensity,
)
from parcels.application_kernels.TEOSseawaterdensity import PolyTEOS10_bsq

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def expr_kernel(name, pset, expr):
    pycode = (f"def {name}(particle, fieldset, time):\n"
              f"    particle.p = {expr}")
    return Kernel(pset.fieldset, pset.particledata.ptype, pyfunc=None,
                  funccode=pycode, funcname=name, funcvars=['particle'])


def fieldset(xdim=20, ydim=20):
    """Standard unit mesh fieldset."""
    lon = np.linspace(0., 1., xdim, dtype=np.float32)
    lat = np.linspace(0., 1., ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    data = {'U': np.array(U, dtype=np.float32), 'V': np.array(V, dtype=np.float32)}
    dimensions = {'lat': lat, 'lon': lon}
    return FieldSet.from_data(data, dimensions, mesh='flat', transpose=True)


@pytest.fixture(name="fieldset")
def fieldset_fixture(xdim=20, ydim=20):
    return fieldset(xdim=xdim, ydim=ydim)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('name, expr, result', [
    ('Add', '2 + 5', 7),
    ('Sub', '6 - 2', 4),
    ('Mul', '3 * 5', 15),
    ('Div', '24 / 4', 6),
])
def test_expression_int(mode, name, expr, result, npart=10):
    """Test basic arithmetic expressions."""
    TestParticle = ptype[mode].add_variable('p', dtype=np.float32, initial=0)
    pset = ParticleSet(fieldset(), pclass=TestParticle,
                       lon=np.linspace(0., 1., npart),
                       lat=np.zeros(npart) + 0.5)
    pset.execute(expr_kernel(f'Test{name}', pset, expr), endtime=1., dt=1.)
    assert np.all([p.p == result for p in pset])


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('name, expr, result', [
    ('Add', '2. + 5.', 7),
    ('Sub', '6. - 2.', 4),
    ('Mul', '3. * 5.', 15),
    ('Div', '24. / 4.', 6),
    ('Pow', '2 ** 3', 8),
])
def test_expression_float(mode, name, expr, result, npart=10):
    """Test basic arithmetic expressions."""
    TestParticle = ptype[mode].add_variable('p', dtype=np.float32, initial=0)
    pset = ParticleSet(fieldset(), pclass=TestParticle,
                       lon=np.linspace(0., 1., npart),
                       lat=np.zeros(npart) + 0.5)
    pset.execute(expr_kernel(f'Test{name}', pset, expr), endtime=1., dt=1.)
    assert np.all([p.p == result for p in pset])


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('name, expr, result', [
    ('True', 'True', True),
    ('False', 'False', False),
    ('And', 'True and False', False),
    ('Or', 'True or False', True),
    ('Equal', '5 == 5', True),
    ('NotEqual', '3 != 4', True),
    ('Lesser', '5 < 3', False),
    ('LesserEq', '3 <= 5', True),
    ('Greater', '4 > 2', True),
    ('GreaterEq', '2 >= 4', False),
    ('CheckNaN', 'math.nan != math.nan', True),
])
def test_expression_bool(mode, name, expr, result, npart=10):
    """Test basic arithmetic expressions."""
    TestParticle = ptype[mode].add_variable('p', dtype=np.float32, initial=0)
    pset = ParticleSet(fieldset(), pclass=TestParticle,
                       lon=np.linspace(0., 1., npart),
                       lat=np.zeros(npart) + 0.5)
    pset.execute(expr_kernel(f'Test{name}', pset, expr), endtime=1., dt=1.)
    if mode == 'jit':
        assert np.all(result == (pset.p == 1))
    else:
        assert np.all(result == pset.p)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_while_if_break(mode):
    """Test while, if and break commands."""
    TestParticle = ptype[mode].add_variable('p', dtype=np.float32, initial=0)
    pset = ParticleSet(fieldset(), pclass=TestParticle, lon=[0], lat=[0])

    def kernel(particle, fieldset, time):
        while particle.p < 30:
            if particle.p > 9:
                break
            particle.p += 1
        if particle.p > 5:
            particle.p *= 2.
    pset.execute(kernel, endtime=1., dt=1.)
    assert np.allclose(pset.p, 20., rtol=1e-12)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_nested_if(mode):
    """Test nested if commands."""
    TestParticle = ptype[mode].add_variables([
        Variable('p0', dtype=np.int32, initial=0),
        Variable('p1', dtype=np.int32, initial=1)])
    pset = ParticleSet(fieldset(), pclass=TestParticle, lon=0, lat=0)

    def kernel(particle, fieldset, time):
        if particle.p1 >= particle.p0:
            var = particle.p0
            if var + 1 < particle.p1:
                particle.p1 = -1

    pset.execute(kernel, endtime=10, dt=1.)
    assert np.allclose([pset.p0[0], pset.p1[0]], [0, 1])


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_pass(mode):
    """Test pass commands."""
    TestParticle = ptype[mode].add_variable('p', dtype=np.float32, initial=0)
    pset = ParticleSet(fieldset(), pclass=TestParticle, lon=0, lat=0)

    def kernel(particle, fieldset, time):
        particle.p = -1
        pass

    pset.execute(kernel, endtime=10, dt=1.)
    assert np.allclose(pset[0].p, -1)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_dt_as_variable_in_kernel(mode):
    pset = ParticleSet(fieldset(), pclass=ptype[mode], lon=0, lat=0)

    def kernel(particle, fieldset, time):
        dt = 1.  # noqa

    pset.execute(kernel, endtime=10, dt=1.)


def test_parcels_tmpvar_in_kernel():
    """Tests for error thrown if variable with 'tmp' defined in custom kernel."""
    error_thrown = False
    pset = ParticleSet(fieldset(), pclass=JITParticle, lon=0, lat=0)

    def kernel_tmpvar(particle, fieldset, time):
        parcels_tmpvar0 = 0  # noqa

    def kernel_pnum(particle, fieldset, time):
        pnum = 0  # noqa

    for kernel in [kernel_tmpvar, kernel_pnum]:
        try:
            pset.execute(kernel, endtime=1, dt=1.)
        except NotImplementedError:
            error_thrown = True
        assert error_thrown


def test_abs():
    """Tests for error thrown if using abs in kernel."""
    error_thrown = False
    pset = ParticleSet(fieldset(), pclass=JITParticle, lon=0, lat=0)

    def kernel_abs(particle, fieldset, time):
        particle.lon = abs(3.1)  # noqa

    try:
        pset.execute(kernel_abs, endtime=1, dt=1.)
    except NotImplementedError:
        error_thrown = True
    assert error_thrown


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_if_withfield(fieldset, mode):
    """Test combination of if and Field sampling commands."""
    TestParticle = ptype[mode].add_variable('p', dtype=np.float32, initial=0)
    pset = ParticleSet(fieldset, pclass=TestParticle, lon=[0], lat=[0])

    def kernel(particle, fieldset, time):
        u, v = fieldset.UV[time, 0, 0, 1.]
        particle.p = 0
        if fieldset.U[time, 0, 0, 1.] == u:
            particle.p += 1
        if fieldset.U[time, 0, 0, 1.] == fieldset.U[time, 0, 0, 1.]:
            particle.p += 1
        if True:
            particle.p += 1
        if fieldset.U[time, 0, 0, 1.] == u and 1 == 1:
            particle.p += 1
        if fieldset.U[time, 0, 0, 1.] == fieldset.U[time, 0, 0, 1.] and fieldset.U[time, 0, 0, 1.] == fieldset.U[time, 0, 0, 1.]:
            particle.p += 1
        if fieldset.U[time, 0, 0, 1.] == u:
            particle.p += 1
        else:
            particle.p += 1000
        if fieldset.U[time, 0, 0, 1.] == 3:
            particle.p += 1000
        else:
            particle.p += 1

    pset.execute(kernel, endtime=1., dt=1.)
    assert np.allclose(pset.p, 7., rtol=1e-12)


@pytest.mark.parametrize('mode', ['scipy'])
def test_print(fieldset, mode, capfd):
    """Test print statements."""
    TestParticle = ptype[mode].add_variable('p', dtype=np.float32, initial=0)
    pset = ParticleSet(fieldset, pclass=TestParticle, lon=[0.5], lat=[0.5])

    def kernel(particle, fieldset, time):
        particle.p = 1e-3
        tmp = 5
        print("%d %f %f" % (particle.id, particle.p, tmp))
    pset.execute(kernel, endtime=1., dt=1., verbose_progress=False)
    out, err = capfd.readouterr()
    lst = out.split(' ')
    tol = 1e-8
    assert abs(float(lst[0]) - pset.id[0]) < tol and abs(float(lst[1]) - pset.p[0]) < tol and abs(float(lst[2]) - 5) < tol

    def kernel2(particle, fieldset, time):
        tmp = 3
        print("%f" % (tmp))
    pset.execute(kernel2, endtime=2., dt=1., verbose_progress=False)
    out, err = capfd.readouterr()
    lst = out.split(' ')
    assert abs(float(lst[0]) - 3) < tol


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_fieldset_access(fieldset, mode):
    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=0, lat=0)

    def kernel(particle, fieldset, time):
        particle.lon = fieldset.U.grid.lon[2]

    error_thrown = False
    try:
        pset.execute(kernel, endtime=1, dt=1.)
    except NotImplementedError:
        error_thrown = True
    if mode == 'jit':
        assert error_thrown
    else:
        assert pset.lon[0] == fieldset.U.grid.lon[2]


def random_series(npart, rngfunc, rngargs, mode):
    random = ParcelsRandom if mode == 'jit' else py_random
    random.seed(1234)
    func = getattr(random, rngfunc)
    series = [func(*rngargs) for _ in range(npart)]
    random.seed(1234)  # Reset the RNG seed
    del random
    return series


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('rngfunc, rngargs', [
    ('random', []),
    ('uniform', [0., 20.]),
    ('randint', [0, 20]),
])
def test_random_float(mode, rngfunc, rngargs, npart=10):
    """Test basic random number generation."""
    TestParticle = ptype[mode].add_variable('p', dtype=np.float32, initial=0)
    pset = ParticleSet(fieldset(), pclass=TestParticle,
                       lon=np.linspace(0., 1., npart),
                       lat=np.zeros(npart) + 0.5)
    series = random_series(npart, rngfunc, rngargs, mode)
    rnglib = 'ParcelsRandom' if mode == 'jit' else 'random'
    kernel = expr_kernel(f"TestRandom_{rngfunc}", pset,
                         f"{rnglib}.{rngfunc}({', '.join([str(a) for a in rngargs])})")
    pset.execute(kernel, endtime=1., dt=1.)
    assert np.allclose(pset.p, series, atol=1e-9)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('concat', [False, True])
def test_random_kernel_concat(fieldset, mode, concat):
    TestParticle = ptype[mode].add_variable('p', dtype=np.float32, initial=0)
    pset = ParticleSet(fieldset, pclass=TestParticle, lon=0, lat=0)

    def RandomKernel(particle, fieldset, time):
        particle.p += ParcelsRandom.uniform(0, 1)

    def AddOne(particle, fieldset, time):
        particle.p += 1.

    kernels = [RandomKernel, AddOne] if concat else RandomKernel
    pset.execute(kernels, runtime=1)
    assert pset.p > 1 if concat else pset.p < 1


@pytest.mark.parametrize('mode', ['jit', pytest.param('scipy', marks=pytest.mark.xfail(reason="c_kernels don't work in scipy mode"))])
@pytest.mark.parametrize('c_inc', ['str', 'file'])
def test_c_kernel(fieldset, mode, c_inc):
    coord_type = np.float32 if c_inc == 'str' else np.float64
    pset = ParticleSet(fieldset, pclass=ptype[mode], lon=[0.5], lat=[0],
                       lonlatdepth_dtype=coord_type)

    def func(U, lon, dt):
        u = U.data[0, 2, 1]
        return lon + u * dt

    if c_inc == 'str':
        c_include = """
                 static inline StatusCode func(CField *f, float *lon, double *dt)
                 {
                   float data2D[2][2][2];
                   StatusCode status = getCell2D(f, 1, 2, 0, data2D, 1); CHECKSTATUS(status);
                   float u = data2D[0][0][0];
                   *lon += u * *dt;
                   return SUCCESS;
                 }
                 """
    else:
        c_include = path.join(path.dirname(__file__), 'customed_header.h')

    def ckernel(particle, fieldset, time):
        func('parcels_customed_Cfunc_pointer_args', fieldset.U, particle.lon, particle.dt)

    def pykernel(particle, fieldset, time):
        particle_dlon = func(fieldset.U, particle.lon, particle.dt)  # noqa

    if mode == 'scipy':
        kernel = pset.Kernel(pykernel)
    else:
        kernel = pset.Kernel(ckernel, c_include=c_include)
    pset.execute(kernel, endtime=3., dt=3.)
    assert np.allclose(pset.lon[0], 0.81578948)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_dt_modif_by_kernel(mode):
    TestParticle = ptype[mode].add_variable('age', dtype=np.float32, initial=0)
    pset = ParticleSet(fieldset(), pclass=TestParticle, lon=[0.5], lat=[0])

    def modif_dt(particle, fieldset, time):
        particle.age += particle.dt
        particle.dt = 2

    endtime = 4
    pset.execute(modif_dt, endtime=endtime+1, dt=1.)
    assert np.isclose(pset.time[0], endtime)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('dt', [1e-2, 1e-5, 1e-6])
def test_small_dt(mode, dt, npart=10):
    pset = ParticleSet(fieldset(), pclass=ptype[mode], lon=np.zeros(npart),
                       lat=np.zeros(npart), time=np.arange(0, npart)*dt*10)

    def DoNothing(particle, fieldset, time):
        pass

    raisederror = False
    try:
        pset.execute(DoNothing, dt=dt, runtime=dt*101)
    except ValueError:
        raisederror = True
    if dt < 1e-5:
        assert raisederror
    else:
        assert np.allclose([p.time for p in pset], dt*100)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_TEOSdensity_kernels(mode):

    def generate_fieldset(xdim=2, ydim=2, zdim=2, tdim=1):
        lon = np.linspace(0., 10., xdim, dtype=np.float32)
        lat = np.linspace(0., 10., ydim, dtype=np.float32)
        depth = np.linspace(0, 2000, zdim, dtype=np.float32)
        time = np.zeros(tdim, dtype=np.float64)
        U = np.ones((tdim, zdim, ydim, xdim))
        V = np.ones((tdim, zdim, ydim, xdim))
        abs_salinity = 30 * np.ones((tdim, zdim, ydim, xdim))
        cons_temperature = 10 * np.ones((tdim, zdim, ydim, xdim))
        dimensions = {'lat': lat, 'lon': lon, 'depth': depth, 'time': time}
        data = {'U': np.array(U, dtype=np.float32), 'V': np.array(V, dtype=np.float32),
                'abs_salinity': np.array(abs_salinity, dtype=np.float32),
                'cons_temperature': np.array(cons_temperature, dtype=np.float32)}
        return (data, dimensions)

    data, dimensions = generate_fieldset()
    fieldset = FieldSet.from_data(data, dimensions)

    DensParticle = ptype[mode].add_variable('density', dtype=np.float32)

    pset = ParticleSet(fieldset, pclass=DensParticle, lon=5, lat=5, depth=1000)

    pset.execute(PolyTEOS10_bsq, runtime=1)
    assert np.allclose(pset[0].density, 1022.85377)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_EOSseawaterproperties_kernels(mode):
    fieldset = FieldSet.from_data(data={'U': 0, 'V': 0,
                                        'psu_salinity': 40,
                                        'temperature': 40,
                                        'potemperature': 36.89073},
                                  dimensions={'lat': 0, 'lon': 0, 'depth': 0})
    fieldset.add_constant('refpressure', float(0))

    PoTempParticle = ptype[mode].add_variables([
        Variable('potemp', dtype=np.float32),
        Variable('pressure', dtype=np.float32, initial=10000)])
    pset = ParticleSet(fieldset, pclass=PoTempParticle, lon=5, lat=5, depth=1000)
    pset.execute(PtempFromTemp, runtime=1)
    assert np.allclose(pset[0].potemp, 36.89073)

    TempParticle = ptype[mode].add_variables([
        Variable('temp', dtype=np.float32),
        Variable('pressure', dtype=np.float32, initial=10000)])
    pset = ParticleSet(fieldset, pclass=TempParticle, lon=5, lat=5, depth=1000)
    pset.execute(TempFromPtemp, runtime=1)
    assert np.allclose(pset[0].temp, 40)

    pset = ParticleSet(fieldset, pclass=TempParticle, lon=5, lat=30, depth=7321.45)
    pset.execute(PressureFromLatDepth, runtime=1)
    assert np.allclose(pset[0].pressure, 7500, atol=1e-2)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('pressure', [0, 10])
def test_UNESCOdensity_kernel(mode, pressure):

    def generate_fieldset(p, xdim=2, ydim=2, zdim=2, tdim=1):
        lon = np.linspace(0., 10., xdim, dtype=np.float32)
        lat = np.linspace(0., 10., ydim, dtype=np.float32)
        depth = np.linspace(0, 2000, zdim, dtype=np.float32)
        time = np.zeros(tdim, dtype=np.float64)
        U = np.ones((tdim, zdim, ydim, xdim))
        V = np.ones((tdim, zdim, ydim, xdim))
        psu_salinity = 8 * np.ones((tdim, zdim, ydim, xdim))
        cons_temperature = 10 * np.ones((tdim, zdim, ydim, xdim))
        cons_pressure = p * np.ones((tdim, zdim, ydim, xdim))
        dimensions = {'lat': lat, 'lon': lon, 'depth': depth, 'time': time}
        data = {'U': np.array(U, dtype=np.float32), 'V': np.array(V, dtype=np.float32),
                'psu_salinity': np.array(psu_salinity, dtype=np.float32),
                'cons_pressure': np.array(cons_pressure, dtype=np.float32),
                'cons_temperature': np.array(cons_temperature, dtype=np.float32)}
        return (data, dimensions)

    data, dimensions = generate_fieldset(pressure)
    fieldset = FieldSet.from_data(data, dimensions)

    DensParticle = ptype[mode].add_variable('density', dtype=np.float32)

    pset = ParticleSet(fieldset, pclass=DensParticle, lon=5, lat=5, depth=1000)

    pset.execute(UNESCODensity, runtime=1)

    if pressure == 0:
        assert np.allclose(pset[0].density, 1005.9465)
    elif pressure == 10:
        assert np.allclose(pset[0].density, 1006.4179)

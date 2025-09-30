import random
from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest

from parcels import (
    Field,
    Kernel,
    Particle,
    ParticleSet,
    Variable,
)
from tests.common_kernels import DoNothing
from tests.utils import create_fieldset_unit_mesh


def expr_kernel(name, pset, expr):
    pycode = (f"def {name}(particle, fieldset, time):\n"
              f"    particle.p = {expr}")  # fmt: skip
    return Kernel(pset.fieldset, pset.particledata.ptype, pyfunc=None, funccode=pycode, funcname=name)


@pytest.fixture
def fieldset_unit_mesh():
    return create_fieldset_unit_mesh()


@pytest.mark.parametrize(
    "name, expr, result",
    [
        ("Add", "2 + 5", 7),
        ("Sub", "6 - 2", 4),
        ("Mul", "3 * 5", 15),
        ("Div", "24 / 4", 6),
    ],
)
def test_expression_int(name, expr, result):
    """Test basic arithmetic expressions."""
    npart = 10
    TestParticle = Particle.add_variable("p", dtype=np.float32, initial=0)
    pset = ParticleSet(
        create_fieldset_unit_mesh(mesh="spherical"),
        pclass=TestParticle,
        lon=np.linspace(0.0, 1.0, npart),
        lat=np.zeros(npart) + 0.5,
    )
    pset.execute(expr_kernel(f"Test{name}", pset, expr), endtime=1.0, dt=1.0)
    assert np.all([p.p == result for p in pset])


@pytest.mark.parametrize(
    "name, expr, result",
    [
        ("Add", "2. + 5.", 7),
        ("Sub", "6. - 2.", 4),
        ("Mul", "3. * 5.", 15),
        ("Div", "24. / 4.", 6),
        ("Pow", "2 ** 3", 8),
    ],
)
def test_expression_float(name, expr, result):
    """Test basic arithmetic expressions."""
    npart = 10
    TestParticle = Particle.add_variable("p", dtype=np.float32, initial=0)
    pset = ParticleSet(
        create_fieldset_unit_mesh(mesh="spherical"),
        pclass=TestParticle,
        lon=np.linspace(0.0, 1.0, npart),
        lat=np.zeros(npart) + 0.5,
    )
    pset.execute(expr_kernel(f"Test{name}", pset, expr), endtime=1.0, dt=1.0)
    assert np.all([p.p == result for p in pset])


@pytest.mark.parametrize(
    "name, expr, result",
    [
        ("True", "True", True),
        ("False", "False", False),
        ("And", "True and False", False),
        ("Or", "True or False", True),
        ("Equal", "5 == 5", True),
        ("NotEqual", "3 != 4", True),
        ("Lesser", "5 < 3", False),
        ("LesserEq", "3 <= 5", True),
        ("Greater", "4 > 2", True),
        ("GreaterEq", "2 >= 4", False),
        ("CheckNaN", "math.nan != math.nan", True),
    ],
)
def test_expression_bool(name, expr, result):
    """Test basic arithmetic expressions."""
    npart = 10
    TestParticle = Particle.add_variable("p", dtype=np.float32, initial=0)
    pset = ParticleSet(
        create_fieldset_unit_mesh(mesh="spherical"),
        pclass=TestParticle,
        lon=np.linspace(0.0, 1.0, npart),
        lat=np.zeros(npart) + 0.5,
    )
    pset.execute(expr_kernel(f"Test{name}", pset, expr), endtime=1.0, dt=1.0)
    assert np.all(result == pset.p)


def test_while_if_break():
    """Test while, if and break commands."""
    TestParticle = Particle.add_variable("p", dtype=np.float32, initial=0)
    pset = ParticleSet(create_fieldset_unit_mesh(mesh="spherical"), pclass=TestParticle, lon=[0], lat=[0])

    def kernel(particle, fieldset, time):  # pragma: no cover
        while particle.p < 30:
            if particle.p > 9:
                break
            particle.p += 1
        if particle.p > 5:
            particle.p *= 2.0

    pset.execute(kernel, endtime=1.0, dt=1.0)
    assert np.allclose(pset.p, 20.0, rtol=1e-12)


def test_nested_if():
    """Test nested if commands."""
    TestParticle = Particle.add_variables(
        [Variable("p0", dtype=np.int32, initial=0), Variable("p1", dtype=np.int32, initial=1)]
    )
    pset = ParticleSet(create_fieldset_unit_mesh(mesh="spherical"), pclass=TestParticle, lon=0, lat=0)

    def kernel(particle, fieldset, time):  # pragma: no cover
        if particle.p1 >= particle.p0:
            var = particle.p0
            if var + 1 < particle.p1:
                particle.p1 = -1

    pset.execute(kernel, endtime=10, dt=1.0)
    assert np.allclose([pset.p0[0], pset.p1[0]], [0, 1])


def test_pass():
    """Test pass commands."""
    TestParticle = Particle.add_variable("p", dtype=np.float32, initial=0)
    pset = ParticleSet(create_fieldset_unit_mesh(mesh="spherical"), pclass=TestParticle, lon=0, lat=0)

    def kernel(particle, fieldset, time):  # pragma: no cover
        particle.p = -1
        pass

    pset.execute(kernel, endtime=10, dt=1.0)
    assert np.allclose(pset[0].p, -1)


def test_dt_as_variable_in_kernel():
    pset = ParticleSet(create_fieldset_unit_mesh(mesh="spherical"), pclass=Particle, lon=0, lat=0)

    def kernel(particle, fieldset, time):  # pragma: no cover
        dt = 1.0  # noqa

    pset.execute(kernel, endtime=10, dt=1.0)


def test_varname_as_fieldname():
    """Tests for error thrown if variable has same name as Field."""
    fset = create_fieldset_unit_mesh(mesh="spherical")
    fset.add_field(Field("speed", 10, lon=0, lat=0))
    fset.add_constant("vertical_speed", 0.1)
    particle = Particle.add_variable("speed")
    pset = ParticleSet(fset, pclass=particle, lon=0, lat=0)

    def kernel_particlename(particle, fieldset, time):  # pragma: no cover
        particle.speed = fieldset.speed[particle]

    pset.execute(kernel_particlename, endtime=1, dt=1.0)
    assert pset[0].speed == 10

    def kernel_varname(particle, fieldset, time):  # pragma: no cover
        vertical_speed = fieldset.vertical_speed  # noqa

    pset.execute(kernel_varname, endtime=1, dt=1.0)


def test_if_withfield(fieldset_unit_mesh):
    """Test combination of if and Field sampling commands."""
    TestParticle = Particle.add_variable("p", dtype=np.float32, initial=0)
    pset = ParticleSet(fieldset_unit_mesh, pclass=TestParticle, lon=[0], lat=[0])

    def kernel(particle, fieldset, time):  # pragma: no cover
        u, v = fieldset.UV[time, 0, 0, 1.0]
        particle.p = 0
        if fieldset.U[time, 0, 0, 1.0] == u:
            particle.p += 1
        if fieldset.U[time, 0, 0, 1.0] == fieldset.U[time, 0, 0, 1.0]:
            particle.p += 1
        if True:
            particle.p += 1
        if fieldset.U[time, 0, 0, 1.0] == u and 1 == 1:
            particle.p += 1
        if (
            fieldset.U[time, 0, 0, 1.0] == fieldset.U[time, 0, 0, 1.0]
            and fieldset.U[time, 0, 0, 1.0] == fieldset.U[time, 0, 0, 1.0]
        ):
            particle.p += 1
        if fieldset.U[time, 0, 0, 1.0] == u:
            particle.p += 1
        else:
            particle.p += 1000
        if fieldset.U[time, 0, 0, 1.0] == 3:
            particle.p += 1000
        else:
            particle.p += 1

    pset.execute(kernel, endtime=1.0, dt=1.0)
    assert np.allclose(pset.p, 7.0, rtol=1e-12)
    return


def test_print(fieldset_unit_mesh, capfd):
    """Test print statements."""
    TestParticle = Particle.add_variable("p", dtype=np.float32, initial=0)
    pset = ParticleSet(fieldset_unit_mesh, pclass=TestParticle, lon=[0.5], lat=[0.5])

    def kernel(particle, fieldset, time):  # pragma: no cover
        particle.p = 1e-3
        tmp = 5
        print(f"{particle.trajectory} {particle.p:f} {tmp:f}")

    pset.execute(kernel, endtime=1.0, dt=1.0, verbose_progress=False)
    out, err = capfd.readouterr()
    lst = out.split(" ")
    tol = 1e-8
    assert (
        abs(float(lst[0]) - pset.trajectory[0]) < tol
        and abs(float(lst[1]) - pset.p[0]) < tol
        and abs(float(lst[2]) - 5) < tol
    )

    def kernel2(particle, fieldset, time):  # pragma: no cover
        tmp = 3
        print(f"{tmp:f}")

    pset.execute(kernel2, endtime=2.0, dt=1.0, verbose_progress=False)
    out, err = capfd.readouterr()
    lst = out.split(" ")
    assert abs(float(lst[0]) - 3) < tol


def test_fieldset_access(fieldset_unit_mesh):
    pset = ParticleSet(fieldset_unit_mesh, pclass=Particle, lon=0, lat=0)

    def kernel(particle, fieldset, time):  # pragma: no cover
        particle.lon = fieldset.U.grid.lon[2]

    pset.execute(kernel, endtime=1, dt=1.0)
    assert pset.lon[0] == fieldset_unit_mesh.U.grid.lon[2]


@pytest.mark.parametrize("concat", [False, True])
def test_random_kernel_concat(fieldset_unit_mesh, concat):
    TestParticle = Particle.add_variable("p", dtype=np.float32, initial=0)
    pset = ParticleSet(fieldset_unit_mesh, pclass=TestParticle, lon=0, lat=0)

    def RandomKernel(particle, fieldset, time):  # pragma: no cover
        particle.p += random.uniform(0, 1)

    def AddOne(particle, fieldset, time):  # pragma: no cover
        particle.p += 1.0

    kernels = [RandomKernel, AddOne] if concat else RandomKernel
    pset.execute(kernels, runtime=1)
    assert pset.p > 1 if concat else pset.p < 1


def test_dt_modif_by_kernel():
    TestParticle = Particle.add_variable("age", dtype=np.float32, initial=0)
    pset = ParticleSet(create_fieldset_unit_mesh(mesh="spherical"), pclass=TestParticle, lon=[0.5], lat=[0])

    def modif_dt(particle, fieldset, time):  # pragma: no cover
        particle.age += particle.dt
        particle.dt = 2

    endtime = 4
    pset.execute(modif_dt, endtime=endtime + 1, dt=1.0)
    assert np.isclose(pset.time[0], endtime)


@pytest.mark.parametrize(
    ("dt", "expectation"), [(1e-2, does_not_raise()), (1e-5, does_not_raise()), (1e-6, pytest.raises(ValueError))]
)
def test_small_dt(dt, expectation):
    npart = 10
    pset = ParticleSet(
        create_fieldset_unit_mesh(mesh="spherical"),
        pclass=Particle,
        lon=np.zeros(npart),
        lat=np.zeros(npart),
        time=np.arange(0, npart) * dt * 10,
    )

    with expectation:
        pset.execute(DoNothing, dt=dt, runtime=dt * 101)
        assert np.allclose([p.time for p in pset], dt * 100)

from operator import attrgetter

import numpy as np
import pytest

from parcels import (
    AdvectionRK4,
    Particle,
    ParticleSet,
    Variable,
)
from tests.utils import create_fieldset_zeros_unit_mesh


@pytest.fixture
def fieldset():
    return create_fieldset_zeros_unit_mesh()


def test_print(fieldset):
    TestParticle = Particle.add_variable("p", to_write=True)
    pset = ParticleSet(fieldset, pclass=TestParticle, lon=[0, 1], lat=[0, 1])
    print(pset)


def test_variable_init(fieldset):
    """Test that checks correct initialisation of custom variables."""
    npart = 10
    extra_vars = [
        Variable("p_float", dtype=np.float32, initial=10.0),
        Variable("p_double", dtype=np.float64, initial=11.0),
    ]
    TestParticle = Particle.add_variables(extra_vars)
    TestParticle = TestParticle.add_variable("p_int", np.int32, initial=12.0)
    pset = ParticleSet(fieldset, pclass=TestParticle, lon=np.linspace(0, 1, npart), lat=np.linspace(1, 0, npart))

    def addOne(particle, fieldset, time):  # pragma: no cover
        particle.p_float += 1.0
        particle.p_double += 1.0
        particle.p_int += 1

    pset.execute(pset.Kernel(AdvectionRK4) + addOne, runtime=1.0, dt=1.0)
    assert np.allclose([p.p_float for p in pset], 11.0, rtol=1e-12)
    assert np.allclose([p.p_double for p in pset], 12.0, rtol=1e-12)
    assert np.allclose([p.p_int for p in pset], 13, rtol=1e-12)


@pytest.mark.parametrize("type", ["np.int8", "mp.float", "np.int16"])
def test_variable_unsupported_dtypes(fieldset, type):
    """Test that checks errors thrown for unsupported dtypes."""
    TestParticle = Particle.add_variable("p", dtype=type, initial=10.0)
    with pytest.raises((RuntimeError, TypeError)):
        ParticleSet(fieldset, pclass=TestParticle, lon=[0], lat=[0])


def test_variable_special_names(fieldset):
    """Test that checks errors thrown for special names."""
    for vars in ["z", "lon"]:
        TestParticle = Particle.add_variable(vars, dtype=np.float32, initial=10.0)
        with pytest.raises(AttributeError):
            ParticleSet(fieldset, pclass=TestParticle, lon=[0], lat=[0])


@pytest.mark.parametrize("coord_type", [np.float32, np.float64])
def test_variable_init_relative(fieldset, coord_type):
    """Test that checks relative initialisation of custom variables."""
    npart = 10
    lonlat_type = np.float64 if coord_type == "double" else np.float32

    TestParticle = Particle.add_variables(
        [
            Variable("p_base", dtype=lonlat_type, initial=10.0),
            Variable("p_relative", dtype=lonlat_type, initial=attrgetter("p_base")),
            Variable("p_lon", dtype=lonlat_type, initial=attrgetter("lon")),
            Variable("p_lat", dtype=lonlat_type, initial=attrgetter("lat")),
        ]
    )

    lon = np.linspace(0, 1, npart, dtype=lonlat_type)
    lat = np.linspace(1, 0, npart, dtype=lonlat_type)
    pset = ParticleSet(fieldset, pclass=TestParticle, lon=lon, lat=lat, lonlatdepth_dtype=coord_type)
    # Adjust base variable to test for aliasing effects
    for p in pset:
        p.p_base += 3.0
    assert np.allclose([p.p_base for p in pset], 13.0, rtol=1e-12)
    assert np.allclose([p.p_relative for p in pset], 10.0, rtol=1e-12)
    assert np.allclose([p.p_lon for p in pset], lon, rtol=1e-12)
    assert np.allclose([p.p_lat for p in pset], lat, rtol=1e-12)

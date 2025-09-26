import numpy as np
import pytest

from parcels import (
    FieldSet,
    Particle,
    ParticleSet,
)
from tests.utils import create_fieldset_unit_mesh


@pytest.fixture
def fieldset_unit_mesh():
    return create_fieldset_unit_mesh()


@pytest.mark.parametrize("kernel_type", ["update_lon", "update_dlon"])
def test_execution_order(kernel_type):
    fieldset = FieldSet.from_data(
        {"U": [[0, 1], [2, 3]], "V": np.ones((2, 2))}, {"lon": [0, 2], "lat": [0, 2]}, mesh="flat"
    )

    def MoveLon_Update_Lon(particle, fieldset, time):  # pragma: no cover
        particle.lon += 0.2

    def MoveLon_Update_dlon(particle, fieldset, time):  # pragma: no cover
        particle.dlon += 0.2

    def SampleP(particle, fieldset, time):  # pragma: no cover
        particle.p = fieldset.U[time, particle.depth, particle.lat, particle.lon]

    SampleParticle = Particle.add_variable("p", dtype=np.float32, initial=0.0)

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


def test_multi_kernel_duplicate_varnames(fieldset_unit_mesh):
    # Testing for merging of two Kernels with the same variable declared
    # Should throw a warning, but go ahead regardless
    def Kernel1(particle, fieldset, time):  # pragma: no cover
        add_lon = 0.1
        particle.dlon += add_lon

    def Kernel2(particle, fieldset, time):  # pragma: no cover
        add_lon = -0.3
        particle.dlon += add_lon

    pset = ParticleSet(fieldset_unit_mesh, pclass=Particle, lon=[0.5], lat=[0.5])
    pset.execute([Kernel1, Kernel2], endtime=2.0, dt=1.0)
    assert np.allclose(pset.lon, 0.3, rtol=1e-5)


def test_update_kernel_in_script(fieldset_unit_mesh):
    # Testing what happens when kernels are updated during runtime of a script
    # Should throw a warning, but go ahead regardless
    def MoveEast(particle, fieldset, time):  # pragma: no cover
        add_lon = 0.1
        particle.dlon += add_lon

    def MoveWest(particle, fieldset, time):  # pragma: no cover
        add_lon = -0.3
        particle.dlon += add_lon

    pset = ParticleSet(fieldset_unit_mesh, pclass=Particle, lon=[0.5], lat=[0.5])
    pset.execute(pset.Kernel(MoveEast), endtime=1.0, dt=1.0)
    pset.execute(pset.Kernel(MoveWest), endtime=3.0, dt=1.0)
    assert np.allclose(pset.lon, 0.3, rtol=1e-5)  # should be 0.5 + 0.1 - 0.3 = 0.3

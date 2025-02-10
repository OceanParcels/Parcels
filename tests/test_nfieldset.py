import numpy as np
import pytest

from parcels import Field, ParticleSet, ScipyParticle, TimeConverter

# from tests.utils import create_fieldset_zeros_simple


@pytest.fixture
def fieldset():
    class GridSet:
        size = 1

        def dimrange(self, *args, **kwargs):
            return [0]

    class FieldSet:
        time_origin = TimeConverter()

        gridset = GridSet()

        U = Field()

        def _check_complete(self):
            return True

    return FieldSet()


def test_pset_create_line(fieldset):
    npart = 100
    lon = np.linspace(0, 1, npart, dtype="float64")
    lat = np.linspace(1, 0, npart, dtype="float64")
    pset = ParticleSet.from_line(
        fieldset,
        size=npart,
        start=(0, 1),
        finish=(1, 0),
        pclass=ScipyParticle,
    )
    assert np.allclose([p.lon for p in pset], lon, rtol=1e-12)
    assert np.allclose([p.lat for p in pset], lat, rtol=1e-12)

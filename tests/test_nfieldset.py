import numpy as np
import pytest

from datetime import timedelta

from parcels import (
    Field,
    ParticleSet,
    ScipyParticle,
    JITParticle,
    TimeConverter,
    AdvectionRK4,
    Grid,
)
from parcels.field import RandomField, ZeroField

# from tests.utils import create_fieldset_zeros_simple


@pytest.fixture
def fieldset():
    class GridSet:
        size = 1
        grids = []

        def dimrange(self, *args, **kwargs):
            return (0, np.inf)

    class FieldSet:
        time_origin = TimeConverter()

        gridset = GridSet()

        V = ZeroField(
            grid=Grid(),
            ndims=1,
        )
        U = RandomField(
            grid=Grid(),
            ndims=1,
            scale=1e-3,
        )
        UV = RandomField(
            grid=Grid(),
            ndims=2,
            scale=1e-3,
        )

        def _check_complete(self):
            return True

        def computeTimeChunk(self, *args, **kwargs):
            return np.inf

        def get_fields(self, *args, **kwargs):
            return []

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


def test_advection_zonal(fieldset):
    npart = 10
    pset = ParticleSet.from_line(
        fieldset,
        size=npart,
        start=(0, 1),
        finish=(1, 0),
        pclass=ScipyParticle,
    )
    print(pset)
    pset.execute(AdvectionRK4, runtime=timedelta(hours=2), dt=timedelta(seconds=30))
    print(pset)

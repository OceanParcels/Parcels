import numpy as np
import pytest
import xarray as xr

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
from parcels.field import RandomField, ZeroField, UVXarrayField

# from tests.utils import create_fieldset_zeros_simple


@pytest.fixture
def xarray_fieldset():
    class GridSet:
        size = 1
        grids = []

        def dimrange(self, *args, **kwargs):
            return (0, np.inf)

    class FieldSet:
        time_origin = TimeConverter()

        gridset = GridSet()

        V = ZeroField(grid=Grid(), ndims=1)
        U = RandomField(grid=Grid(), ndims=1, scale=1e-3)
        UV = UVXarrayField(
            ds=xr.Dataset(
                {
                    "U": (
                        ("time", "depth", "lat", "lon"),
                        1e-3 * np.random.normal(size=(1, 1, 181, 360)) + 2e-3,
                    ),
                    "V": (("time", "depth", "lat", "lon"), np.zeros((1, 1, 181, 360))),
                },
                coords={
                    "time": [
                        0,
                    ],
                    "depth": [
                        0,
                    ],
                    "lat": np.linspace(-90, 90, 181),
                    "lon": np.linspace(-180, 180, 361)[:-1],
                },
            ),
            grid=Grid(),
        )

        def _check_complete(self):
            return True

        def computeTimeChunk(self, *args, **kwargs):
            return np.inf

        def get_fields(self, *args, **kwargs):
            return []

    return FieldSet()


@pytest.fixture
def random_fieldset():
    class GridSet:
        size = 1
        grids = []

        def dimrange(self, *args, **kwargs):
            return (0, np.inf)

    class FieldSet:
        time_origin = TimeConverter()

        gridset = GridSet()

        V = ZeroField(grid=Grid(), ndims=1)
        U = RandomField(grid=Grid(), ndims=1, scale=1e-3)
        UV = RandomField(grid=Grid(), ndims=2, scale=1e-3)

        def _check_complete(self):
            return True

        def computeTimeChunk(self, *args, **kwargs):
            return np.inf

        def get_fields(self, *args, **kwargs):
            return []

    return FieldSet()


def test_pset_create_line(random_fieldset):
    npart = 100
    lon = np.linspace(0, 1, npart, dtype="float64")
    lat = np.linspace(1, 0, npart, dtype="float64")
    pset = ParticleSet.from_line(
        random_fieldset,
        size=npart,
        start=(0, 1),
        finish=(1, 0),
        pclass=ScipyParticle,
    )
    assert np.allclose([p.lon for p in pset], lon, rtol=1e-12)
    assert np.allclose([p.lat for p in pset], lat, rtol=1e-12)


def test_advection_random(random_fieldset):
    npart = 10
    pset = ParticleSet.from_line(
        random_fieldset,
        size=npart,
        start=(0, 1),
        finish=(1, 0),
        pclass=ScipyParticle,
    )
    print(pset)
    pset.execute(AdvectionRK4, runtime=timedelta(hours=2), dt=timedelta(seconds=30))
    print(pset)


def test_advection_xarray(xarray_fieldset):
    npart = 10
    pset = ParticleSet.from_line(
        xarray_fieldset,
        size=npart,
        start=(0, 1),
        finish=(1, 0),
        pclass=ScipyParticle,
    )
    pset.execute(AdvectionRK4, runtime=timedelta(hours=2), dt=timedelta(seconds=30))

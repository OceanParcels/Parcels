"""Tests for outputs from structured GCMs."""

import pytest


@pytest.mark.v4alpha
@pytest.mark.skip(reason="From_pop is not supported during v4-alpha development. This will be reconsidered in v4.")
def test_fieldset_frompop():
    # # Initial v3 test
    # filenames = str(TEST_DATA / "POPtestdata_time.nc")
    # variables = {"U": "U", "V": "V", "W": "W", "T": "T"}
    # dimensions = {"lon": "lon", "lat": "lat", "time": "time"}

    # fieldset = FieldSet.from_pop(filenames, variables, dimensions, mesh="flat")
    # pset = ParticleSet(fieldset, Particle, lon=[3, 5, 1], lat=[3, 5, 1])
    # pset.execute(AdvectionRK4, runtime=3, dt=1)
    pass

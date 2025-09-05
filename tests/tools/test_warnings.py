import numpy as np
import pytest

from parcels import (
    AdvectionRK4,
    AdvectionRK45,
    FieldSet,
    FieldSetWarning,
    KernelWarning,
    Particle,
    ParticleSet,
    ParticleSetWarning,
)
from parcels.particlefile import ParticleFile
from tests.utils import TEST_DATA


@pytest.mark.v4alpha
@pytest.mark.xfail(reason="From_pop is not supported during v4-alpha development. This will be reconsidered in v4.")
def test_fieldset_warning_pop():
    filenames = str(TEST_DATA / "POPtestdata_time.nc")
    variables = {"U": "U", "V": "V", "W": "W", "T": "T"}
    dimensions = {"lon": "lon", "lat": "lat", "depth": "w_deps", "time": "time"}
    with pytest.warns(FieldSetWarning, match="General s-levels are not supported in B-grid.*"):
        # b-grid with s-levels and POP output in meters warning
        FieldSet.from_pop(filenames, variables, dimensions, mesh="flat")


def test_file_warnings(tmp_zarrfile):
    fieldset = FieldSet.from_data(
        data={"U": np.zeros((1, 1)), "V": np.zeros((1, 1))}, dimensions={"lon": [0], "lat": [0]}
    )
    pset = ParticleSet(fieldset=fieldset, pclass=Particle, lon=[0, 0], lat=[0, 0], time=[0, 1])
    pfile = ParticleFile(name=tmp_zarrfile, outputdt=2)
    with pytest.warns(ParticleSetWarning, match="Some of the particles have a start time difference.*"):
        pset.execute(AdvectionRK4, runtime=3, dt=1, output_file=pfile)


def test_kernel_warnings():
    # RK45 warnings
    lat = [0, 1, 5, 10]
    lon = [0, 1, 5, 10]
    u = [[1, 1, 1, 1] for _ in range(4)]
    v = [[1, 1, 1, 1] for _ in range(4)]
    fieldset = FieldSet.from_data(data={"U": u, "V": v}, dimensions={"lon": lon, "lat": lat})
    pset = ParticleSet(
        fieldset=fieldset,
        pclass=Particle.add_variable("next_dt", dtype=np.float32, initial=1),
        lon=[0],
        lat=[0],
        depth=[0],
        time=[0],
        next_dt=1,
    )
    with pytest.warns(KernelWarning):
        pset.execute(AdvectionRK45, runtime=1, dt=1)

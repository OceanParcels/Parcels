import warnings

import numpy as np
import pytest

from parcels import (
    AdvectionRK4,
    AdvectionRK4_3D,
    AdvectionRK45,
    FieldSet,
    FieldSetWarning,
    KernelWarning,
    Particle,
    ParticleSet,
    ParticleSetWarning,
)
from tests.utils import TEST_DATA


def test_fieldset_warnings():
    # halo with inconsistent boundaries
    lat = [0, 1, 5, 10]
    lon = [0, 1, 5, 10]
    u = [[1, 1, 1, 1] for _ in range(4)]
    v = [[1, 1, 1, 1] for _ in range(4)]
    fieldset = FieldSet.from_data(data={"U": u, "V": v}, dimensions={"lon": lon, "lat": lat}, transpose=True)
    with pytest.warns(FieldSetWarning):
        fieldset.add_periodic_halo(meridional=True, zonal=True)

    # flipping lats warning
    lat = [0, 1, 5, -5]
    lon = [0, 1, 5, 10]
    u = [[1, 1, 1, 1] for _ in range(4)]
    v = [[1, 1, 1, 1] for _ in range(4)]
    with pytest.warns(FieldSetWarning):
        fieldset = FieldSet.from_data(data={"U": u, "V": v}, dimensions={"lon": lon, "lat": lat}, transpose=True)

    with pytest.warns(FieldSetWarning):
        # allow_time_extrapolation with time_periodic warning
        fieldset = FieldSet.from_data(
            data={"U": u, "V": v},
            dimensions={"lon": lon, "lat": lat},
            transpose=True,
            allow_time_extrapolation=True,
            time_periodic=1,
        )

    filenames = str(TEST_DATA / "POPtestdata_time.nc")
    variables = {"U": "U", "V": "V", "W": "W", "T": "T"}
    dimensions = {"lon": "lon", "lat": "lat", "depth": "w_deps", "time": "time"}
    with pytest.warns(FieldSetWarning):
        # b-grid with s-levels and POP output in meters warning
        fieldset = FieldSet.from_pop(filenames, variables, dimensions, mesh="flat")
    with pytest.warns(FieldSetWarning):
        # timestamps with time in file warning
        fieldset = FieldSet.from_pop(filenames, variables, dimensions, mesh="flat", timestamps=[0, 1, 2, 3])


def test_file_warnings(tmp_zarrfile):
    fieldset = FieldSet.from_data(
        data={"U": np.zeros((1, 1)), "V": np.zeros((1, 1))}, dimensions={"lon": [0], "lat": [0]}
    )
    pset = ParticleSet(fieldset=fieldset, pclass=Particle, lon=[0, 0], lat=[0, 0], time=[0, 1])
    pfile = pset.ParticleFile(name=tmp_zarrfile, outputdt=2)
    with pytest.warns(ParticleSetWarning, match="Some of the particles have a start time difference.*"):
        pset.execute(AdvectionRK4, runtime=3, dt=1, output_file=pfile)


def test_kernel_warnings():
    # positive scaling factor for W
    filenames = str(TEST_DATA / "POPtestdata_time.nc")
    variables = {"U": "U", "V": "V", "W": "W", "T": "T"}
    dimensions = {"lon": "lon", "lat": "lat", "depth": "w_deps", "time": "time"}
    with warnings.catch_warnings():
        # ignore FieldSetWarnings (tested in test_fieldset_warnings)
        warnings.simplefilter("ignore", FieldSetWarning)
        fieldset = FieldSet.from_pop(filenames, variables, dimensions, mesh="flat")
        fieldset.W._scaling_factor = 0.01
        pset = ParticleSet(fieldset=fieldset, pclass=Particle, lon=[0], lat=[0], depth=[0], time=[0])
        with pytest.warns(KernelWarning):
            pset.execute(AdvectionRK4_3D, runtime=1, dt=1)

    # RK45 warnings
    lat = [0, 1, 5, 10]
    lon = [0, 1, 5, 10]
    u = [[1, 1, 1, 1] for _ in range(4)]
    v = [[1, 1, 1, 1] for _ in range(4)]
    fieldset = FieldSet.from_data(data={"U": u, "V": v}, dimensions={"lon": lon, "lat": lat}, transpose=True)
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

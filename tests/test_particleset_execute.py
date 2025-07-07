import tempfile
from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pytest
import xarray as xr

import parcels


@dataclass
class Params:
    outputdt: timedelta
    dt: timedelta
    runtime: timedelta = None
    endtime: timedelta = None


def assert_all_particles_same_time(ds: xr.Dataset):
    assert np.allclose(
        ds.time.diff(dim="trajectory").astype("float64"), 0
    ), "All particles should have the same time value. Not coercible to 1D time array."


def get_1d_time_output_in_float_hours(ds: xr.Dataset) -> xr.DataArray:
    assert_all_particles_same_time(ds)
    return ds.time.isel(trajectory=0) / np.timedelta64(1, "h")


def execute_particles(fieldset, lon, lat, params: Params) -> xr.Dataset:
    pset = parcels.ParticleSet.from_list(
        fieldset=fieldset,
        pclass=parcels.JITParticle,
        lon=lon,
        lat=lat,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = pset.ParticleFile(
            name=f"{tmpdir}/temp.zarr",
            outputdt=params.outputdt,
        )

        pset.execute(
            parcels.AdvectionRK4,
            runtime=params.runtime,
            dt=params.dt,
            output_file=output_file,
        )

        return xr.open_zarr(f"{tmpdir}/temp.zarr").load()


def get_some_fieldset():
    example_dataset_folder = parcels.download_example_dataset("MovingEddies_data")

    filenames = {
        "U": str(example_dataset_folder / "moving_eddiesU.nc"),
        "V": str(example_dataset_folder / "moving_eddiesV.nc"),
    }
    variables = {"U": "vozocrtx", "V": "vomecrty"}
    dimensions = {"lon": "nav_lon", "lat": "nav_lat", "time": "time_counter"}
    fieldset = parcels.FieldSet.from_netcdf(filenames, variables, dimensions)

    fieldset.computeTimeChunk()
    return fieldset


def test_runtime_and_outputdt_evenly_divisible():
    """Test case where runtime % outputdt==0 and runtime % dt == 0"""
    params = Params(
        outputdt=timedelta(hours=1),
        runtime=timedelta(hours=6),  # divisible by both outputdt and dt
        dt=timedelta(minutes=10),
    )
    ds = execute_particles(
        get_some_fieldset(),
        lon=[3.3e5, 3.3e5],
        lat=[1e5, 2.8e5],
        params=params,
    )
    assert np.allclose(get_1d_time_output_in_float_hours(ds), [0, 1, 2, 3, 4, 5])  # 0 to 6 hours (excluding 6)


def test_runtime_and_outputdt_not_evenly_divisible():
    params = Params(
        outputdt=timedelta(hours=1),
        runtime=timedelta(hours=6 + 11 / 60),  # runtime is not evenly divisible by outputdt
        dt=timedelta(minutes=10),
    )
    ds = execute_particles(
        get_some_fieldset(),
        lon=[3.3e5, 3.3e5],
        lat=[1e5, 2.8e5],
        params=params,
    )

    assert np.allclose(get_1d_time_output_in_float_hours(ds), [0, 1, 2, 3, 4, 5, 6])  # 0 to 6 hours (including 6)


def test_outputdt_not_divisible_by_dt():
    params = Params(
        outputdt=timedelta(hours=1),
        runtime=timedelta(hours=6),
        dt=timedelta(minutes=25),  # outputdt is not a multiple of dt
    )
    fieldset = get_some_fieldset()
    ds = execute_particles(
        fieldset,
        lon=[3.3e5, 3.3e5],
        lat=[1e5, 2.8e5],
        params=params,
    )
    breakpoint()
    # ???? Not the actual time of the expected output since its at a dt cadence? Expected [0, 3*25/60, 5*25/60, ...]?
    assert np.allclose(get_1d_time_output_in_float_hours(ds), [0, 1, 2, 3, 4, 5])  # 0 to 6 hours (excluding 6)


def test_zero_runtime():
    """Test case where runtime is 0"""
    params = Params(
        outputdt=timedelta(hours=1),
        runtime=timedelta(0),
        dt=timedelta(minutes=5),
    )
    with pytest.raises(FileNotFoundError):
        execute_particles(
            get_some_fieldset(),
            lon=[3.3e5, 3.3e5],
            lat=[1e5, 2.8e5],
            params=params,
        )


def test_zero_dt():
    """Test case where dt is 0"""
    params = Params(
        outputdt=timedelta(hours=1),
        runtime=timedelta(hours=6),
        dt=timedelta(0),
    )
    with pytest.raises(ValueError, msg="Time step dt is too small"):
        execute_particles(
            get_some_fieldset(),
            lon=[3.3e5, 3.3e5],
            lat=[1e5, 2.8e5],
            params=params,
        )


def test_outputdt_equal_to_runtime():
    params = Params(
        outputdt=timedelta(hours=1),  # outputdt equal to runtime
        runtime=timedelta(hours=1),
        dt=timedelta(minutes=5),
    )
    ds = execute_particles(
        get_some_fieldset(),
        lon=[3.3e5, 3.3e5],
        lat=[1e5, 2.8e5],
        params=params,
    )
    assert np.allclose(get_1d_time_output_in_float_hours(ds), [0])  # 0 (only one output at start time)


def test_outputdt_greater_than_runtime():
    params = Params(
        outputdt=timedelta(hours=2),  # outputdt greater than runtime
        runtime=timedelta(hours=1),
        dt=timedelta(minutes=5),
    )
    ds = execute_particles(
        get_some_fieldset(),
        lon=[3.3e5, 3.3e5],
        lat=[1e5, 2.8e5],
        params=params,
    )
    assert np.allclose(get_1d_time_output_in_float_hours(ds), [0])  # 0 (only one output at start time)


def test_outputdt_less_than_dt():
    params = Params(
        outputdt=timedelta(minutes=1),
        runtime=timedelta(hours=1),
        dt=timedelta(minutes=5),
    )
    ds = execute_particles(
        get_some_fieldset(),
        lon=[3.3e5, 3.3e5],
        lat=[1e5, 2.8e5],
        params=params,
    )
    assert np.allclose(get_1d_time_output_in_float_hours(ds) * 60, np.arange(4, 60, 1))  # ?? Why np.arange(4, 60, 1) ?


def test_timed_particle_release():
    """Test to verify if runtime is from first particle release or first fieldset time"""
    pytest.skip("# TODO: This test needs additional setup for timed particle release")
    params = Params(
        outputdt=timedelta(hours=1),
        runtime=timedelta(hours=6),
        dt=timedelta(minutes=5),
    )
    _ = execute_particles(
        get_some_fieldset(),
        lon=[3.3e5, 3.3e5],
        lat=[1e5, 2.8e5],
        params=params,
    )

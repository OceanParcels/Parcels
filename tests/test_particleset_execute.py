import tempfile
from dataclasses import dataclass
from datetime import timedelta

import xarray as xr

import parcels


@dataclass
class Params:
    outputdt: timedelta
    dt: timedelta
    runtime: timedelta = None
    endtime: timedelta = None


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
        dt=timedelta(minutes=30),
    )
    ds = execute_particles(
        get_some_fieldset(),
        lon=[3.3e5, 3.3e5],
        lat=[1e5, 2.8e5],
        params=params,
    )
    # TODO: Add validation code here


def test_runtime_divisible_by_outputdt_only():
    """Test case where runtime % outputdt==0 and runtime % dt != 0"""
    params = Params(
        outputdt=timedelta(hours=1),
        runtime=timedelta(hours=6),
        dt=timedelta(minutes=17),  # not evenly divisible
    )
    ds = execute_particles(
        get_some_fieldset(),
        lon=[3.3e5, 3.3e5],
        lat=[1e5, 2.8e5],
        params=params,
    )
    # TODO: Add validation code here


def test_runtime_divisible_by_dt_only():
    """Test case where runtime % outputdt!=0 and runtime % dt == 0"""
    params = Params(
        outputdt=timedelta(minutes=45),
        runtime=timedelta(hours=2),  # divisible by dt but not outputdt
        dt=timedelta(minutes=30),
    )
    ds = execute_particles(
        get_some_fieldset(),
        lon=[3.3e5, 3.3e5],
        lat=[1e5, 2.8e5],
        params=params,
    )
    # TODO: Add validation code here


def test_no_even_divisions():
    """Test case where runtime % outputdt!=0 and runtime % dt != 0"""
    params = Params(
        outputdt=timedelta(minutes=45),
        runtime=timedelta(minutes=137),  # prime number of minutes
        dt=timedelta(minutes=17),
    )
    ds = execute_particles(
        get_some_fieldset(),
        lon=[3.3e5, 3.3e5],
        lat=[1e5, 2.8e5],
        params=params,
    )
    # TODO: Add validation code here


def test_outputdt_not_multiple_of_dt():
    """Test case where outputdt is not a multiple of dt"""
    params = Params(
        outputdt=timedelta(minutes=17),
        runtime=timedelta(hours=1),
        dt=timedelta(minutes=5),
    )
    ds = execute_particles(
        get_some_fieldset(),
        lon=[3.3e5, 3.3e5],
        lat=[1e5, 2.8e5],
        params=params,
    )
    # TODO: Add validation code here


def test_zero_runtime():
    """Test case where runtime is 0"""
    params = Params(
        outputdt=timedelta(hours=1),
        runtime=timedelta(0),
        dt=timedelta(minutes=5),
    )
    ds = execute_particles(
        get_some_fieldset(),
        lon=[3.3e5, 3.3e5],
        lat=[1e5, 2.8e5],
        params=params,
    )
    # TODO: Add validation code here


def test_zero_dt():
    """Test case where dt is 0"""
    params = Params(
        outputdt=timedelta(hours=1),
        runtime=timedelta(hours=6),
        dt=timedelta(0),
    )
    ds = execute_particles(
        get_some_fieldset(),
        lon=[3.3e5, 3.3e5],
        lat=[1e5, 2.8e5],
        params=params,
    )
    # TODO: Add validation code here


def test_outputdt_greater_than_runtime():
    """Test case where outputdt > runtime"""
    params = Params(
        outputdt=timedelta(hours=2),
        runtime=timedelta(hours=1),
        dt=timedelta(minutes=5),
    )
    ds = execute_particles(
        get_some_fieldset(),
        lon=[3.3e5, 3.3e5],
        lat=[1e5, 2.8e5],
        params=params,
    )
    # TODO: Add validation code here


def test_outputdt_less_than_dt():
    """Test case where outputdt < dt"""
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
    # TODO: Add validation code here


def test_timed_particle_release():
    """Test to verify if runtime is from first particle release or first fieldset time"""
    # TODO: This test needs additional setup for timed particle release
    params = Params(
        outputdt=timedelta(hours=1),
        runtime=timedelta(hours=6),
        dt=timedelta(minutes=5),
    )
    ds = execute_particles(
        get_some_fieldset(),
        lon=[3.3e5, 3.3e5],
        lat=[1e5, 2.8e5],
        params=params,
    )
    # TODO: Add validation code here

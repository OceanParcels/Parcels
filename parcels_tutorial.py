import tempfile
from dataclasses import dataclass
from datetime import timedelta

import xarray as xr

import parcels


@dataclass
class Params:
    outputdt: timedelta
    runtime: timedelta = None
    endtime: timedelta = None
    dt: timedelta


def execute_particles(fieldset, lon, lat, params: Params) -> xr.Dataset:
    pset = parcels.ParticleSet.from_list(
        fieldset=fieldset,
        pclass=parcels.JITParticle,
        lon=lon,
        lat=lat,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        output_file = pset.ParticleFile(
            name=f"{tmpdir}/EddyParticles.zarr",
            outputdt=params.outputdt,
        )

        pset.execute(
            parcels.AdvectionRK4,
            runtime=params.runtime,
            dt=params.dt,
            output_file=output_file,
        )

        return xr.open_zarr(f"{tmpdir}/EddyParticles.zarr").load()


example_dataset_folder = parcels.download_example_dataset("MovingEddies_data")

filenames = {
    "U": str(example_dataset_folder / "moving_eddiesU.nc"),
    "V": str(example_dataset_folder / "moving_eddiesV.nc"),
}
variables = {"U": "vozocrtx", "V": "vomecrty"}
dimensions = {"lon": "nav_lon", "lat": "nav_lat", "time": "time_counter"}
fieldset = parcels.FieldSet.from_netcdf(filenames, variables, dimensions)

fieldset.computeTimeChunk()

params = Params(
    outputdt=timedelta(hours=1),
    runtime=timedelta(days=6),
    dt=timedelta(minutes=5),
)


ds = execute_particles(
    fieldset,
    lon=[3.3e5, 3.3e5],
    lat=[1e5, 2.8e5],
    params=params,
)
print(ds)

from datetime import timedelta

import xarray as xr

import parcels

example_dataset_folder = parcels.download_example_dataset("MovingEddies_data")

filenames = {
    "U": str(example_dataset_folder / "moving_eddiesU.nc"),
    "V": str(example_dataset_folder / "moving_eddiesV.nc"),
}
variables = {"U": "vozocrtx", "V": "vomecrty"}
dimensions = {"lon": "nav_lon", "lat": "nav_lat", "time": "time_counter"}
fieldset = parcels.FieldSet.from_netcdf(filenames, variables, dimensions)

fieldset.computeTimeChunk()

pset = parcels.ParticleSet.from_list(
    fieldset=fieldset,
    pclass=parcels.JITParticle,
    lon=[3.3e5, 3.3e5],
    lat=[1e5, 2.8e5],
)

output_file = pset.ParticleFile(
    name="EddyParticles.zarr",
    outputdt=timedelta(hours=1),
)

pset.execute(
    parcels.AdvectionRK4,
    runtime=timedelta(days=6),
    dt=timedelta(minutes=5),
    output_file=output_file,
)

ds = xr.open_zarr("EddyParticles.zarr")

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


params = Params(
    outputdt=timedelta(hours=1),
    runtime=timedelta(days=6),
    dt=timedelta(minutes=5),
)

runs = {
    ("run1", r"If runtime % outputdt==0 and runtime % dt == 0"): Params(...),
    ("run2", r"If runtime % outputdt==0 and runtime % dt != 0"): Params(...),
    ("run3", r"If runtime % outputdt!=0 and runtime % dt == 0"): Params(...),
    ("run4", r"If runtime % outputdt!=0 and runtime % dt != 0"): Params(...),
    ("siren", r"what if outputdt is not a multiple of dt?"): Params(...),
    ("run5", r"runtime is 0"): Params(...),
    ("run6", r"dt is 0"): Params(...),
    ("run7", r"outputdt > runtime"): Params(...),
    ("run8", r"outputdt < dt"): Params(...),
    (
        "run9",
        "If you have a timed particle release, is the runtime the time "
        "from the first particle release or from the first fieldset time?",
    ): ...,
}
#
# outputdt is tied to the particle release -> if the first particle is released at 43min with outputdt=1h then all subsequent particles will be recorded at 43min


# How is the particle writing done with adding and deleting particles to the simulation?
# - Are the initial particle locations written?
# - if you delete particles, is the deletion point recorded?
#

# Usecases:
# 1. Meike -> intialisation and then running the simulation

ds = execute_particles(
    get_some_fieldset(),
    lon=[3.3e5, 3.3e5],
    lat=[1e5, 2.8e5],
    params=params,
)
print(ds)

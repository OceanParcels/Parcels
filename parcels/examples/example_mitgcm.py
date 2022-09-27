from datetime import timedelta as delta
from os import path

import numpy as np
import xarray as xr
from parcels import (
    AdvectionRK4,
    FieldSet,
    JITParticle,
    ParticleFile,
    ParticleSet,
    ScipyParticle,
)

ptype = {"scipy": ScipyParticle, "jit": JITParticle}


def run_mitgcm_zonally_reentrant(mode):
    """Function that shows how to load MITgcm data in a zonally periodic domain"""

    data_path = path.join(path.dirname(__file__), "MITgcm_example_data/")
    filenames = {
        "U": data_path + "mitgcm_UV_surface_zonally_reentrant.nc",
        "V": data_path + "mitgcm_UV_surface_zonally_reentrant.nc",
    }
    variables = {"U": "UVEL", "V": "VVEL"}
    dimensions = {
        "U": {"lon": "XG", "lat": "YG", "time": "time"},
        "V": {"lon": "XG", "lat": "YG", "time": "time"},
    }
    fieldset = FieldSet.from_mitgcm(filenames, variables, dimensions, mesh="flat")

    fieldset.add_periodic_halo(zonal=True)
    fieldset.add_constant('domain_width', 1000000)

    def periodicBC(particle, fieldset, time):
        if particle.lon < 0:
            particle.lon += fieldset.domain_width
        elif particle.lon > fieldset.domain_width:
            particle.lon -= fieldset.domain_width

    # Release particles 5 cells away from the Eastern boundary
    pset = ParticleSet.from_line(
        fieldset,
        pclass=ptype[mode],
        start=(fieldset.U.grid.lon[-5], fieldset.U.grid.lat[5]),
        finish=(fieldset.U.grid.lon[-5], fieldset.U.grid.lat[-5]),
        size=10,
    )
    pfile = ParticleFile(
        "MIT_particles_" + str(mode) + ".nc", pset, outputdt=delta(days=1)
    )
    kernels = AdvectionRK4 + pset.Kernel(periodicBC)
    pset.execute(
        kernels, runtime=delta(days=5), dt=delta(minutes=30), output_file=pfile
    )
    pfile.close()


def test_mitgcm_output_compare():
    run_mitgcm_zonally_reentrant("scipy")
    run_mitgcm_zonally_reentrant("jit")

    ds_jit = xr.open_dataset("MIT_particles_jit.nc")
    ds_scipy = xr.open_dataset("MIT_particles_scipy.nc")

    np.testing.assert_allclose(ds_jit.lat.data, ds_scipy.lat.data)
    np.testing.assert_allclose(ds_jit.lon.data, ds_scipy.lon.data)

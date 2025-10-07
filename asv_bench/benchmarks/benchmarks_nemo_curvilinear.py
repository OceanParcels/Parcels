from datetime import timedelta

import numpy as np

import parcels

# ptype = {"scipy": parcels.ScipyParticle, "jit": parcels.JITParticle}
# advection = {"RK4": parcels.AdvectionRK4, "AA": parcels.AdvectionAnalytical}
# path_nemo = "~/Documents/PhD/projects/2025-02_parcels_benchmarking/NemoCurvilinear_data"
path_nemo = parcels.download_example_dataset("NemoCurvilinear_data")


class NemoCurvilinearJIT:
    particle_type = parcels.JITParticle

    def setup(self):
        filenames = {
            "U": {
                "lon": f"{path_nemo}/mesh_mask.nc4",
                "lat": f"{path_nemo}/mesh_mask.nc4",
                "data": f"{path_nemo}/U_purely_zonal-ORCA025_grid_U.nc4",
            },
            "V": {
                "lon": f"{path_nemo}/mesh_mask.nc4",
                "lat": f"{path_nemo}/mesh_mask.nc4",
                "data": f"{path_nemo}/V_purely_zonal-ORCA025_grid_V.nc4",
            },
        }
        variables = {"U": "U", "V": "V"}

        dimensions = {"lon": "glamf", "lat": "gphif", "time": "time_counter"}

        fieldset = parcels.FieldSet.from_nemo(filenames, variables, dimensions, allow_time_extrapolation=True)

        # Start 20 particles on a meridional line at 180W
        npart = 20
        lonp = -180 * np.ones(npart)
        latp = [i for i in np.linspace(-70, 85, npart)]

        self.pset = parcels.ParticleSet.from_list(fieldset, self.particle_type, lon=lonp, lat=latp)
        # pfile = parcels.ParticleFile("nemo_particles", pset, outputdt=timedelta(days=1))

    def time_run_experiment(self):
        self.pset.execute(
            parcels.AdvectionRK4,
            runtime=timedelta(days=30),
            dt=timedelta(hours=6),
            # output_file=pfile,
        )


class NemoCurvilinearScipy(NemoCurvilinearJIT):
    particle_type = parcels.ScipyParticle

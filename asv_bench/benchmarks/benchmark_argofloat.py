from datetime import timedelta

import numpy as np
import xarray as xr

from parcels import AdvectionRK4, FieldSet, JITParticle, ParticleSet, StatusCode, Variable


def ArgoVerticalMovement(particle, fieldset, time):
    driftdepth = 1000  # maximum depth in m
    maxdepth = 2000  # maximum depth in m
    vertical_speed = 0.10  # sink and rise speed in m/s
    cycletime = 10 * 86400  # total time of cycle in seconds
    drifttime = 9 * 86400  # time of deep drift in seconds

    if particle.cycle_phase == 0:
        # Phase 0: Sinking with vertical_speed until depth is driftdepth
        particle_ddepth += vertical_speed * particle.dt
        if particle.depth + particle_ddepth >= driftdepth:
            particle_ddepth = driftdepth - particle.depth
            particle.cycle_phase = 1

    elif particle.cycle_phase == 1:
        # Phase 1: Drifting at depth for drifttime seconds
        particle.drift_age += particle.dt
        if particle.drift_age >= drifttime:
            particle.drift_age = 0  # reset drift_age for next cycle
            particle.cycle_phase = 2

    elif particle.cycle_phase == 2:
        # Phase 2: Sinking further to maxdepth
        particle_ddepth += vertical_speed * particle.dt
        if particle.depth + particle_ddepth >= maxdepth:
            particle_ddepth = maxdepth - particle.depth
            particle.cycle_phase = 3

    elif particle.cycle_phase == 3:
        # Phase 3: Rising with vertical_speed until at surface
        particle_ddepth -= vertical_speed * particle.dt
        # particle.temp = fieldset.temp[time, particle.depth, particle.lat, particle.lon]  # if fieldset has temperature
        if particle.depth + particle_ddepth <= fieldset.mindepth:
            particle_ddepth = fieldset.mindepth - particle.depth
            # particle.temp = 0./0.  # reset temperature to NaN at end of sampling cycle
            particle.cycle_phase = 4

    elif particle.cycle_phase == 4:
        # Phase 4: Transmitting at surface until cycletime is reached
        if particle.cycle_age > cycletime:
            particle.cycle_phase = 0
            particle.cycle_age = 0

    if particle.state == StatusCode.Evaluate:
        particle.cycle_age += particle.dt  # update cycle_age


class ArgoFloatJIT:
    particle_type = JITParticle

    def setup(self):
        self.runtime_days = 45
        time = np.datetime64("2025-01-01") + np.arange(self.runtime_days + 1) * np.timedelta64(1, "D")
        lon = np.linspace(-180, 180, 120)
        lat = np.linspace(-90, 90, 100)

        Lon, Lat = np.meshgrid(lon, lat)

        # Create large-scale gyre flow
        U_gyre = np.cos(np.radians(Lat)) * np.sin(np.radians(Lon))  # Zonal flow
        V_gyre = -np.sin(np.radians(Lat)) * np.cos(np.radians(Lon))  # Meridional flow

        f = 2 * 7.2921e-5 * np.sin(np.radians(Lat))

        U_coriolis = U_gyre * (1 - 0.5 * np.abs(f))
        V_coriolis = V_gyre * (1 - 0.5 * np.abs(f))

        noise_level = 0.1  # Adjust for more or less variability
        U_noise = noise_level * np.random.randn(*U_coriolis.shape)
        V_noise = noise_level * np.random.randn(*V_coriolis.shape)

        # Final realistic U and V velocity fields
        U_final = U_coriolis + U_noise
        V_final = V_coriolis + V_noise

        depth = np.linspace(0, 2000, 100)

        U_val = np.tile(U_final[None, None, :, :], (len(time), len(depth), 1, 1))  # Repeat for each time step
        V_val = np.tile(V_final[None, None, :, :], (len(time), len(depth), 1, 1))

        U = xr.DataArray(
            U_val,
            dims=["time", "depth", "lat", "lon"],
            coords={"time": time, "depth": depth, "lat": lat, "lon": lon},
            name="U_velocity",
        )

        V = xr.DataArray(
            V_val,
            dims=["time", "depth", "lat", "lon"],
            coords={"time": time, "depth": depth, "lat": lat, "lon": lon},
            name="V_velocity",
        )

        ds = xr.Dataset({"U": U, "V": V})

        variables = {
            "U": "U",
            "V": "V",
        }
        dimensions = {"lat": "lat", "lon": "lon", "time": "time", "depth": "depth"}
        fieldset = FieldSet.from_xarray_dataset(ds, variables, dimensions)
        # uppermost layer in the hydrodynamic data
        fieldset.mindepth = fieldset.U.depth[0]
        # Define a new Particle type including extra Variables

        ArgoParticle = self.particle_type.add_variables(
            [
                Variable("cycle_phase", dtype=np.int32, initial=0.0),
                Variable("cycle_age", dtype=np.float32, initial=0.0),
                Variable("drift_age", dtype=np.float32, initial=0.0),
            ]
        )

        self.pset = ParticleSet(fieldset=fieldset, pclass=ArgoParticle, lon=[32], lat=[-31], depth=[0])

    def time_run_many_timesteps(self):
        self.pset.execute(
            [ArgoVerticalMovement, AdvectionRK4], runtime=timedelta(days=self.runtime_days), dt=timedelta(seconds=30)
        )


# How do we derive benchmarks ?
# class ArgoFloatScipy(ArgoFloatJIT):
#     particle_type = ScipyParticle

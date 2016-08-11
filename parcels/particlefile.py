import numpy as np
import netCDF4


__all__ = ['ParticleFile']


class ParticleFile(object):

    def __init__(self, name, particleset, initial_dump=True):
        """Initialise netCDF4.Dataset for trajectory output.

        The output follows the format outlined in the Discrete
        Sampling Geometries section of the CF-conventions:
        http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#discrete-sampling-geometries

        The current implementation is based on the NCEI template:
        http://www.nodc.noaa.gov/data/formats/netcdf/v2.0/trajectoryIncomplete.cdl

        Developer note: We cannot use xray.Dataset here, since it does
        not yet allow incremental writes to disk:
        https://github.com/xray/xray/issues/199

        :param name: Basename of the output file
        :param particlset: ParticleSet to output
        :param initial_dump: Perform initial output at time 0.
        :param user_vars: A list of additional user defined particle variables to write
        """
        self.dataset = netCDF4.Dataset("%s.nc" % name, "w", format="NETCDF4")
        self.dataset.createDimension("obs", None)
        self.dataset.createDimension("trajectory", particleset.size)
        self.dataset.feature_type = "trajectory"
        self.dataset.Conventions = "CF-1.6"
        self.dataset.ncei_template_version = "NCEI_NetCDF_Trajectory_Template_v2.0"

        # Create ID variable according to CF conventions
        self.trajectory = self.dataset.createVariable("trajectory", "i4", ("trajectory",))
        self.trajectory.long_name = "Unique identifier for each particle"
        self.trajectory.cf_role = "trajectory_id"
        self.trajectory[:] = np.arange(particleset.size, dtype=np.int32)

        # Create time, lat, lon and z variables according to CF conventions:
        self.time = self.dataset.createVariable("time", "f8", ("trajectory", "obs"), fill_value=np.nan)
        self.time.long_name = ""
        self.time.standard_name = "time"
        if particleset.time_origin == 0:
            self.time.units = "seconds"
        else:
            self.time.units = "seconds since " + str(particleset.time_origin)
            self.time.calendar = "julian"
        self.time.axis = "T"

        self.lat = self.dataset.createVariable("lat", "f4", ("trajectory", "obs"), fill_value=np.nan)
        self.lat.long_name = ""
        self.lat.standard_name = "latitude"
        self.lat.units = "degrees_north"
        self.lat.axis = "Y"

        self.lon = self.dataset.createVariable("lon", "f4", ("trajectory", "obs"), fill_value=np.nan)
        self.lon.long_name = ""
        self.lon.standard_name = "longitude"
        self.lon.units = "degrees_east"
        self.lon.axis = "X"

        self.z = self.dataset.createVariable("z", "f4", ("trajectory", "obs"), fill_value=np.nan)
        self.z.long_name = ""
        self.z.standard_name = "depth"
        self.z.units = "m"
        self.z.positive = "down"

        if particleset.ptype.user_vars is not None:
            self.user_vars = particleset.ptype.user_vars.keys()
            for var in self.user_vars:
                setattr(self, var, self.dataset.createVariable(var, "f4", ("trajectory", "obs"), fill_value=0.))
                getattr(self, var).long_name = ""
                getattr(self, var).standard_name = var
                getattr(self, var).units = "unknown"
        else:
            self.user_vars = {}

        self.idx = 0

        if initial_dump:
            self.write(particleset, 0.)

    def __del__(self):
        self.dataset.close()

    def write(self, pset, time):
        """Write particle set data to file"""
        self.time[:, self.idx] = time
        self.lat[:, self.idx] = np.array([p.lat for p in pset])
        self.lon[:, self.idx] = np.array([p.lon for p in pset])
        self.z[:, self.idx] = np.zeros(pset.size, dtype=np.float32)
        for var in self.user_vars:
            getattr(self, var)[:, self.idx] = np.array([getattr(p, var) for p in pset])

        self.idx += 1

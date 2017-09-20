"""Module controlling the writing of ParticleSets to NetCDF file"""
import numpy as np
import netCDF4
from datetime import timedelta as delta
from parcels.loggers import logger


__all__ = ['ParticleFile']


class ParticleFile(object):
    """Initialise netCDF4.Dataset for trajectory output.

    The output follows the format outlined in the Discrete Sampling Geometries
    section of the CF-conventions:
    http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#discrete-sampling-geometries

    The current implementation is based on the NCEI template:
    http://www.nodc.noaa.gov/data/formats/netcdf/v2.0/trajectoryIncomplete.cdl

    Both 'Orthogonal multidimensional array' and 'Indexed ragged array' representation
    are implemented. The former is simpler to post-process, but the latter is required
    when particles will be added during the .execute (i.e. the number of particles in
    the pset increases).

    Developer note: We cannot use xray.Dataset here, since it does not yet allow
    incremental writes to disk: https://github.com/pydata/xarray/issues/199

    :param name: Basename of the output file
    :param particleset: ParticleSet to output
    :param user_vars: A list of additional user defined particle variables to write
    :param type: Either 'array' for default matrix style, or 'indexed' for indexed ragged array
    """

    def __init__(self, name, particleset, type='array'):

        self.type = type
        self.name = name
        self.lasttime_written = None  # variable to check if time has been written already
        self.dataset = netCDF4.Dataset("%s.nc" % name, "w", format="NETCDF4")
        self.dataset.createDimension("obs", None)
        if self.type is 'array':
            self.dataset.createDimension("trajectory", particleset.size)
            coords = ("trajectory", "obs")
        elif self.type is 'indexed':
            coords = ("obs")
        else:
            raise RuntimeError("ParticleFile type must be either 'array' or 'indexed'")
        self.dataset.feature_type = "trajectory"
        self.dataset.Conventions = "CF-1.6/CF-1.7"
        self.dataset.ncei_template_version = "NCEI_NetCDF_Trajectory_Template_v2.0"

        # Create ID variable according to CF conventions
        if self.type is 'array':
            self.id = self.dataset.createVariable("trajectory", "i4", ("trajectory",))
            self.id.long_name = "Unique identifier for each particle"
            self.id.cf_role = "trajectory_id"
            self.id[:] = np.array([p.id for p in particleset])
        elif self.type is 'indexed':
            self.id = self.dataset.createVariable("trajectory", "i4", ("obs",))
            self.id.long_name = "index of trajectory this obs belongs to"

        # Create time, lat, lon and z variables according to CF conventions:
        self.time = self.dataset.createVariable("time", "f8", coords, fill_value=np.nan)
        self.time.long_name = ""
        self.time.standard_name = "time"
        if particleset.time_origin == 0:
            self.time.units = "seconds"
        else:
            self.time.units = "seconds since " + str(particleset.time_origin)
            self.time.calendar = "julian"
        self.time.axis = "T"

        self.lat = self.dataset.createVariable("lat", "f4", coords, fill_value=np.nan)
        self.lat.long_name = ""
        self.lat.standard_name = "latitude"
        self.lat.units = "degrees_north"
        self.lat.axis = "Y"

        self.lon = self.dataset.createVariable("lon", "f4", coords, fill_value=np.nan)
        self.lon.long_name = ""
        self.lon.standard_name = "longitude"
        self.lon.units = "degrees_east"
        self.lon.axis = "X"

        self.z = self.dataset.createVariable("z", "f4", coords, fill_value=np.nan)
        self.z.long_name = ""
        self.z.standard_name = "depth"
        self.z.units = "m"
        self.z.positive = "down"

        self.user_vars = []
        for v in particleset.ptype.variables:
            if v.name in ['time', 'lat', 'lon', 'depth', 'z', 'id']:
                continue
            if v.to_write is True:
                setattr(self, v.name, self.dataset.createVariable(v.name, "f4", coords, fill_value=np.nan))
                getattr(self, v.name).long_name = ""
                getattr(self, v.name).standard_name = v.name
                getattr(self, v.name).units = "unknown"
                self.user_vars += [v.name]

        self.idx = 0

    def __del__(self):
        self.dataset.close()

    def sync(self):
        """Write all buffered data to disk"""
        self.dataset.sync()

    def write(self, pset, time, sync=True):
        """Write :class:`parcels.particleset.ParticleSet` data to file

        :param pset: ParticleSet object to write
        :param time: Time at which to write ParticleSet
        :param sync: Optional argument whether to write data to disk immediately. Default is True

        """
        if isinstance(time, delta):
            time = time.total_seconds()
        if self.lasttime_written != time:  # only write if 'time' hasn't been written yet
            self.lasttime_written = time
            if self.type is 'array':
                # Check if largest particle ID is smaller than the last ID in ParticleFile.
                # Otherwise, new particles have been added and netcdf will fail
                if pset.size > 0:
                    if max([p.id for p in pset]) > self.id[-1]:
                        logger.error("Number of particles appears to increase. Use type='indexed' for ParticleFile")

                    # Finds the indices (inds) of the particle IDs in the ParticleFile,
                    # because particles can have been deleted
                    pids = [p.id for p in pset]
                    inds = np.in1d(self.id[:], pids, assume_unique=True)
                    inds = np.arange(len(self.id[:]))[inds]

                    self.time[inds, self.idx] = time
                    self.lat[inds, self.idx] = np.array([p.lat for p in pset])
                    self.lon[inds, self.idx] = np.array([p.lon for p in pset])
                    self.z[inds, self.idx] = np.array([p.depth for p in pset])
                    for var in self.user_vars:
                        getattr(self, var)[inds, self.idx] = np.array([getattr(p, var) for p in pset])
                else:
                    logger.warning("ParticleSet is empty on writing as array")

                self.idx += 1
            elif self.type is 'indexed':
                ind = np.arange(pset.size) + self.idx
                self.id[ind] = np.array([p.id for p in pset])
                self.time[ind] = time
                self.lat[ind] = np.array([p.lat for p in pset])
                self.lon[ind] = np.array([p.lon for p in pset])
                self.z[ind] = np.array([p.depth for p in pset])
                for var in self.user_vars:
                    getattr(self, var)[ind] = np.array([getattr(p, var) for p in pset])

                self.idx += pset.size

        if sync:
            self.sync()

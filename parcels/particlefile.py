"""Module controlling the writing of ParticleSets to NetCDF file"""
import numpy as np
import netCDF4
from datetime import timedelta as delta
from parcels.tools.loggers import logger
from parcels.tools.error import ErrorCode
from os import path
try:
    from parcels._version import version as parcels_version
except:
    raise EnvironmentError('Parcels version can not be retrieved. Have you run ''python setup.py install''?')


__all__ = ['ParticleFile']


def _is_particle_started_yet(particle, time):
    """We don't want to write a particle that is not started yet.
    Particle will be written if:
      * particle.time is equal to time argument of pfile.write()
      * particle.time is before time (in case particle was deleted between previous export and current one)
    """
    return (particle.dt*particle.time <= particle.dt*time or np.isclose(particle.time, time))


def _set_calendar(origin_calendar):
    if origin_calendar == 'np_datetime64':
        return 'standard'
    else:
        return origin_calendar


class ParticleFile(object):
    """Initialise netCDF4.Dataset for trajectory output.

    The output follows the format outlined in the Discrete Sampling Geometries
    section of the CF-conventions:
    http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#discrete-sampling-geometries

    The current implementation is based on the NCEI template:
    http://www.nodc.noaa.gov/data/formats/netcdf/v2.0/trajectoryIncomplete.cdl

    Developer note: We cannot use xray.Dataset here, since it does not yet allow
    incremental writes to disk: https://github.com/pydata/xarray/issues/199

    :param name: Basename of the output file
    :param particleset: ParticleSet to output
    :param outputdt: Interval which dictates the update frequency of file output
                     while ParticleFile is given as an argument of ParticleSet.execute()
                     It is either a timedelta object or a positive double.
    :param write_ondelete: Boolean to write particle data only when they are deleted. Default is False
    :param chunksizes: 2d vector of sizes for NetCDF chunking. Lower values means smaller files, but also slower IO.
                       See e.g. https://www.unidata.ucar.edu/blogs/developer/entry/chunking_data_choosing_shapes
    """

    def __init__(self, name, particleset, outputdt=np.infty, write_ondelete=False, chunksizes=None):

        self.name = name
        self.chunksizes = [max([len(particleset), 1]), 1] if chunksizes is None else chunksizes
        self.write_ondelete = write_ondelete
        self.outputdt = outputdt
        self.lasttraj = 0  # id of last particle written
        self.lasttime_written = None  # variable to check if time has been written already

        self.dataset = None
        self.particleset = particleset

    def open_dataset(self):
        extension = path.splitext(str(self.name))[1]
        fname = self.name if extension in ['.nc', '.nc4'] else "%s.nc" % self.name
        self.dataset = netCDF4.Dataset(fname, "w", format="NETCDF4")
        self.dataset.createDimension("obs", None)
        self.dataset.createDimension("traj", None)
        coords = ("traj", "obs")
        self.dataset.feature_type = "trajectory"
        self.dataset.Conventions = "CF-1.6/CF-1.7"
        self.dataset.ncei_template_version = "NCEI_NetCDF_Trajectory_Template_v2.0"
        self.dataset.parcels_version = parcels_version
        self.dataset.parcels_mesh = self.particleset.fieldset.gridset.grids[0].mesh

        # Create ID variable according to CF conventions
        self.id = self.dataset.createVariable("trajectory", "i4", coords, chunksizes=self.chunksizes)
        self.id.long_name = "Unique identifier for each particle"
        self.id.cf_role = "trajectory_id"

        # Create time, lat, lon and z variables according to CF conventions:
        self.time = self.dataset.createVariable("time", "f8", coords, fill_value=np.nan, chunksizes=self.chunksizes)
        self.time.long_name = ""
        self.time.standard_name = "time"
        if self.particleset.time_origin.calendar is None:
            self.time.units = "seconds"
        else:
            self.time.units = "seconds since " + str(self.particleset.time_origin)
            self.time.calendar = _set_calendar(self.particleset.time_origin.calendar)
        self.time.axis = "T"

        if self.particleset.lonlatdepth_dtype is np.float64:
            lonlatdepth_precision = "f8"
        else:
            lonlatdepth_precision = "f4"

        self.lat = self.dataset.createVariable("lat", lonlatdepth_precision, coords, fill_value=np.nan, chunksizes=self.chunksizes)
        self.lat.long_name = ""
        self.lat.standard_name = "latitude"
        self.lat.units = "degrees_north"
        self.lat.axis = "Y"

        self.lon = self.dataset.createVariable("lon", lonlatdepth_precision, coords, fill_value=np.nan, chunksizes=self.chunksizes)
        self.lon.long_name = ""
        self.lon.standard_name = "longitude"
        self.lon.units = "degrees_east"
        self.lon.axis = "X"

        self.z = self.dataset.createVariable("z", lonlatdepth_precision, coords, fill_value=np.nan, chunksizes=self.chunksizes)
        self.z.long_name = ""
        self.z.standard_name = "depth"
        self.z.units = "m"
        self.z.positive = "down"

        self.user_vars = []
        self.user_vars_once = []
        """
        :user_vars: list of additional user defined particle variables to write for all particles and all times
        :user_vars_once: list of additional user defined particle variables to write for all particles only once at initial time.
        """

        for v in self.particleset.ptype.variables:
            if v.name in ['time', 'lat', 'lon', 'depth', 'z', 'id']:
                continue
            if v.to_write:
                if v.to_write is True:
                    setattr(self, v.name, self.dataset.createVariable(v.name, "f4", coords, fill_value=np.nan, chunksizes=self.chunksizes))
                    self.user_vars += [v.name]
                elif v.to_write == 'once':
                    setattr(self, v.name, self.dataset.createVariable(v.name, "f4", "traj", fill_value=np.nan, chunksizes=[self.chunksizes[0]]))
                    self.user_vars_once += [v.name]
                getattr(self, v.name).long_name = ""
                getattr(self, v.name).standard_name = v.name
                getattr(self, v.name).units = "unknown"

        self.idx = np.empty(shape=0)

    def __del__(self):
        if self.dataset:
            self.dataset.close()

    def sync(self):
        """Write all buffered data to disk"""
        self.dataset.sync()

    def add_metadata(self, name, message):
        """Add metadata to :class:`parcels.particleset.ParticleSet`
        :param name: Name of the metadata variabale
        :param message: message to be written
        """
        if self.dataset is None:
            self.open_dataset()
        setattr(self.dataset, name, message)

    def write(self, pset, time, sync=True, deleted_only=False):
        """Write :class:`parcels.particleset.ParticleSet` data to file

        :param pset: ParticleSet object to write
        :param time: Time at which to write ParticleSet
        :param sync: Optional argument whether to write data to disk immediately. Default is True

        """
        if self.dataset is None:
            self.open_dataset()
        if isinstance(time, delta):
            time = time.total_seconds()
        if self.lasttime_written != time and \
           (self.write_ondelete is False or deleted_only is True):
            if pset.size > 0:

                first_write = [p for p in pset if (p.fileid < 0 or len(self.idx) == 0) and _is_particle_started_yet(p, time)]  # len(self.idx)==0 in case pset is written to new ParticleFile
                for p in first_write:
                    p.fileid = self.lasttraj  # particle id in current file
                    self.lasttraj += 1

                self.idx = np.append(self.idx, np.zeros(len(first_write)))

                for p in pset:
                    if _is_particle_started_yet(p, time):
                        i = p.fileid
                        self.id[i, self.idx[i]] = p.id
                        self.time[i, self.idx[i]] = p.time
                        self.lat[i, self.idx[i]] = p.lat
                        self.lon[i, self.idx[i]] = p.lon
                        self.z[i, self.idx[i]] = p.depth
                        for var in self.user_vars:
                            getattr(self, var)[i, self.idx[i]] = getattr(p, var)
                        if p.state != ErrorCode.Delete and not np.allclose(p.time, time):
                            logger.warning_once('time argument in pfile.write() is %g, but a particle has time %g.' % (time, p.time))

                for p in first_write:
                    for var in self.user_vars_once:
                        getattr(self, var)[p.fileid] = getattr(p, var)
            else:
                logger.warning("ParticleSet is empty on writing as array at time %g" % time)

            if not deleted_only:
                self.idx += 1
                self.lasttime_written = time

        if sync:
            self.sync()

"""Module controlling the writing of ParticleSets to NetCDF file"""
import numpy as np
import netCDF4
from datetime import timedelta as delta
from parcels.tools.loggers import logger
import os
import psutil
import shutil
import string
import random
from parcels.tools.error import ErrorCode
try:
    from parcels._version import version as parcels_version
except:
    raise EnvironmentError('Parcels version can not be retrieved. Have you run ''python setup.py install''?')
try:
    from os import getuid
except:
    # Windows does not have getuid(), so define to simply return 'tmp'
    def getuid():
        return 'tmp'


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
    """Initialise trajectory output.
    :param name: Basename of the output file
    :param particleset: ParticleSet to output
    :param outputdt: Interval which dictates the update frequency of file output
                     while ParticleFile is given as an argument of ParticleSet.execute()
                     It is either a timedelta object or a positive double.
    :param write_ondelete: Boolean to write particle data only when they are deleted. Default is False
    :param tempwritedir: directory to write temporary files to during executing.
                        Default is out-XXXXXX where Xs are random capitals
    :param pset_info: dictionary of info on the ParticleSet, stored in tempwritedir/pset_info.npy,
                      used to create NetCDF file from npy-files.
    """

    def __init__(self, name, particleset, outputdt=np.infty, write_ondelete=False, tempwritedir=None, pset_info=None):

        self.write_ondelete = write_ondelete
        self.outputdt = outputdt
        self.lasttime_written = None  # variable to check if time has been written already

        self.dataset = None
        self.metadata = {}
        if pset_info is not None:
            for v in pset_info.keys():
                setattr(self, v, pset_info[v])
        else:
            self.name = name
            self.particleset = particleset
            self.parcels_mesh = self.particleset.fieldset.gridset.grids[0].mesh
            self.time_origin = self.particleset.time_origin
            self.lonlatdepth_dtype = self.particleset.lonlatdepth_dtype
            self.var_names = []
            self.var_names_once = []
            for v in self.particleset.ptype.variables:
                if v.to_write == 'once':
                    self.var_names_once += [v.name]
                elif v.to_write is True:
                    self.var_names += [v.name]
            if len(self.var_names_once) > 0:
                self.written_once = []
                self.file_list_once = []

            self.file_list = []
            self.time_written = []
            self.maxid_written = -1
        self.to_export = False

        if tempwritedir is None:
            self.tempwritedir = os.path.join(os.path.dirname(str(self.name)), "out-%s" % ''.join(random.choice(string.ascii_uppercase) for _ in range(8)))
        else:
            self.tempwritedir = tempwritedir

        self.delete_tempwritedir()

    def open_netcdf_file(self, data_shape):
        """Initialise NetCDF4.Dataset for trajectory output.
        The output follows the format outlined in the Discrete Sampling Geometries
        section of the CF-conventions:
        http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#discrete-sampling-geometries
        The current implementation is based on the NCEI template:
        http://www.nodc.noaa.gov/data/formats/netcdf/v2.0/trajectoryIncomplete.cdl
        :param data_shape: shape of the variables in the NetCDF4 file
        """
        extension = os.path.splitext(str(self.name))[1]
        fname = self.name if extension in ['.nc', '.nc4'] else "%s.nc" % self.name
        if os.path.exists(str(fname)):
            os.remove(str(fname))
        self.dataset = netCDF4.Dataset(fname, "w", format="NETCDF4")
        self.dataset.createDimension("obs", data_shape[1])
        self.dataset.createDimension("traj", data_shape[0])
        coords = ("traj", "obs")
        self.dataset.feature_type = "trajectory"
        self.dataset.Conventions = "CF-1.6/CF-1.7"
        self.dataset.ncei_template_version = "NCEI_NetCDF_Trajectory_Template_v2.0"
        self.dataset.parcels_version = parcels_version
        self.dataset.parcels_mesh = self.parcels_mesh

        # Create ID variable according to CF conventions
        self.id = self.dataset.createVariable("trajectory", "i4", coords, fill_value=-2147483647, chunksizes=data_shape)
        self.id.long_name = "Unique identifier for each particle"
        self.id.cf_role = "trajectory_id"

        # Create time, lat, lon and z variables according to CF conventions:
        self.time = self.dataset.createVariable("time", "f8", coords, fill_value=np.nan, chunksizes=data_shape)
        self.time.long_name = ""
        self.time.standard_name = "time"
        if self.time_origin.calendar is None:
            self.time.units = "seconds"
        else:
            self.time.units = "seconds since " + str(self.time_origin)
            self.time.calendar = 'standard' if self.time_origin.calendar == 'np_datetime64' else self.time_origin.calendar
        self.time.axis = "T"

        if self.lonlatdepth_dtype is np.float64:
            lonlatdepth_precision = "f8"
        else:
            lonlatdepth_precision = "f4"

        self.lat = self.dataset.createVariable("lat", lonlatdepth_precision, coords, fill_value=np.nan, chunksizes=data_shape)
        self.lat.long_name = ""
        self.lat.standard_name = "latitude"
        self.lat.units = "degrees_north"
        self.lat.axis = "Y"

        self.lon = self.dataset.createVariable("lon", lonlatdepth_precision, coords, fill_value=np.nan, chunksizes=data_shape)
        self.lon.long_name = ""
        self.lon.standard_name = "longitude"
        self.lon.units = "degrees_east"
        self.lon.axis = "X"

        self.z = self.dataset.createVariable("z", lonlatdepth_precision, coords, fill_value=np.nan, chunksizes=data_shape)
        self.z.long_name = ""
        self.z.standard_name = "depth"
        self.z.units = "m"
        self.z.positive = "down"

        for vname in self.var_names:
            if vname not in ['time', 'lat', 'lon', 'depth', 'id']:
                setattr(self, vname, self.dataset.createVariable(vname, "f4", coords, fill_value=np.nan, chunksizes=data_shape))
                getattr(self, vname).long_name = ""
                getattr(self, vname).standard_name = vname
                getattr(self, vname).units = "unknown"

        for vname in self.var_names_once:
            setattr(self, vname, self.dataset.createVariable(vname, "f4", "traj", fill_value=np.nan, chunksizes=[data_shape[0]]))
            getattr(self, vname).long_name = ""
            getattr(self, vname).standard_name = vname
            getattr(self, vname).units = "unknown"

        for name, message in self.metadata.items():
            setattr(self.dataset, name, message)

    def __del__(self):
        if self.to_export:
            self.close()

    def close(self):
        """Close the ParticleFile object by exporting and then deleting
        the temporary npy files"""
        self.export()
        self.delete_tempwritedir()
        self.dataset.close()
        self.to_export = False

    def add_metadata(self, name, message):
        """Add metadata to :class:`parcels.particleset.ParticleSet`
        :param name: Name of the metadata variabale
        :param message: message to be written
        """
        if self.dataset is None:
            self.metadata[name] = message
        else:
            setattr(self.dataset, name, message)

    def convert_pset_to_dict(self, pset, time, deleted_only=False):
        """Convert all Particle data from one time step to a python dictionary.
        :param pset: ParticleSet object to write
        :param time: Time at which to write ParticleSet
        :param deleted_only: Flag to write only the deleted Particles
        returns two dictionaries: one for all variables to be written each outputdt,
         and one for all variables to be written once
        """
        data_dict = {}
        data_dict_once = {}

        time = time.total_seconds() if isinstance(time, delta) else time

        if self.lasttime_written != time and \
           (self.write_ondelete is False or deleted_only is True):
            if pset.size == 0:
                logger.warning("ParticleSet is empty on writing as array at time %g" % time)
            else:
                for var in self.var_names:
                    data_dict[var] = np.nan * np.zeros(len(pset))

                i = 0
                for p in pset:
                    if p.dt*p.time <= p.dt*time:
                        for var in self.var_names:
                            data_dict[var][i] = getattr(p, var)
                        if p.state != ErrorCode.Delete and not np.allclose(p.time, time):
                            logger.warning_once('time argument in pfile.write() is %g, but a particle has time %g.' % (time, p.time))
                        self.maxid_written = np.max([self.maxid_written, p.id])
                        i += 1

                save_ind = np.isfinite(data_dict["id"])
                for key in self.var_names:
                    data_dict[key] = data_dict[key][save_ind]

                if time not in self.time_written:
                    self.time_written.append(time)

                if len(self.var_names_once) > 0:
                    first_write = [p for p in pset if (p.id not in self.written_once) and _is_particle_started_yet(p, time)]
                    data_dict_once['id'] = np.nan * np.zeros(len(first_write))
                    for var in self.var_names_once:
                        data_dict_once[var] = np.nan * np.zeros(len(first_write))

                    i = 0
                    for p in first_write:
                        self.written_once.append(p.id)
                        data_dict_once['id'][i] = p.id
                        for var in self.var_names_once:
                            data_dict_once[var][i] = getattr(p, var)
                        i += 1

            if not deleted_only:
                self.lasttime_written = time

        return data_dict, data_dict_once

    def dump_dict_to_npy(self, data_dict, data_dict_once):
        """Buffer data to set of temporary numpy files, using np.save"""

        if not os.path.exists(self.tempwritedir):
            os.mkdir(self.tempwritedir)

        if len(data_dict) > 0:
            tmpfilename = os.path.join(self.tempwritedir, str(len(self.file_list) + 1) + ".npy")
            with open(tmpfilename, 'wb') as f:
                np.save(f, data_dict)
            self.file_list.append(tmpfilename)

        if len(data_dict_once) > 0:
            tmpfilename = os.path.join(self.tempwritedir, str(len(self.file_list) + 1) + '_once.npy')
            with open(tmpfilename, 'wb') as f:
                np.save(f, data_dict_once)
            self.file_list_once.append(tmpfilename)

        self.to_export = True

    def dump_psetinfo_to_npy(self):
        pset_info = {}
        attrs_to_dump = ['name', 'var_names', 'var_names_once', 'time_origin', 'lonlatdepth_dtype',
                         'file_list', 'file_list_once', 'maxid_written', 'time_written', 'parcels_mesh',
                         'metadata']
        for a in attrs_to_dump:
            if hasattr(self, a):
                pset_info[a] = getattr(self, a)
        with open(os.path.join(self.tempwritedir, 'pset_info.npy'), 'wb') as f:
            np.save(f, pset_info)

    def write(self, pset, time, deleted_only=False):
        """Write all data from one time step to a temporary npy-file
        using a python dictionary. The data is saved in the folder 'out'.
        :param pset: ParticleSet object to write
        :param time: Time at which to write ParticleSet
        :param deleted_only: Flag to write only the deleted Particles
        """

        data_dict, data_dict_once = self.convert_pset_to_dict(pset, time, deleted_only=deleted_only)
        self.dump_dict_to_npy(data_dict, data_dict_once)
        self.dump_psetinfo_to_npy()

    def read_from_npy(self, file_list, time_steps, var):
        """Read NPY-files for one variable using a loop over all files.
        :param file_list: List that  contains all file names in the output directory
        :param time_steps: Number of time steps that were written in out directory
        :param var: name of the variable to read
        """

        data = np.nan * np.zeros((self.maxid_written+1, time_steps))
        time_index = np.zeros(self.maxid_written+1, dtype=int)
        t_ind_used = np.zeros(time_steps, dtype=int)

        # loop over all files
        for npyfile in file_list:
            data_dict = np.load(npyfile, allow_pickle=True).item()
            id_ind = np.array(data_dict["id"], dtype=int)
            t_ind = time_index[id_ind] if 'once' not in file_list[0] else 0
            t_ind_used[t_ind] = 1
            data[id_ind, t_ind] = data_dict[var]
            time_index[id_ind] = time_index[id_ind] + 1

        # remove rows and columns that are completely filled with nan values
        tmp = data[time_index > 0, :]
        return tmp[:, t_ind_used == 1]

    def export(self):
        """Exports outputs in temporary NPY-files to NetCDF file"""
        memory_estimate_total = self.maxid_written+1 * len(self.time_written) * 8
        if memory_estimate_total > 0.9*psutil.virtual_memory().available:
            raise MemoryError("Not enough memory available for export. npy files are stored at %s", self.npy_path)

        for var in self.var_names:
            data = self.read_from_npy(self.file_list, len(self.time_written), var)
            if var == self.var_names[0]:
                self.open_netcdf_file(data.shape)
            varout = 'z' if var == 'depth' else var
            getattr(self, varout)[:, :] = data

        if len(self.var_names_once) > 0:
            for var in self.var_names_once:
                getattr(self, var)[:] = self.read_from_npy(self.file_list_once, 1, var)

    def delete_tempwritedir(self):
        """Deleted all temporary npy files"""
        if os.path.exists(self.tempwritedir):
            shutil.rmtree(self.tempwritedir)

"""Module controlling the writing of ParticleSets to NetCDF file"""
import os
from glob import glob
import numpy as np

try:
    from mpi4py import MPI
except:
    MPI = None

from parcels.particlefile.baseparticlefile import BaseParticleFile

__all__ = ['ParticleFileNodes']


class ParticleFileNodes(BaseParticleFile):
    """Initialise trajectory output.

    :param name: Basename of the output file
    :param particleset: ParticleSet to output
    :param outputdt: Interval which dictates the update frequency of file output
                     while ParticleFile is given as an argument of ParticleSet.execute()
                     It is either a timedelta object or a positive double.
    :param write_ondelete: Boolean to write particle data only when they are deleted. Default is False
    :param convert_at_end: Boolean to convert npy files to netcdf at end of run. Default is True
    :param tempwritedir: directories to write temporary files to during executing.
                     Default is out-XXXXXX where Xs are random capitals. Files for individual
                     processors are written to subdirectories 0, 1, 2 etc under tempwritedir
    :param pset_info: dictionary of info on the ParticleSet, stored in tempwritedir/XX/pset_info.npy,
                     used to create NetCDF file from npy-files.
    """
    max_index_written = -1

    def __init__(self, name, particleset, outputdt=np.infty, write_ondelete=False, convert_at_end=True,
                 tempwritedir=None, pset_info=None):
        """
        ParticleFileNodes - Constructor
        :param name: Basename of the output file
        :param particleset: ParticleSet to output
        :param outputdt: Interval which dictates the update frequency of file output
                         while ParticleFile is given as an argument of ParticleSet.execute()
                         It is either a timedelta object or a positive double.
        :param write_ondelete: Boolean to write particle data only when they are deleted. Default is False
        :param convert_at_end: Boolean to convert npy files to netcdf at end of run. Default is True
        :param tempwritedir: directories to write temporary files to during executing.
                         Default is out-XXXXXX where Xs are random capitals. Files for individual
                         processors are written to subdirectories 0, 1, 2 etc under tempwritedir
        :param pset_info: dictionary of info on the ParticleSet, stored in tempwritedir/XX/pset_info.npy,
                         used to create NetCDF file from npy-files.
        """
        super(ParticleFileNodes, self).__init__(name=name, particleset=particleset, outputdt=outputdt,
                                                write_ondelete=write_ondelete, convert_at_end=convert_at_end,
                                                tempwritedir=tempwritedir, pset_info=pset_info)
        self.var_names.append('index')
        self.max_index_written = 0
        self.time_written = []

    def __del__(self):
        """
        ParticleFileNodes - Destructor
        """
        super(ParticleFileNodes, self).__del__()

    def _reserved_var_names(self):
        """
        :returns the reserved dimension names not to be written just once.
        """
        return ['time', 'lat', 'lon', 'depth', 'id', 'index']

    def _create_trajectory_records(self, coords):
        """
        This function creates the NetCDF record of the ParticleSet inside the output NetCDF file
        :arg coords: tuple of dictionary keys for # entities ("traj(ectories)") and timesteps ("obs(ervations)")
        """
        super(ParticleFileNodes, self)._create_trajectory_records(coords=coords)

    def _create_variable_records(self, coords):
        """
        creates the NetCDF record structure for the (user-defined) variables of a trajectory.
        :arg coords: tuple of dictionary keys for # entities ("traj(ectories)") and timesteps ("obs(ervations)")
        """

        self.index = self.dataset.createVariable("index", "i4", coords, fill_value=-2**(31))
        self.index.long_name = "running (zero-based continuous) indexing element referring to the LOCAL index of a particle within one timestep"

        for vname in self.var_names:
            if vname not in self._reserved_var_names():
                # hm, shouldn't that be adaptive instead of fixed to "f4" (e.g. "f8") ? I think I looked that up already once and it violates the CF convention, sadly
                setattr(self, vname, self.dataset.createVariable(vname, "f4", coords, fill_value=np.nan))
                getattr(self, vname).long_name = ""
                getattr(self, vname).standard_name = vname
                getattr(self, vname).units = "unknown"

        for vname in self.var_names_once:
            # hm, shouldn't that be adaptive instead of fixed to "f4" (e.g. "f8") ? I think I looked that up already once and it violates the CF convention, sadly
            setattr(self, vname, self.dataset.createVariable(vname, "f4", "traj", fill_value=np.nan))
            getattr(self, vname).long_name = ""
            getattr(self, vname).standard_name = vname
            getattr(self, vname).units = "unknown"

    def _create_trajectory_records_old(self, coords):
        """
        This function creates the NetCDF record of the ParticleSet inside the output NetCDF file
        :arg coords: tuple of dictionary keys for # entities ("traj(ectories)") and timesteps ("obs(ervations)")
        """
        # Create ID variable according to CF conventions
        self.id = self.dataset.createVariable("trajectory", "i8", coords, fill_value=-2**(63))  # minint64 fill_value
        self.id.long_name = "Unique identifier for each particle"
        self.id.cf_role = "trajectory_id"

        self.index = self.dataset.createVariable("index", "i4", coords, fill_value=-2**(31))
        self.index.long_name = "running (zero-based continuous) indexing element referring to the LOCAL index of a particle within one timestep"

        # Create time, lat, lon and z variables according to CF conventions:
        self.time = self.dataset.createVariable("time", "f8", coords, fill_value=np.nan)
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

        if ('lat' in self.var_names):
            self.lat = self.dataset.createVariable("lat", lonlatdepth_precision, coords, fill_value=np.nan)
            self.lat.long_name = ""
            self.lat.standard_name = "latitude"
            self.lat.units = "degrees_north"
            self.lat.axis = "Y"

        if ('lon' in self.var_names):
            self.lon = self.dataset.createVariable("lon", lonlatdepth_precision, coords, fill_value=np.nan)
            self.lon.long_name = ""
            self.lon.standard_name = "longitude"
            self.lon.units = "degrees_east"
            self.lon.axis = "X"

        if ('depth' in self.var_names) or ('z' in self.var_names):
            self.z = self.dataset.createVariable("z", lonlatdepth_precision, coords, fill_value=np.nan)
            self.z.long_name = ""
            self.z.standard_name = "depth"
            self.z.units = "m"
            self.z.positive = "down"

        for vname in self.var_names:
            if vname not in self._reserved_var_names():
                # hm, shouldn't that be adaptive instead of fixed to "f4" (e.g. "f8") ? I think I looked that up already once and it violates the CF convention, sadly
                setattr(self, vname, self.dataset.createVariable(vname, "f4", coords, fill_value=np.nan))
                getattr(self, vname).long_name = ""
                getattr(self, vname).standard_name = vname
                getattr(self, vname).units = "unknown"

        for vname in self.var_names_once:
            # hm, shouldn't that be adaptive instead of fixed to "f4" (e.g. "f8") ? I think I looked that up already once and it violates the CF convention, sadly
            setattr(self, vname, self.dataset.createVariable(vname, "f4", "traj", fill_value=np.nan))
            getattr(self, vname).long_name = ""
            getattr(self, vname).standard_name = vname
            getattr(self, vname).units = "unknown"

    def get_pset_info_attributes(self):
        """
        :returns the main attributes of the pset_info.npy file.

        Attention:
        For ParticleSet structures other than SoA, and structures where ID != index, this has to be overridden.
        """
        attributes = ['name', 'var_names', 'var_names_once', 'time_origin', 'lonlatdepth_dtype',
                      'file_list', 'file_list_once', 'max_index_written', 'time_written', 'parcels_mesh',
                      'metadata']
        return attributes

    def read_from_npy(self, file_list, time_steps, var):
        """
        Read NPY-files for one variable using a loop over all files. This differs from indexable structures,
        as here we count the max_index_written, not the maxid_written

        :param file_list: List that  contains all file names in the output directory
        :param time_steps: Number of time steps that were written in out directory
        :param var: name of the variable to read
        """

        data = np.nan * np.zeros((self.max_index_written+1, time_steps))
        time_index = np.zeros(self.max_index_written+1, dtype=np.int64)
        t_ind_used = np.zeros(time_steps, dtype=np.int64)

        # loop over all files
        for npyfile in file_list:
            try:
                data_dict = np.load(npyfile, allow_pickle=True).item()
            except NameError:
                raise RuntimeError('Cannot combine npy files into netcdf file because your ParticleFile is '
                                   'still open on interpreter shutdown.\nYou can use '
                                   '"parcels_convert_npydir_to_netcdf %s" to convert these to '
                                   'a NetCDF file yourself.\nTo avoid this error, make sure you '
                                   'close() your ParticleFile at the end of your script.' % self.tempwritedir)
            id_ind = np.array(data_dict['index'])
            t_ind = time_index[id_ind] if 'once' not in file_list[0] else 0
            data[id_ind, t_ind] = data_dict[var]
            time_index[id_ind] = time_index[id_ind] + 1
            t_ind_used[t_ind] = 1

        # remove rows and columns that are completely filled with nan values
        tmp = data[time_index > 0, :]
        return tmp[:, t_ind_used == 1]

    def export(self):
        """
        Exports outputs in temporary NPY-files to NetCDF file

        Attention:
        For ParticleSet structures other than SoA, and structures where ID != index, this has to be overridden.
        """
        if MPI:
            # The export can only start when all threads are done.
            MPI.COMM_WORLD.Barrier()
            if MPI.COMM_WORLD.Get_rank() > 0:
                return  # export only on threat 0

        # Retrieve all temporary writing directories and sort them in numerical order
        temp_names = sorted(glob(os.path.join("%s" % self.tempwritedir_base, "*")),
                            key=lambda x: int(os.path.basename(x)))

        if len(temp_names) == 0:
            raise RuntimeError("No npy files found in %s" % self.tempwritedir_base)

        global_max_index_written = -1
        global_time_written = []
        global_file_list = []
        global_file_list_once = None
        if len(self.var_names_once) > 0:
            global_file_list_once = []
        for tempwritedir in temp_names:
            if os.path.exists(tempwritedir):
                pset_info_local = np.load(os.path.join(tempwritedir, 'pset_info.npy'), allow_pickle=True).item()
                global_max_index_written = np.max([global_max_index_written, pset_info_local['max_index_written']])
                global_time_written += pset_info_local['time_written']
                global_file_list += pset_info_local['file_list']
                if len(self.var_names_once) > 0:
                    global_file_list_once += pset_info_local['file_list_once']
        self.max_index_written = global_max_index_written
        self.time_written = np.unique(global_time_written)

        for var in self.var_names:
            data = self.read_from_npy(global_file_list, len(self.time_written), var)
            if var == self.var_names[0]:
                self.open_netcdf_file(data.shape)
            varout = 'z' if var == 'depth' else var
            getattr(self, varout)[:, :] = data

        if len(self.var_names_once) > 0:
            for var in self.var_names_once:
                getattr(self, var)[:] = self.read_from_npy(global_file_list_once, 1, var)

        self.close_netcdf_file()

"""Module controlling the writing of ParticleSets to NetCDF file"""
import os
from glob import glob
import numpy as np

try:
    from mpi4py import MPI
except:
    MPI = None

from parcels.particlefile.baseparticlefile import BaseParticleFile

__all__ = ['ParticleFileSOA']


class ParticleFileSOA(BaseParticleFile):
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

    def __init__(self, name, particleset, outputdt=np.infty, write_ondelete=False, convert_at_end=True,
                 tempwritedir=None, pset_info=None):
        super(ParticleFileSOA, self).__init__(name=name, particleset=particleset, outputdt=outputdt,
                                              write_ondelete=write_ondelete, convert_at_end=convert_at_end,
                                              tempwritedir=tempwritedir, pset_info=pset_info)

    def __del__(self):
        super(ParticleFileSOA, self).__del__()

    def _reserved_var_names(self):
        """
        returns the reserved dimension names not to be written just once.
        """
        return ['time', 'lat', 'lon', 'depth', 'id']  # , 'index'

    def _create_trajectory_records(self, coords):
        super(ParticleFileSOA, self)._create_trajectory_records(coords=coords)

    def get_pset_info_attributes(self):
        """
        returns the main attributes of the pset_info.npy file.

        Attention:
        For ParticleSet structures other than SoA, and structures where ID != index, this has to be overridden.
        """
        attributes = ['name', 'var_names', 'var_dtypes', 'var_names_once', 'var_dtypes_once',
                      'time_origin', 'lonlatdepth_dtype', 'file_list', 'file_list_once',
                      'parcels_mesh', 'metadata']
        return attributes

    def read_from_npy(self, file_list, n_timesteps, var, dtype):
        """
        Read NPY-files for one variable using a loop over all files.

        Attention:
        For ParticleSet structures other than SoA, and structures where ID != index, this has to be overridden.

        :param file_list: List that  contains all file names in the output directory
        :param n_timesteps: Dictionary with (for each particle) number of time steps that were written in out directory
        :param var: name of the variable to read
        """
        max_timesteps = max(n_timesteps.values()) if n_timesteps.keys() else 0
        fill_value = self.fill_value_map[dtype]
        data = fill_value * np.ones((len(n_timesteps), max_timesteps), dtype=dtype)
        time_index = np.zeros(len(n_timesteps))
        id_index = {}
        count = 0
        for i in sorted(n_timesteps.keys()):
            id_index[i] = count
            count += 1

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
            for ii, i in enumerate(data_dict["id"]):
                id_ind = id_index[i]
                t_ind = int(time_index[id_ind]) if 'once' not in file_list[0] else 0
                data[id_ind, t_ind] = data_dict[var][ii]
                time_index[id_ind] = time_index[id_ind] + 1

        # remove rows and columns that are completely filled with nan values
        return data[time_index > 0, :]

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

        # Create dictionary to translate datatypes and fill_values
        self.fmt_map = {np.float32: 'f4', np.float64: 'f8',
                        np.bool_: 'i1', np.int16: 'i2', np.int32: 'i4', np.int64: 'i8'}
        self.fill_value_map = {np.float32: np.nan, np.float64: np.nan,
                               np.bool_: np.iinfo(np.int8).max, np.int16: np.iinfo(np.int16).max,
                               np.int32: np.iinfo(np.int32).max, np.int64: np.iinfo(np.int64).max}

        # Retrieve all temporary writing directories and sort them in numerical order
        temp_names = sorted(glob(os.path.join("%s" % self.tempwritedir_base, "*")),
                            key=lambda x: int(os.path.basename(x)))

        if len(temp_names) == 0:
            raise RuntimeError("No npy files found in %s" % self.tempwritedir_base)

        n_timesteps = {}
        global_file_list = []
        if len(self.var_names_once) > 0:
            global_file_list_once = []
        for tempwritedir in temp_names:
            if os.path.exists(tempwritedir):
                pset_info_local = np.load(os.path.join(tempwritedir, 'pset_info.npy'), allow_pickle=True).item()
                for npyfile in pset_info_local['file_list']:
                    tmp_dict = np.load(npyfile, allow_pickle=True).item()
                    for i in tmp_dict['id']:
                        if i in n_timesteps:
                            n_timesteps[i] += 1
                        else:
                            n_timesteps[i] = 1
                global_file_list += pset_info_local['file_list']
                if len(self.var_names_once) > 0:
                    global_file_list_once += pset_info_local['file_list_once']

        for var, dtype in zip(self.var_names, self.var_dtypes):
            data = self.read_from_npy(global_file_list, n_timesteps, var, dtype)
            if var == self.var_names[0]:
                self.open_netcdf_file(data.shape)
            varout = 'z' if var == 'depth' else var
            getattr(self, varout)[:, :] = data

        if len(self.var_names_once) > 0:
            n_timesteps_once = {}
            for i in n_timesteps:
                n_timesteps_once[i] = 1
            for var in self.var_names_once:
                getattr(self, var)[:] = self.read_from_npy(global_file_list_once, n_timesteps_once, var, dtype)

        self.close_netcdf_file()

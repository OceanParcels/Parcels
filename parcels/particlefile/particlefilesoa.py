"""Module controlling the writing of ParticleSets to NetCDF file"""
import os
import psutil
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
        attributes = ['name', 'var_names', 'var_names_once', 'time_origin', 'lonlatdepth_dtype',
                      'file_list', 'file_list_once', 'maxid_written', 'parcels_mesh', 'metadata']
        return attributes

    def read_from_npy(self, file_list, time_steps, var, id_range):
        """
        Read NPY-files for one variable using a loop over all files.

        Attention:
        For ParticleSet structures other than SoA, and structures where ID != index, this has to be overridden.

        :param file_list: List that  contains all file names in the output directory
        :param time_steps: Number of time steps that were written in out directory
        :param var: name of the variable to read
        """

        valid_id = np.arange(id_range[0], id_range[1]+1)
        n_ids = len(valid_id)

        data = np.nan * np.zeros((n_ids, time_steps))
        time_index = np.zeros(n_ids, dtype=np.int64)
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

            id_avail = np.array(data_dict["id"], dtype=np.int64)
            id_mask_full = np.in1d(id_avail, valid_id) # which ids in data are present in this chunk
            id_mask_chunk = np.in1d(valid_id, id_avail) # which ids in this chunk are present in data
            t_ind = time_index[id_mask_chunk] if 'once' not in file_list[0] else 0
            t_ind_used[t_ind] = 1
            data[id_mask_chunk, t_ind] = data_dict[var][id_mask_full]
            time_index[id_mask_chunk] = time_index[id_mask_chunk] + 1

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

        global_maxid_written = -1
        global_time_written = []
        global_id = []
        global_file_list = []
        if len(self.var_names_once) > 0:
            global_file_list_once = []
        for tempwritedir in temp_names:
            if os.path.exists(tempwritedir):
                pset_info_local = np.load(os.path.join(tempwritedir, 'pset_info.npy'), allow_pickle=True).item()
                global_maxid_written = np.max([global_maxid_written, pset_info_local['maxid_written']])
                for npyfile in pset_info_local['file_list']:
                    tmp_dict = np.load(npyfile, allow_pickle=True).item()
                    global_time_written.append([t for t in tmp_dict['time']])
                    global_id.append([i for i in tmp_dict['id']])
                global_file_list += pset_info_local['file_list']
                if len(self.var_names_once) > 0:
                    global_file_list_once += pset_info_local['file_list_once']
        self.maxid_written = global_maxid_written

        # These steps seem to be quite expensive...
        self.time_written = np.unique(global_time_written)
        self.id_present = np.unique([pid for frame in global_id for pid in frame])

        for var in self.var_names:
            # Find available memory to check if output file is too large
            avail_mem = psutil.virtual_memory()[1]
            req_mem   = len(self.id_present)*len(self.time_written)*8*1.2
            # avail_mem = req_mem/2 # ! HACK FOR TESTING !

            if req_mem > avail_mem:
                # Read id_per_chunk ids at a time to keep memory use down
                total_chunks = int(np.ceil(req_mem/avail_mem))
                id_per_chunk = int(np.ceil(len(self.id_present)/total_chunks))
            else:
                total_chunks = 1
                id_per_chunk = len(self.id_present)

            for chunk in range(total_chunks):
                # Minimum and maximum particle indices for this chunk
                idx_range = [0, 0]
                idx_range[0] = int(chunk*id_per_chunk)
                idx_range[1] = int(np.min(((chunk+1)*id_per_chunk,
                                          len(self.id_present))))

                # Minimum and maximum id for this chunk
                id_range = [self.id_present[idx_range[0]],
                            self.id_present[idx_range[1]-1]]

                # Read chunk-sized data from NPY-files
                data = self.read_from_npy(global_file_list, len(self.time_written), var, id_range)
                if (var == self.var_names[0]) & (chunk == 0):
                    # !! unacceptable assumption !!
                    # Assumes that the number of time-steps in the first chunk
                    # == number of time-steps across all chunks.
                    self.open_netcdf_file((len(self.id_present), data.shape[1]))

                varout = 'z' if var == 'depth' else var
                # Write to correct location in netcdf file
                getattr(self, varout)[idx_range[0]:idx_range[1], :] = data

        if len(self.var_names_once) > 0:
            for var in self.var_names_once:
                getattr(self, var)[:] = self.read_from_npy(global_file_list_once, 1, var, [0, self.maxid_written+1])

        self.close_netcdf_file()

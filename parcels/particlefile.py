"""Module controlling the writing of ParticleSets to NetCDF file"""
import numpy as np
import netCDF4
from datetime import timedelta as delta
from parcels.tools.loggers import logger
import os
from tempfile import gettempdir
import psutil
from parcels.tools.error import ErrorCode

try:
    from parcels._version import version as parcels_version
except:
    raise EnvironmentError('Parcels version can not be retrieved. Have you run ''python setup.py install''?')


__all__ = ['ParticleFile']


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
    """

    def __init__(self, name, particleset, outputdt=np.infty, write_ondelete=False, chunksizes=None):

        self.name = name
        self.write_ondelete = write_ondelete
        self.outputdt = outputdt
        self.lasttraj = 0  # id of last particle written
        self.lasttime_written = None  # variable to check if time has been written already

        self.dataset = None
        self.metadata = {}
        self.particleset = particleset
        self.var_names = []
        self.user_vars_once = []
        for v in self.particleset.ptype.variables:
            if v.name in ['time', 'lat', 'lon', 'depth', 'id']:
                self.var_names += [v.name]
            elif v.to_write:
                if v.to_write is True:
                    self.var_names += [v.name]
                elif v.to_write == 'once':
                    self.user_vars_once += [v.name]

        self.npy_path = os.path.join(gettempdir(), "parcels-%s" % os.getuid(), "out")
        self.file_list = []
        self.time_written = []
        self.maxid_written = -1
        self.dataset_closed = False
        if os.path.exists(self.npy_path):
            os.system("rm -rf " + self.npy_path)
            print("Existing temporary output folder " + self.npy_path + " from previous runs (probably aborted) was deleted")

    def open_dataset(self, data_shape):
        extension = os.path.splitext(str(self.name))[1]
        fname = self.name if extension in ['.nc', '.nc4'] else "%s.nc" % self.name
        self.dataset = netCDF4.Dataset(fname, "w", format="NETCDF4")
        self.dataset.createDimension("obs", data_shape[1])
        self.dataset.createDimension("traj", data_shape[0])
        coords = ("traj", "obs")
        self.dataset.feature_type = "trajectory"
        self.dataset.Conventions = "CF-1.6/CF-1.7"
        self.dataset.ncei_template_version = "NCEI_NetCDF_Trajectory_Template_v2.0"
        self.dataset.parcels_version = parcels_version
        self.dataset.parcels_mesh = self.particleset.fieldset.gridset.grids[0].mesh

        # Create ID variable according to CF conventions
        self.id = self.dataset.createVariable("trajectory", "i4", coords, fill_value=-2147483647, chunksizes=data_shape)
        self.id.long_name = "Unique identifier for each particle"
        self.id.cf_role = "trajectory_id"

        # Create time, lat, lon and z variables according to CF conventions:
        self.time = self.dataset.createVariable("time", "f8", coords, fill_value=np.nan, chunksizes=data_shape)
        self.time.long_name = ""
        self.time.standard_name = "time"
        if self.particleset.time_origin.calendar is None:
            self.time.units = "seconds"
        else:
            self.time.units = "seconds since " + str(self.particleset.time_origin)
            self.time.calendar = self.particleset.time_origin.calendar
        self.time.axis = "T"

        self.lat = self.dataset.createVariable("lat", "f4", coords, fill_value=np.nan, chunksizes=data_shape)
        self.lat.long_name = ""
        self.lat.standard_name = "latitude"
        self.lat.units = "degrees_north"
        self.lat.axis = "Y"

        self.lon = self.dataset.createVariable("lon", "f4", coords, fill_value=np.nan, chunksizes=data_shape)
        self.lon.long_name = ""
        self.lon.standard_name = "longitude"
        self.lon.units = "degrees_east"
        self.lon.axis = "X"

        self.z = self.dataset.createVariable("z", "f4", coords, fill_value=np.nan, chunksizes=data_shape)
        self.z.long_name = ""
        self.z.standard_name = "depth"
        self.z.units = "m"
        self.z.positive = "down"

        for v in self.particleset.ptype.variables:
            if v.to_write and v.name not in ['time', 'lat', 'lon', 'z', 'id']:
                if v.to_write is True:
                    setattr(self, v.name, self.dataset.createVariable(v.name, "f4", coords, fill_value=np.nan, chunksizes=data_shape))
                elif v.to_write == 'once':
                    setattr(self, v.name, self.dataset.createVariable(v.name, "f4", "traj", fill_value=np.nan, chunksizes=[data_shape[0]]))
                getattr(self, v.name).long_name = ""
                getattr(self, v.name).standard_name = v.name
                getattr(self, v.name).units = "unknown"

        for name, message in self.metadata.items():
            setattr(self.dataset, name, message)

    def __del__(self):
        if not self.dataset_closed:
            self.close()

    def close(self):
        self.export()
        self.delete_npyfiles()
        self.dataset.close()
        self.dataset_closed = True

    def add_metadata(self, name, message):
        """Add metadata to :class:`parcels.particleset.ParticleSet`
        :param name: Name of the metadata variabale
        :param message: message to be written
        """
        if self.dataset is None:
            self.metadata[name] = message
        else:
            setattr(self.dataset, name, message)

    def write(self, pset, time, deleted_only=False):
        """Write :class:`parcels.particleset.ParticleSet`
        All data from one time step is saved to one NPY-file using a python
        dictionary. The data is saved in the folder 'out'.

        :param pset: ParticleSet object to write
        :param time: Time at which to write ParticleSet

        """
        if isinstance(time, delta):
            time = time.total_seconds()
        if self.lasttime_written != time and \
           (self.write_ondelete is False or deleted_only is True):
            if pset.size > 0:
                data = {}
                for var in self.var_names:
                    data[var] = np.nan * np.zeros(len(pset))

                i = 0
                for p in pset:
                    if p.dt*p.time <= p.dt*time:
                        for var in self.var_names:
                            data[var][i] = getattr(p, var)
                        if p.state != ErrorCode.Delete and not np.allclose(p.time, time):
                            logger.warning_once('time argument in pfile.write() is %g, but a particle has time %g.' % (time, p.time))
                        self.maxid_written = np.max([self.maxid_written, p.id])
                        i += 1

                if not os.path.exists(self.npy_path):
                    os.mkdir(self.npy_path)

                save_ind = np.isfinite(data["id"])
                for key in self.var_names:
                    data[key] = data[key][save_ind]

                tmpfilename = os.path.join(self.npy_path, str(len(self.file_list)+1))
                np.save(tmpfilename, data)
                self.file_list.append(tmpfilename+".npy")
                if time not in self.time_written:
                    self.time_written.append(time)

            else:
                logger.warning("ParticleSet is empty on writing as array at time %g" % time)

            if not deleted_only:
                self.lasttime_written = time

    def export(self, batch_processing=False, batch_size=2**30):
        """Exports outputs in temporary NPY-files to NetCDF file

        :param batch_processing: If True batch processing is applied. Batches
        of files are loaded and then written to the output netcdf-file. This
        option is still work in progress. Default:True

        :param batch_size: size of one batch in bytes. Default: 2**30 (1 Gbyte).

        :type batch_processing: boolean
        :type batch_size: int
        """

        def get_available_memory():
            """return available memory in bytes"""
            memory_info = psutil.virtual_memory()
            available_memory = memory_info.available
            return available_memory

        def read(file_list, time_steps):
            """Read NPY-files using a loop over all files and return one array
            for each variable.

            :param file_list: List that  contains all file names in the output
            directory
            :param id_fill_value: FillValue for the IDs as used in the output
            netcdf

            :type file_list: list
            :type id_fill_value: int

            :return: Python dictionary with the data that was read from the
            NPY-files
            """

            merge_dict = {}
            for var in self.var_names:
                merge_dict[var] = np.nan * np.zeros((self.maxid_written+1, time_steps))
            time_index = np.zeros(self.maxid_written+1, dtype=int)

            # loop over all files
            for npyfile in file_list:
                data_dict = np.load(npyfile).item()

                id_ind = np.array(data_dict["id"], dtype=int)
                t_ind = time_index[id_ind]
                for key in self.var_names:
                    merge_dict[key][id_ind, t_ind] = data_dict[key]

                time_index[id_ind] = time_index[id_ind] + 1

            # remove rows and columns that are completely filled with nan values
            out_dict = {}
            for var in self.var_names:
                tmp = merge_dict[var][~np.isnan(merge_dict["lat"]).all(axis=1)]
                out_dict[var] = tmp[:, ~np.isnan(merge_dict["lat"]).all(axis=0)]

            return out_dict

        if batch_processing:
            print("=============convert NPY-files to NetCDF-file===============")

            available_memory = get_available_memory()
            print("Available memory: '" + str(available_memory/float(2**20)) + "' Mbytes")
            self.batch_size = batch_size

            # estimate the total size in bytes for merging array
            self.memory_estimate_total = (self.maxid_written+1) * len(self.time_written) * len(self.var_names) * 8
            self.memory_per_file = self.memory_estimate_total/len(self.file_list)
            print("Estimated memory needed: '" + str(float(self.memory_estimate_total)/2**20) + "' Mbytes")
            print("Estimated memory per file needed: '" + str(float(self.memory_per_file)/2**20) + "' Mbytes")

            if self.memory_per_file > available_memory*0.9:
                raise MemoryError("Too little available memory is available to load even one tempory output file. With 10% safety margin.")

            if 0.7*available_memory < self.batch_size:
                self.batch_size = 0.7 * available_memory
                print("Too little available  memory available in ParticleFile.conversion_npy() for batch_size:" + str(batch_size/2**20) + " Mbytes.")
                print("Setting batch_size to 70% of available  memory:"+str(float(self.batch_size)/2**20) + " Mbytes.")

            if batch_size < self.memory_per_file:
                self.batch_size = self.memory_per_file
                print("'batch_size' size too little.'batch_size' has to be at least as big as the memory needed per file.")
                print("Set batch the memory that is needed per file: " + str(self.batch_size/2**20) + "Mbytes.")

            print("Convert NPY-files using batch processing in batches of max. size '"+str(float(self.batch_size)/2**20)+"' Mbytes.")

            # divide the data set into chunks
            n_reading_loops = int(self.memory_estimate_total//self.batch_size)
            if n_reading_loops == 0:
                n_reading_loops = 1

            files_per_loop = len(self.file_list)//n_reading_loops

            if files_per_loop == 0:
                raise ValueError("'files_per_loop' can not be equal to 0. Some bug might exist in ParticleFile.export().")

            time_list_splitted = [self.file_list[i:i + files_per_loop] for i in range(0, len(self.file_list), files_per_loop)]

            print("Convert data using " + str(len(time_list_splitted))+" batch(es).")

            # last time index that was filled
            last_filled = 0

            # reading loop
            for time_list_loop in time_list_splitted:
                n_time_step = len(time_list_loop)
                data_dict = read(time_list_loop, n_time_step)

                for var in self.var_names:
                    getattr(self, var)[:, last_filled:last_filled+n_time_step] = data_dict[var]

                last_filled += n_time_step

        else:
            data_dict = read(self.file_list, len(self.time_written))
            self.open_dataset(data_dict["id"].shape)
            for var in self.var_names:
                if var == "depth":
                    self.z[:, :] = data_dict["depth"]
                else:
                    getattr(self, var)[:, :] = data_dict[var]

    def delete_npyfiles(self):
        if os.path.exists(self.npy_path):
            print("Remove folder '"+self.npy_path+"' after conversion of NPY-files to NetCDF file '" + str(self.name) + "'.")
            os.system("rm -rf "+self.npy_path)

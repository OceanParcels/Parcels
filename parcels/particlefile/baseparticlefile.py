"""Module controlling the writing of ParticleSets to NetCDF file"""
import os
import sys
import random
import shutil
import string
from abc import ABC
from abc import abstractmethod

import netCDF4
import numpy as np

try:
    from mpi4py import MPI
except:
    MPI = None
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


__all__ = ['BaseParticleFile']


def _set_calendar(origin_calendar):
    if origin_calendar == 'np_datetime64':
        return 'standard'
    else:
        return origin_calendar


class BaseParticleFile(ABC):
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
    write_ondelete = None
    convert_at_end = None
    outputdt = None
    lasttime_written = None
    dataset = None
    metadata = None
    name = None
    particleset = None
    parcels_mesh = None
    time_origin = None
    lonlatdepth_dtype = None
    var_names = None
    var_dtypes = None
    file_list = None
    var_names_once = None
    var_dtypes_once = None
    fill_value_map = dict()
    fmt_map = dict()
    file_list_once = None
    maxid_written = -1
    tempwritedir_base = None
    tempwritedir = None

    def __init__(self, name, particleset, outputdt=np.infty, write_ondelete=False, convert_at_end=True,
                 tempwritedir=None, pset_info=None):
        """
        BaseParticleFile - Constructor
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

        self.write_ondelete = write_ondelete
        self.convert_at_end = convert_at_end
        self.outputdt = outputdt
        self.lasttime_written = None

        self.dataset = None
        self.metadata = {}
        if pset_info:
            for v in pset_info.keys():
                setattr(self, v, pset_info[v])
        else:
            self.name = name
            self.particleset = particleset
            self.parcels_mesh = 'spherical'
            if self.particleset.fieldset is not None:
                self.parcels_mesh = self.particleset.fieldset.gridset.grids[0].mesh
            self.time_origin = self.particleset.time_origin
            self.lonlatdepth_dtype = self.particleset.collection.lonlatdepth_dtype
            self.var_names = []
            self.var_dtypes = []
            self.var_names_once = []
            self.var_dtypes_once = []
            for v in self.particleset.collection.ptype.variables:
                if v.to_write == 'once':
                    self.var_names_once += [v.name]
                    self.var_dtypes_once += [v.dtype]
                elif v.to_write is True:
                    self.var_names += [v.name]
                    self.var_dtypes += [v.dtype]
            if len(self.var_names_once) > 0:
                self.written_once = []
                self.file_list_once = []

            self.file_list = []

        # Create dictionary to translate datatypes and fill_values
        self.fmt_map = {np.float32: 'f4', np.float64: 'f8',
                        np.bool_: 'i1', np.int16: 'i2', np.int32: 'i4', np.int64: 'i8'}
        self.fill_value_map = {np.float32: np.nan, np.float64: np.nan,
                               np.bool_: np.iinfo(np.int8).min, np.int16: np.iinfo(np.int16).max,
                               np.int32: np.iinfo(np.int32).max, np.int64: np.iinfo(np.int64).max}

        tmp_dir = tempwritedir
        if tempwritedir is None:
            tmp_dir = os.path.join(os.path.dirname(str(self.name)), "out-%s" % ''.join(random.choice(string.ascii_uppercase) for _ in range(8)))
        else:
            tmp_dir = tempwritedir

        if MPI:
            mpi_rank = MPI.COMM_WORLD.Get_rank()
            self.tempwritedir_base = MPI.COMM_WORLD.bcast(tmp_dir, root=0)
        else:
            self.tempwritedir_base = tmp_dir
            mpi_rank = 0
        self.tempwritedir = os.path.join(self.tempwritedir_base, "%d" % mpi_rank)

        if not os.path.exists(self.tempwritedir):
            os.makedirs(self.tempwritedir)
        elif pset_info is None:
            raise IOError("output directory %s already exists. Please remove the directory." % self.tempwritedir)

    def __del__(self):
        """
        BaseParticleFile - Destructor
        """
        if self.convert_at_end:
            self.close()

    def __sizeof__(self):
        """
        :returns size (in bytes) of the ParticleFile
        """
        sz = 0
        if self.var_names is not None:
            for vname in self.var_names:
                sz += sys.getsizeof(getattr(self, vname)) if getattr(self, vname) is not None else 0
        if self.var_names_once is not None:
            for vname in self.var_names_once:
                sz += sys.getsizeof(getattr(self, vname)) if getattr(self, vname) is not None else 0
        for var in [self.outputdt, self.lasttime_written, self.dataset, self.metadata, self.name, self.parcels_mesh,
                    self.time_origin, self.lonlatdepth_dtype, self.var_names, self.file_list, self.var_names_once,
                    self.file_list_once, self.tempwritedir_base, self.tempwritedir]:
            sz += sys.getsizeof(var) if var is not None else 0
        sz += sys.getsizeof(self.maxid_written)
        return sz

    @abstractmethod
    def _reserved_var_names(self):
        """
        returns the reserved dimension names not to be written just once.
        """
        pass

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

        coords = self._create_trajectory_file(fname=fname, data_shape=data_shape)
        self._create_trajectory_records(coords=coords)
        self._create_metadata_records()

    def close_netcdf_file(self):
        """
        closes the NetCDF file and flushes the written content to disk
        """
        self.dataset.close()

    def _create_trajectory_file(self, fname, data_shape):
        """
        This function opens the related NetCDF file and creates its dataset.
        :arg fname: filepath of the NetCDF file to be written
        :arg data_shape: shape of the record to be written
        """
        self.dataset = netCDF4.Dataset(fname, "w", format="NETCDF4")
        self.dataset.createDimension("obs", data_shape[1])
        self.dataset.createDimension("traj", data_shape[0])
        coords = ("traj", "obs")
        self.dataset.feature_type = "trajectory"
        self.dataset.Conventions = "CF-1.6/CF-1.7"
        self.dataset.ncei_template_version = "NCEI_NetCDF_Trajectory_Template_v2.0"
        self.dataset.parcels_version = parcels_version
        self.dataset.parcels_mesh = self.parcels_mesh
        return coords

    def _create_trajectory_records(self, coords):
        """
        creates the NetCDF record structure of a trajectory.
        :arg coords: tuple of dictionary keys for # entities ("traj(ectories)") and timesteps ("obs(ervations)")

        Attention:
        For ParticleSet structures other than SoA, and structures where ID != index, this has to be overridden.
        """
        # Create ID variable according to CF conventions
        self.id = self.dataset.createVariable("trajectory", "i8", coords, fill_value=-2**(63))
        self.id.long_name = "Unique identifier for each particle"
        self.id.cf_role = "trajectory_id"

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

        self._create_variable_records(coords=coords)

    def _create_variable_records(self, coords):
        """
        creates the NetCDF record structure for the (user-defined) variables of a trajectory.
        :arg coords: tuple of dictionary keys for # entities ("traj(ectories)") and timesteps ("obs(ervations)")
        """
        for vname, dtype in zip(self.var_names, self.var_dtypes):
            if vname not in self._reserved_var_names():
                fill_value = self.fill_value_map[dtype]
                nc_dtype_fmt = self.fmt_map[dtype]
                setattr(self, vname, self.dataset.createVariable(vname, nc_dtype_fmt, coords, fill_value=fill_value))
                getattr(self, vname).long_name = ""
                getattr(self, vname).standard_name = vname
                getattr(self, vname).units = "unknown"

        for vname, dtype in zip(self.var_names_once, self.var_dtypes_once):
            fill_value = self.fill_value_map[dtype]
            nc_dtype_fmt = self.fmt_map[dtype]
            setattr(self, vname, self.dataset.createVariable(vname, nc_dtype_fmt, "traj", fill_value=fill_value))
            getattr(self, vname).long_name = ""
            getattr(self, vname).standard_name = vname
            getattr(self, vname).units = "unknown"

    def _create_metadata_records(self):
        """
        this function creates the metadata (i.e. the header) of the NetCDF file
        """
        for name, message in self.metadata.items():
            setattr(self.dataset, name, message)

    def close(self, delete_tempfiles=True):
        """
        Close the ParticleFile object by exporting and then deleting
        the temporary npy files
        :arg delete_tempfiles: boolean, telling if the temporary npy files are to be deleted or not
        """
        self.export()
        mpi_rank = MPI.COMM_WORLD.Get_rank() if MPI else 0
        if mpi_rank == 0:
            if delete_tempfiles:
                self.delete_tempwritedir(tempwritedir=self.tempwritedir_base)
        self.convert_at_end = False

    def add_metadata(self, name, message):
        """Add metadata to :class:`parcels.particleset.ParticleSet`

        :param name: Name of the metadata variabale
        :param message: message to be written
        """
        if self.dataset is None:
            self.metadata[name] = message
        else:
            setattr(self.dataset, name, message)

    @abstractmethod
    def get_pset_info_attributes(self):
        """
        :returns the main attributes of the pset_info.npy file.

        Attention:
        For ParticleSet structures other than SoA, and structures where ID != index, this has to be overridden.
        """
        return None

    def dump_dict_to_npy(self, data_dict, data_dict_once):
        """
        Buffer data to set of temporary numpy files, using np.save
        :arg data_dict: a dict of data, each entry associating 'var_name' -> 2D numpy.ndarray of values (# particles x timesteps)
        :arg data_dict_once:  a dict of data, each entry associating 'var_name' -> 2D numpy.ndarray of values, only written once
        """

        if not os.path.exists(self.tempwritedir):
            os.makedirs(self.tempwritedir)

        if len(data_dict) > 0:
            tmpfilename = os.path.join(self.tempwritedir, str(len(self.file_list)) + ".npy")
            with open(tmpfilename, 'wb') as f:
                np.save(f, data_dict)
            self.file_list.append(tmpfilename)

        if len(data_dict_once) > 0:
            tmpfilename = os.path.join(self.tempwritedir, str(len(self.file_list)) + '_once.npy')
            with open(tmpfilename, 'wb') as f:
                np.save(f, data_dict_once)
            self.file_list_once.append(tmpfilename)

    def dump_psetinfo_to_npy(self):
        """
        function writes the major attributes and values to a pset information file (*.npy).
        """
        pset_info = {}
        attrs_to_dump = self.get_pset_info_attributes()
        if attrs_to_dump is None:
            return
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

        data_dict, data_dict_once = pset.to_dict(self, time, deleted_only=deleted_only)
        self.dump_dict_to_npy(data_dict, data_dict_once)
        self.dump_psetinfo_to_npy()

    @abstractmethod
    def read_from_npy(self, file_list, var, dtype, time_steps=None, n_timesteps=None):
        """
        Read NPY-files for one variable using a loop over all files.

        Attention:
        For ParticleSet structures other than SoA, and structures where ID != index, this has to be overridden.

        :param file_list: List that  contains all file names in the output directory
        :param time_steps: Number of time steps that were written in out directory
        :param n_timesteps: Dictionary with (for each particle) number of time steps that were written in out directory
        :param dtype: 'dtype' of the variable's data to be written
        :param var: name of the variable to read
        """
        return None

    @abstractmethod
    def export(self):
        """
        Exports outputs in temporary NPY-files to NetCDF file

        Attention:
        For ParticleSet structures other than SoA, and structures where ID != index, this has to be overridden.
        """
        pass

    def delete_tempwritedir(self, tempwritedir=None):
        """Deleted all temporary npy files

        :param tempwritedir Optional path of the directory to delete
        """
        if tempwritedir is None:
            tempwritedir = self.tempwritedir
        if os.path.exists(tempwritedir):
            shutil.rmtree(tempwritedir)

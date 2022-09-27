"""Module controlling the writing of ParticleSets to NetCDF file"""
import os
import random
import shutil
import string
from abc import ABC
from abc import abstractmethod
import gzip

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
    file_list_once = None
    maxid_written = -1
    tempwritedir_base = None
    tempwritedir = None

    def __init__(self, name, particleset, outputdt=np.infty, write_ondelete=False, convert_at_end=True,
                 tempwritedir=None, pset_info=None):

        self.write_ondelete = write_ondelete
        self.convert_at_end = convert_at_end
        self.outputdt = outputdt
        self.lasttime_written = None  # variable to check if time has been written already

        self.dataset = None
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

        self.metadata = {"feature_type": "trajectory", "Conventions": "CF-1.6/CF-1.7",
                         "ncei_template_version": "NCEI_NetCDF_Trajectory_Template_v2.0",
                         "parcels_version": parcels_version,
                         "parcels_mesh": self.parcels_mesh}

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

    @abstractmethod
    def _reserved_var_names(self):
        """
        returns the reserved dimension names not to be written just once.
        """
        pass

    def open_output_file(self, data_shape):
        """Initialise file for trajectory output.
        The output follows the format outlined in the Discrete Sampling Geometries
        section of the CF-conventions:
        http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#discrete-sampling-geometries
        The current implementation is based on the NCEI template:
        http://www.nodc.noaa.gov/data/formats/netcdf/v2.0/trajectoryIncomplete.cdl

        :param data_shape: shape of the variables in the output file
        """
        extension = os.path.splitext(str(self.name))[1]
        self.fname = self.name if extension in ['.nc', '.nc4', '.zarr'] else "%s.nc" % self.name
        self.outputformat = extension
        if os.path.exists(str(self.fname)):
            if 'zarr' in self.outputformat:
                shutil.rmtree(str(self.fname))
            else:
                os.remove(str(self.fname))
        self.attrs = self._create_variables_attribute_dict()

    def _create_variables_attribute_dict(self):
        """
        creates the dictionary with variable attributes.

        Attention:
        For ParticleSet structures other than SoA, and structures where ID != index, this has to be overridden.
        """

        attrs = {'z': {"long_name": "",
                       "standard_name": "depth",
                       "units": "m",
                       "positive": "down"},
                 'trajectory': {"long_name": "Unique identifier for each particle",
                                "cf_role": "trajectory_id",
                                "_FillValue": self.fill_value_map[np.int64]},
                 'time': {"long_name": "",
                          "standard_name": "time",
                          "units": "seconds",
                          "axis": "T"},
                 'lon': {"long_name": "",
                         "standard_name": "longitude",
                         "units": "degrees_east",
                         "axis":
                             "X"},
                 'lat': {"long_name": "",
                         "standard_name": "latitude",
                         "units": "degrees_north",
                         "axis": "Y"}}

        if self.time_origin.calendar is not None:
            attrs['time']['units'] = "seconds since " + str(self.time_origin)
            attrs['time']['calendar'] = 'standard' if self.time_origin.calendar == 'np_datetime64' else self.time_origin.calendar

        for vname, dtype in zip(self.var_names, self.var_dtypes):
            if vname not in self._reserved_var_names():
                attrs[vname] = {"_FillValue": self.fill_value_map[dtype],
                                "long_name": "",
                                "standard_name": vname,
                                "units": "unknown"}

        for vname, dtype in zip(self.var_names_once, self.var_dtypes_once):
            attrs[vname] = {"_FillValue": self.fill_value_map[dtype],
                            "long_name": "",
                            "standard_name": vname,
                            "units": "unknown"}

        return attrs

    def __del__(self):
        if self.convert_at_end:
            self.close()

    def close(self, delete_tempfiles=True):
        """Close the ParticleFile object by exporting and then deleting
        the temporary npy files"""
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
        self.metadata[name] = message

    def dump_dict_to_npy(self, data_dict, data_dict_once):
        """Buffer data to set of temporary numpy files, using np.save"""

        if not os.path.exists(self.tempwritedir):
            os.makedirs(self.tempwritedir)

        if len(data_dict) > 0:
            tmpfilename = os.path.join(self.tempwritedir, str(len(self.file_list)) + ".npy.gz")
            with gzip.open(tmpfilename, 'wb') as f:
                np.save(f, data_dict)
            self.file_list.append(tmpfilename)

        if len(data_dict_once) > 0:
            tmpfilename = os.path.join(self.tempwritedir, str(len(self.file_list)) + '_once.npy.gz')
            with gzip.open(tmpfilename, 'wb') as f:
                np.save(f, data_dict_once)
            self.file_list_once.append(tmpfilename)

    @abstractmethod
    def get_pset_info_attributes(self):
        """
        returns the main attributes of the pset_info.npy file.

        Attention:
        For ParticleSet structures other than SoA, and structures where ID != index, this has to be overridden.
        """
        return None

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
    def read_from_npy(self, file_list, time_steps, var):
        """
        Read NPY-files for one variable using a loop over all files.

        Attention:
        For ParticleSet structures other than SoA, and structures where ID != index, this has to be overridden.

        :param file_list: List that  contains all file names in the output directory
        :param time_steps: Number of time steps that were written in out directory
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

"""Module controlling the writing of ParticleSets to Zarr file"""
from abc import ABC
from abc import abstractmethod
import os
import numpy as np
import xarray as xr

try:
    from mpi4py import MPI
except:
    MPI = None
try:
    from parcels._version import version as parcels_version
except:
    raise EnvironmentError('Parcels version can not be retrieved. Have you run ''python setup.py install''?')


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
    """
    write_ondelete = None
    convert_at_end = None
    outputdt = None
    lasttime_written = None
    name = None
    particleset = None
    parcels_mesh = None
    time_origin = None
    lonlatdepth_dtype = None
    var_names = None
    var_dtypes = None
    var_names_once = None
    var_dtypes_once = None

    def __init__(self, name, particleset, outputdt=np.infty, write_ondelete=False, convert_at_end=True):

        self.write_ondelete = write_ondelete
        self.outputdt = outputdt
        self.lasttime_written = None  # variable to check if time has been written already

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
        self.written_first = False

        self.metadata = {"feature_type": "trajectory", "Conventions": "CF-1.6/CF-1.7",
                         "ncei_template_version": "NCEI_NetCDF_Trajectory_Template_v2.0",
                         "parcels_version": parcels_version,
                         "parcels_mesh": self.parcels_mesh}

        # Create dictionary to translate datatypes and fill_values
        self.fmt_map = {np.float16: 'f2', np.float32: 'f4', np.float64: 'f8',
                        np.bool_: 'i1', np.int8: 'i1', np.int16: 'i2',
                        np.int32: 'i4', np.int64: 'i8', np.uint8: 'u1',
                        np.uint16: 'u2', np.uint32: 'u4', np.uint64: 'u8'}
        self.fill_value_map = {np.float16: np.nan, np.float32: np.nan, np.float64: np.nan,
                               np.bool_: np.iinfo(np.int8).max, np.int8: np.iinfo(np.int8).max,
                               np.int16: np.iinfo(np.int16).max, np.int32: np.iinfo(np.int32).max,
                               np.int64: np.iinfo(np.int64).max, np.uint8: np.iinfo(np.uint8).max,
                               np.uint16: np.iinfo(np.uint16).max, np.uint32: np.iinfo(np.uint32).max,
                               np.uint64: np.iinfo(np.uint64).max}

        extension = os.path.splitext(str(self.name))[1]
        self.fname = self.name if extension in ['.nc', '.nc4', '.zarr'] else "%s.zarr" % self.name
        self.outputformat = extension

    @abstractmethod
    def _reserved_var_names(self):
        """
        returns the reserved dimension names not to be written just once.
        """
        pass

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
        self.close()

    def close(self, delete_tempfiles=True):
        pass

    def add_metadata(self, name, message):
        """Add metadata to :class:`parcels.particleset.ParticleSet`

        :param name: Name of the metadata variabale
        :param message: message to be written
        """
        self.metadata[name] = message

    def write(self, pset, time, deleted_only=False):
        """Write all data from one time step to the zarr file

        :param pset: ParticleSet object to write
        :param time: Time at which to write ParticleSet
        :param deleted_only: Flag to write only the deleted Particles
        """

        ds = xr.Dataset(attrs=self.metadata)
        attrs = self._create_variables_attribute_dict()
        datalen = max(pset.id) + 1
        data = np.nan * np.ones((datalen, 1))

        for var, dtype in zip(self.var_names, self.var_dtypes):
            varout = 'z' if var == 'depth' else var
            varout = 'trajectory' if varout == 'id' else varout

            data[pset.id, 0] = getattr(pset, var)
            ds[varout] = xr.DataArray(data=data, dims=["traj", "obs"], attrs=attrs[varout])
            if self.written_first and "_FillValue" in ds[varout].attrs:
                del ds[varout].attrs["_FillValue"]
        if not self.written_first:
            ds.to_zarr(self.fname, mode='w')
            self.written_first = True
        else:
            ds.to_zarr(self.fname, mode='a', append_dim='obs')

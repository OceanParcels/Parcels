from datetime import date
from datetime import datetime
from datetime import timedelta as delta

# import sys
import numpy as np
# import xarray as xr

# from parcels.grid import GridCode
# from parcels.grid import CurvilinearGrid
# from parcels.kernel import Kernel
from parcels.particle import JITParticle
# from parcels.particlefile import ParticleFile
# from parcels.tools.statuscodes import StateCode
from parcels.particlesets.baseparticleset import BaseParticleSet
from parcels.collectionaos import ParticleCollectionAOS

# from .collectionsoa import ParticleCollectionSOA
# from .collectionsoa import ParticleCollectionIteratorSOA

from parcels.tools.converters import _get_cftime_calendars
from parcels.tools.loggers import logger
try:
    from mpi4py import MPI
except:
    MPI = None
# == comment CK: prevents us from adding KDTree as 'mandatory' dependency == #
try:
    from pykdtree.kdtree import KDTree
except:
    KDTree = None

__all__ = ['ParticleSetAOS']


def _convert_to_array(var):
    """Convert lists and single integers/floats to one-dimensional numpy
    arrays
    """
    if isinstance(var, np.ndarray):
        return var.flatten()
    elif isinstance(var, (int, float, np.float32, np.int32)):
        return np.array([var])
    else:
        return np.array(var)


def _convert_to_reltime(time):
    """Check to determine if the value of the time parameter needs to be
    converted to a relative value (relative to the time_origin).
    """
    if isinstance(time, np.datetime64) or (hasattr(time, 'calendar') and time.calendar in _get_cftime_calendars()):
        return True
    return False


class ParticleSetAOS(BaseParticleSet):
    """Container class for storing particle and executing kernel over them.

    Please note that this currently only supports fixed size particle sets.

    :param fieldset: :mod:`parcels.fieldset.FieldSet` object from which to sample velocity.
           While fieldset=None is supported, this will throw a warning as it breaks most Parcels functionality
    :param pclass: Optional :mod:`parcels.particle.JITParticle` or
                 :mod:`parcels.particle.ScipyParticle` object that defines custom particle
    :param lon: List of initial longitude values for particles
    :param lat: List of initial latitude values for particles
    :param depth: Optional list of initial depth values for particles. Default is 0m
    :param time: Optional list of initial time values for particles. Default is fieldset.U.grid.time[0]
    :param repeatdt: Optional interval (in seconds) on which to repeat the release of the ParticleSet
    :param lonlatdepth_dtype: Floating precision for lon, lat, depth particle coordinates.
           It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
           and np.float64 if the interpolation method is 'cgrid_velocity'
    :param pid_orig: Optional list of (offsets for) the particle IDs
    :param partitions: List of cores on which to distribute the particles for MPI runs. Default: None, in which case particles
           are distributed automatically on the processors

    Other Variables can be initialised using further arguments (e.g. v=... for a Variable named 'v')
    """

    def __init__(self, fieldset=None, pclass=JITParticle, lon=None, lat=None, depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None, pid_orig=None, **kwargs):
        super(ParticleSetAOS, self).__init__()
        self.fieldset = fieldset
        if self.fieldset is None:
            logger.warning_once("No FieldSet provided in ParticleSet generation. "
                                "This breaks most Parcels functionality")
        else:
            self.fieldset.check_complete()
        partitions = kwargs.pop('partitions', None)

        lon = np.empty(shape=0) if lon is None else _convert_to_array(lon)
        lat = np.empty(shape=0) if lat is None else _convert_to_array(lat)

        if isinstance(pid_orig, (type(None), type(False))):
            pid_orig = np.arange(lon.size)

        if depth is None:
            mindepth = self.fieldset.gridset.dimrange('depth')[0] if self.fieldset is not None else 0
            depth = np.ones(lon.size) * mindepth
        else:
            depth = _convert_to_array(depth)
        assert lon.size == lat.size and lon.size == depth.size, (
            'lon, lat, depth don''t all have the same lenghts')

        time = _convert_to_array(time)
        time = np.repeat(time, lon.size) if time.size == 1 else time

        if time.size > 0 and type(time[0]) in [datetime, date]:
            time = np.array([np.datetime64(t) for t in time])
        self.time_origin = fieldset.time_origin if self.fieldset is not None else 0
        if time.size > 0 and isinstance(time[0], np.timedelta64) and not self.time_origin:
            raise NotImplementedError('If fieldset.time_origin is not a date, time of a particle must be a double')
        time = np.array([self.time_origin.reltime(t) if _convert_to_reltime(t) else t for t in time])
        assert lon.size == time.size, (
            'time and positions (lon, lat, depth) don''t have the same lengths.')

        if lonlatdepth_dtype is None:
            if fieldset is not None:
                lonlatdepth_dtype = self.lonlatdepth_dtype_from_field_interp_method(fieldset.U)
            else:
                lonlatdepth_dtype = np.float32
        assert lonlatdepth_dtype in [np.float32, np.float64], \
            'lon lat depth precision should be set to either np.float32 or np.float64'

        for kwvar in kwargs:
            kwargs[kwvar] = _convert_to_array(kwargs[kwvar])
            assert lon.size == kwargs[kwvar].size, (
                '%s and positions (lon, lat, depth) don''t have the same lengths.' % kwvar)

        self.repeatdt = repeatdt.total_seconds() if isinstance(repeatdt, delta) else repeatdt
        if self.repeatdt:
            if self.repeatdt <= 0:
                raise('Repeatdt should be > 0')
            if time[0] and not np.allclose(time, time[0]):
                raise ('All Particle.time should be the same when repeatdt is not None')
            self.repeatpclass = pclass
            self.repeatkwargs = kwargs

        ngrids = fieldset.gridset.size if fieldset is not None else 0
        self._collection = ParticleCollectionAOS(pclass, lon=lon, lat=lat, depth=depth, time=time, lonlatdepth_dtype=lonlatdepth_dtype, pid_orig=pid_orig, partitions=partitions, ngrid=ngrids, **kwargs)

        if self.repeatdt:
            if len(time) > 0 and time[0] is None:
                self.repeat_starttime = time[0]
            else:
                if self._collection.data['time'][0] and not np.allclose(self._collection.data['time'], self._collection.data['time'][0]):
                    raise ValueError('All Particle.time should be the same when repeatdt is not None')
                self.repeat_starttime = self._collection.data['time'][0]
            self.repeatlon = self._collection.data['lon']
            self.repeatlat = self._collection.data['lat']
            self.repeatdepth = self._collection.data['depth']
            for kwvar in kwargs:
                self.repeatkwargs[kwvar] = self._collection.data[kwvar]

        if self.repeatdt:
            if MPI and self._collection.pu_indicators is not None:
                mpi_comm = MPI.COMM_WORLD
                mpi_rank = mpi_comm.Get_rank()
                self.repeatpid = pid_orig[self._collection.pu_indicators == mpi_rank]

        self.kernel = None

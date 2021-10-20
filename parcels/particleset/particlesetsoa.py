from datetime import date
from datetime import datetime
from datetime import timedelta as delta

import sys
import numpy as np
import xarray as xr
from copy import copy

from parcels.grid import GridCode
from parcels.grid import CurvilinearGrid
from parcels.kernel import Kernel
from parcels.particle import Variable, ScipyParticle, JITParticle  # noqa
from parcels.particlefile import ParticleFile
from parcels.tools.statuscodes import StateCode
from parcels.particleset.baseparticleset import BaseParticleSet
from parcels.collection.collectionsoa import ParticleCollectionSOA
from parcels.collection.collectionsoa import ParticleCollectionIteratorSOA  # noqa
from parcels.collection.collectionsoa import ParticleCollectionIterableSOA  # noqa
from parcels.tools.converters import _get_cftime_calendars
from parcels.tools.loggers import logger
from parcels.interaction.interactionkernelsoa import InteractionKernelSOA
from parcels.interaction.neighborsearch import BruteSphericalNeighborSearch
from parcels.interaction.neighborsearch import BruteFlatNeighborSearch
from parcels.interaction.neighborsearch import KDTreeFlatNeighborSearch
from parcels.interaction.neighborsearch import HashSphericalNeighborSearch
try:
    from mpi4py import MPI
except:
    MPI = None
# == comment CK: prevents us from adding KDTree as 'mandatory' dependency == #
try:
    from pykdtree.kdtree import KDTree
except:
    KDTree = None

__all__ = ['ParticleSetSOA']


def _convert_to_array(var):
    """Convert lists and single integers/floats to one-dimensional numpy
    arrays
    """
    if isinstance(var, np.ndarray):
        return var.flatten()
    elif isinstance(var, (int, float, np.float32, np.float64, np.int32)):
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


class ParticleSetSOA(BaseParticleSet):
    """Container class for storing particle and executing kernel over them.

    Please note that this currently only supports fixed size particle sets, meaning that the particle set only
    holds the particles defined on construction. Individual particles can neither be added nor deleted individually,
    and individual particles can only be deleted as a set procedually (i.e. by 'particle.delete()'-call during
    kernel execution).

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
    :param periodic_domain_zonal: Zonal domain size, used to apply zonally periodic boundaries for particle-particle
           interaction. If None, no zonally periodic boundaries are applied

    Other Variables can be initialised using further arguments (e.g. v=... for a Variable named 'v')
    """

    def __init__(self, fieldset=None, pclass=JITParticle, lon=None, lat=None,
                 depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None,
                 pid_orig=None, interaction_distance=None, periodic_domain_zonal=None, **kwargs):
        super(ParticleSetSOA, self).__init__()

        # ==== first: create a new subclass of the pclass that includes the required variables ==== #
        # ==== see dynamic-instantiation trick here: https://www.python-course.eu/python3_classes_and_type.php ==== #
        class_name = "Array"+pclass.__name__
        array_class = None
        if class_name not in dir():
            def ArrayClass_init(self, *args, **kwargs):
                fieldset = kwargs.get('fieldset', None)
                ngrids = kwargs.get('ngrids', None)
                if type(self).ngrids.initial < 0:
                    numgrids = ngrids
                    if numgrids is None and fieldset is not None:
                        numgrids = fieldset.gridset.size
                    assert numgrids is not None, "Neither fieldsets nor number of grids are specified - exiting."
                    type(self).ngrids.initial = numgrids
                self.ngrids = type(self).ngrids.initial
                if self.ngrids >= 0:
                    for index in ['xi', 'yi', 'zi', 'ti']:
                        if index != 'ti':
                            setattr(self, index, np.zeros(self.ngrids, dtype=np.int32))
                        else:
                            setattr(self, index, -1*np.ones(self.ngrids, dtype=np.int32))
                super(type(self), self).__init__(*args, **kwargs)

            array_class_vdict = {"ngrids": Variable('ngrids', dtype=np.int32, to_write=False, initial=-1),
                                 "xi": Variable('xi', dtype=np.int32, to_write=False),
                                 "yi": Variable('yi', dtype=np.int32, to_write=False),
                                 "zi": Variable('zi', dtype=np.int32, to_write=False),
                                 "ti": Variable('ti', dtype=np.int32, to_write=False, initial=-1),
                                 "__init__": ArrayClass_init}
            array_class = type(class_name, (pclass, ), array_class_vdict)
        else:
            array_class = locals()[class_name]
        # ==== dynamic re-classing completed ==== #
        _pclass = array_class

        self.fieldset = fieldset
        if self.fieldset is None:
            logger.warning_once("No FieldSet provided in ParticleSet generation. This breaks most Parcels functionality")
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

        ngrids = fieldset.gridset.size if fieldset is not None else 1

        # Variables used for interaction kernels.
        inter_dist_horiz = None
        inter_dist_vert = None
        # The _dirty_neighbor attribute keeps track of whether
        # the neighbor search structure needs to be rebuilt.
        # If indices change (for example adding/deleting a particle)
        # The NS structure needs to be rebuilt and _dirty_neighbor should be
        # set to true. Since the NS structure isn't immediately initialized,
        # it is set to True here.
        self._dirty_neighbor = True

        self._collection = ParticleCollectionSOA(
            _pclass, lon=lon, lat=lat, depth=depth, time=time,
            lonlatdepth_dtype=lonlatdepth_dtype, pid_orig=pid_orig,
            partitions=partitions, ngrid=ngrids, **kwargs)

        # Initialize neighbor search data structure (used for interaction).
        if interaction_distance is not None:
            meshes = [g.mesh for g in fieldset.gridset.grids]
            # Assert all grids have the same mesh type
            assert np.all(np.array(meshes) == meshes[0])
            mesh_type = meshes[0]
            if mesh_type == "spherical":
                if len(self) < 1000:
                    interaction_class = BruteSphericalNeighborSearch
                else:
                    interaction_class = HashSphericalNeighborSearch
            elif mesh_type == "flat":
                if len(self) < 1000:
                    interaction_class = BruteFlatNeighborSearch
                else:
                    interaction_class = KDTreeFlatNeighborSearch
            else:
                assert False, ("Interaction is only possible on 'flat' and "
                               "'spherical' meshes")
            try:
                if len(interaction_distance) == 2:
                    inter_dist_vert, inter_dist_horiz = interaction_distance
                else:
                    inter_dist_vert = interaction_distance[0]
                    inter_dist_horiz = interaction_distance[0]
            except TypeError:
                inter_dist_vert = interaction_distance
                inter_dist_horiz = interaction_distance
            self._neighbor_tree = interaction_class(
                inter_dist_vert=inter_dist_vert,
                inter_dist_horiz=inter_dist_horiz,
                periodic_domain_zonal=periodic_domain_zonal)
        # End of neighbor search data structure initialization.

        if self.repeatdt:
            if len(time) > 0 and time[0] is None:
                self.repeat_starttime = time[0]
            else:
                if self._collection.data['time'][0] and not np.allclose(self._collection.data['time'], self._collection.data['time'][0]):
                    raise ValueError('All Particle.time should be the same when repeatdt is not None')
                self.repeat_starttime = copy(self._collection.data['time'][0])
            self.repeatlon = copy(self._collection.data['lon'])
            self.repeatlat = copy(self._collection.data['lat'])
            self.repeatdepth = copy(self._collection.data['depth'])
            for kwvar in kwargs:
                self.repeatkwargs[kwvar] = copy(self._collection.data[kwvar])

        if self.repeatdt:
            if MPI and self._collection.pu_indicators is not None:
                mpi_comm = MPI.COMM_WORLD
                mpi_rank = mpi_comm.Get_rank()
                self.repeatpid = pid_orig[self._collection.pu_indicators == mpi_rank]

        self.kernel = None

    def __del__(self):
        super(ParticleSetSOA, self).__del__()

    def _set_particle_vector(self, name, value):
        """Set attributes of all particles to new values.

        :param name: Name of the attribute (str).
        :param value: New value to set the attribute of the particles to.
        """
        self.collection._data[name][:] = value

    def _impute_release_times(self, default):
        """Set attribute 'time' to default if encountering NaN values.

        :param default: Default release time.
        :return: Minimum and maximum release times.
        """
        if np.any(np.isnan(self._collection.data['time'])):
            self._collection.data['time'][np.isnan(self._collection.data['time'])] = default
        return np.min(self._collection.data['time']), np.max(self._collection.data['time'])

    def data_indices(self, variable_name, compare_values, invert=False):
        """Get the indices of all particles where the value of
        `variable_name` equals (one of) `compare_values`.

        :param variable_name: Name of the variable to check.
        :param compare_values: Value or list of values to compare to.
        :param invert: Whether to invert the selection. I.e., when True,
                       return all indices that do not equal (one of)
                       `compare_values`.
        :return: Numpy array of indices that satisfy the test.
        """
        compare_values = np.array([compare_values, ]) if type(compare_values) not in [list, dict, np.ndarray] else compare_values
        return np.where(np.isin(self._collection.data[variable_name], compare_values, invert=invert))[0]

    def indexed_subset(self, indices):
        return ParticleCollectionIterableSOA(self._collection, subset=indices)

    def populate_indices(self):
        """Pre-populate guesses of particle xi/yi indices using a kdtree.

        This is only intended for curvilinear grids, where the initial index search
        may be quite expensive.
        """

        if self.fieldset is None:
            # we need to be attached to a fieldset to have a valid
            # gridset to search for indices
            return

        if KDTree is None:
            return
        else:
            for i, grid in enumerate(self.fieldset.gridset.grids):
                if not isinstance(grid, CurvilinearGrid):
                    continue

                tree_data = np.stack((grid.lon.flat, grid.lat.flat), axis=-1)
                tree = KDTree(tree_data)
                # stack all the particle positions for a single query
                pts = np.stack((self._collection.data['lon'], self._collection.data['lat']), axis=-1)
                # query datatype needs to match tree datatype
                _, idx = tree.query(pts.astype(tree_data.dtype))
                yi, xi = np.unravel_index(idx, grid.lon.shape)

                self._collection.data['xi'][:, i] = xi
                self._collection.data['yi'][:, i] = yi

    @property
    def error_particles(self):
        """Get an iterator over all particles that are in an error state.

        :return: Collection iterator over error particles.
        """
        error_indices = self.data_indices('state', [StateCode.Success, StateCode.Evaluate], invert=True)
        return ParticleCollectionIterableSOA(self._collection, subset=error_indices)

    def active_particles_mask(self, time, dt):
        active_indices = (time - self._collection.data['time'])/dt >= 0
        non_err_indices = np.isin(self._collection.data['state'], [StateCode.Success, StateCode.Evaluate])
        active_indices = np.logical_and(active_indices, non_err_indices)
        self._active_particle_idx = np.where(active_indices)[0]
        return active_indices

    @property
    def num_error_particles(self):
        """Get the number of particles that are in an error state.

        :return: The number of error particles.
        """
        return np.sum(np.isin(
            self._collection.data['state'],
            [StateCode.Success, StateCode.Evaluate], invert=True))

    def __getitem__(self, index):
        """Get a single particle by index"""
        return self._collection.get_single_by_index(index)

    def cstruct(self):
        """
        'cstruct' returns the ctypes mapping of the combined collections cstruct and the fieldset cstruct.
        This depends on the specific structure in question.
        """
        cstruct = self._collection.cstruct()
        return cstruct

    @property
    def ctypes_struct(self):
        return self.cstruct()

    @classmethod
    def monte_carlo_sample(cls, start_field, size, mode='monte_carlo'):
        """
        Converts a starting field into a monte-carlo sample of lons and lats.

        :param start_field: :mod:`parcels.fieldset.Field` object for initialising particles stochastically (horizontally)  according to the presented density field.

        returns list(lon), list(lat)
        """
        if mode == 'monte_carlo':
            data = start_field.data if isinstance(start_field.data, np.ndarray) else np.array(start_field.data)
            if start_field.interp_method == 'cgrid_tracer':
                p_interior = np.squeeze(data[0, 1:, 1:])
            else:  # if A-grid
                d = data
                p_interior = (d[0, :-1, :-1] + d[0, 1:, :-1] + d[0, :-1, 1:] + d[0, 1:, 1:])/4.
                p_interior = np.where(d[0, :-1, :-1] == 0, 0, p_interior)
                p_interior = np.where(d[0, 1:, :-1] == 0, 0, p_interior)
                p_interior = np.where(d[0, 1:, 1:] == 0, 0, p_interior)
                p_interior = np.where(d[0, :-1, 1:] == 0, 0, p_interior)
            p = np.reshape(p_interior, (1, p_interior.size))
            inds = np.random.choice(p_interior.size, size, replace=True, p=p[0] / np.sum(p))
            xsi = np.random.uniform(size=len(inds))
            eta = np.random.uniform(size=len(inds))
            j, i = np.unravel_index(inds, p_interior.shape)
            grid = start_field.grid
            lon, lat = ([], [])
            if grid.gtype in [GridCode.RectilinearZGrid, GridCode.RectilinearSGrid]:
                lon = grid.lon[i] + xsi * (grid.lon[i + 1] - grid.lon[i])
                lat = grid.lat[j] + eta * (grid.lat[j + 1] - grid.lat[j])
            else:
                lons = np.array([grid.lon[j, i], grid.lon[j, i+1], grid.lon[j+1, i+1], grid.lon[j+1, i]])
                if grid.mesh == 'spherical':
                    lons[1:] = np.where(lons[1:] - lons[0] > 180, lons[1:]-360, lons[1:])
                    lons[1:] = np.where(-lons[1:] + lons[0] > 180, lons[1:]+360, lons[1:])
                lon = (1-xsi)*(1-eta) * lons[0] +\
                    xsi*(1-eta) * lons[1] +\
                    xsi*eta * lons[2] +\
                    (1-xsi)*eta * lons[3]
                lat = (1-xsi)*(1-eta) * grid.lat[j, i] +\
                    xsi*(1-eta) * grid.lat[j, i+1] +\
                    xsi*eta * grid.lat[j+1, i+1] +\
                    (1-xsi)*eta * grid.lat[j+1, i]
            return list(lon), list(lat)
        else:
            raise NotImplementedError('Mode %s not implemented. Please use "monte carlo" algorithm instead.' % mode)

    @classmethod
    def from_particlefile(cls, fieldset, pclass, filename, restart=True, restarttime=None, repeatdt=None, lonlatdepth_dtype=None, **kwargs):
        """Initialise the ParticleSet from a netcdf ParticleFile.
        This creates a new ParticleSet based on locations of all particles written
        in a netcdf ParticleFile at a certain time. Particle IDs are preserved if restart=True

        :param fieldset: :mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        :param pclass: mod:`parcels.particle.JITParticle` or :mod:`parcels.particle.ScipyParticle`
                 object that defines custom particle
        :param filename: Name of the particlefile from which to read initial conditions
        :param restart: Boolean to signal if pset is used for a restart (default is True).
               In that case, Particle IDs are preserved.
        :param restarttime: time at which the Particles will be restarted. Default is the last time written.
               Alternatively, restarttime could be a time value (including np.datetime64) or
               a callable function such as np.nanmin. The last is useful when running with dt < 0.
        :param repeatdt: Optional interval (in seconds) on which to repeat the release of the ParticleSet
        :param lonlatdepth_dtype: Floating precision for lon, lat, depth particle coordinates.
               It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
               and np.float64 if the interpolation method is 'cgrid_velocity'
        """

        if repeatdt is not None:
            logger.warning('Note that the `repeatdt` argument is not retained from %s, and that '
                           'setting a new repeatdt will start particles from the _new_ particle '
                           'locations.' % filename)

        pfile = xr.open_dataset(str(filename), decode_cf=True)
        pfile_vars = [v for v in pfile.data_vars]

        vars = {}
        to_write = {}
        for v in pclass.getPType().variables:
            if v.name in pfile_vars:
                vars[v.name] = np.ma.filled(pfile.variables[v.name], np.nan)
            elif v.name not in ['xi', 'yi', 'zi', 'ti', 'dt', '_next_dt', 'depth', 'id', 'fileid', 'state'] \
                    and v.to_write:
                raise RuntimeError('Variable %s is in pclass but not in the particlefile' % v.name)
            to_write[v.name] = v.to_write
        vars['depth'] = np.ma.filled(pfile.variables['z'], np.nan)
        vars['id'] = np.ma.filled(pfile.variables['trajectory'], np.nan)

        if isinstance(vars['time'][0, 0], np.timedelta64):
            vars['time'] = np.array([t/np.timedelta64(1, 's') for t in vars['time']])

        if restarttime is None:
            restarttime = np.nanmax(vars['time'])
        elif callable(restarttime):
            restarttime = restarttime(vars['time'])
        else:
            restarttime = restarttime

        inds = np.where(vars['time'] == restarttime)
        for v in vars:
            if to_write[v] is True:
                vars[v] = vars[v][inds]
            elif to_write[v] == 'once':
                vars[v] = vars[v][inds[0]]
            if v not in ['lon', 'lat', 'depth', 'time', 'id']:
                kwargs[v] = vars[v]

        if restart:
            pclass.setLastID(0)  # reset to zero offset
        else:
            vars['id'] = None

        return cls(fieldset=fieldset, pclass=pclass, lon=vars['lon'], lat=vars['lat'],
                   depth=vars['depth'], time=vars['time'], pid_orig=vars['id'],
                   lonlatdepth_dtype=lonlatdepth_dtype, repeatdt=repeatdt, **kwargs)

    def to_dict(self, pfile, time, deleted_only=False):
        """
        Convert all Particle data from one time step to a python dictionary.
        :param pfile: ParticleFile object requesting the conversion
        :param time: Time at which to write ParticleSet
        :param deleted_only: Flag to write only the deleted Particles
        returns two dictionaries: one for all variables to be written each outputdt, and one for all variables to be written once
        """
        return self._collection.toDictionary(pfile=pfile, time=time,
                                             deleted_only=deleted_only)

    def compute_neighbor_tree(self, time, dt):
        active_mask = self.active_particles_mask(time, dt)

        self._values = np.vstack((
            self._collection.data['depth'],
            self._collection.data['lat'],
            self._collection.data['lon'],
        ))
        if self._dirty_neighbor:
            self._neighbor_tree.rebuild(self._values, active_mask=active_mask)
            self._dirty_neighbor = False
        else:
            self._neighbor_tree.update_values(self._values, new_active_mask=active_mask)

    def neighbors_by_index(self, particle_idx):
        neighbor_idx, distances = self._neighbor_tree.find_neighbors_by_idx(
            particle_idx)
        neighbor_idx = self._active_particle_idx[neighbor_idx]
        mask = (neighbor_idx != particle_idx)
        neighbor_idx = neighbor_idx[mask]
        if 'horiz_dist' in self._collection._ptype.variables:
            self._collection.data['vert_dist'][neighbor_idx] = distances[0, mask]
            self._collection.data['horiz_dist'][neighbor_idx] = distances[1, mask]
        return ParticleCollectionIterableSOA(self._collection, subset=neighbor_idx)

    def neighbors_by_coor(self, coor):
        neighbor_idx = self._neighbor_tree.find_neighbors_by_coor(coor)
        neighbor_ids = self._collection.data['id'][neighbor_idx]
        return neighbor_ids

    @property
    def size(self):
        # ==== to change at some point - len and size are different things ==== #
        return len(self._collection)

    def __repr__(self):
        return "\n".join([str(p) for p in self])

    def __len__(self):
        return len(self._collection)

    def __sizeof__(self):
        return sys.getsizeof(self._collection)

    def __iadd__(self, particles):
        """Add particles to the ParticleSet. Note that this is an
        incremental add, the particles will be added to the ParticleSet
        on which this function is called.

        :param particles: Another ParticleSet containing particles to add
                          to this one.
        :return: The current ParticleSet
        """
        self.add(particles)
        return self

    def __iter__(self):
        return super(ParticleSetSOA, self).__iter__()

    def iterator(self):
        return super(ParticleSetSOA, self).iterator()

    def add(self, particles):
        """Add particles to the ParticleSet. Note that this is an
        incremental add, the particles will be added to the ParticleSet
        on which this function is called.

        :param particles: Another ParticleSet containing particles to add to this one.
        :return: The current ParticleSet
        """
        if isinstance(particles, BaseParticleSet):
            particles = particles.collection
        self._collection += particles
        # Adding particles invalidates the neighbor search structure.
        self._dirty_neighbor = True
        return self

    def remove_indices(self, indices):
        """Method to remove particles from the ParticleSet, based on their `indices`"""
        # Removing particles invalidates the neighbor search structure.
        self._dirty_neighbor = True
        if type(indices) in [int, np.int32, np.intp]:
            self._collection.remove_single_by_index(indices)
        else:
            self._collection.remove_multi_by_indices(indices)

    def remove_booleanvector(self, indices):
        """Method to remove particles from the ParticleSet, based on an array of booleans"""
        # Removing particles invalidates the neighbor search structure.
        self._dirty_neighbor = True
        self.remove_indices(np.where(indices)[0])

    def density(self, field_name=None, particle_val=None, relative=False, area_scale=False):
        """Method to calculate the density of particles in a ParticleSet from their locations,
        through a 2D histogram.

        :param field: Optional :mod:`parcels.field.Field` object to calculate the histogram
                      on. Default is `fieldset.U`
        :param particle_val: Optional numpy-array of values to weigh each particle with,
                             or string name of particle variable to use weigh particles with.
                             Default is None, resulting in a value of 1 for each particle
        :param relative: Boolean to control whether the density is scaled by the total
                         weight of all particles. Default is False
        :param area_scale: Boolean to control whether the density is scaled by the area
                           (in m^2) of each grid cell. Default is False
        """

        field_name = field_name if field_name else "U"
        field = getattr(self.fieldset, field_name)

        f_str = """
def search_kernel(particle, fieldset, time):
    x = fieldset.{}[time, particle.depth, particle.lat, particle.lon]
        """.format(field_name)

        k = Kernel(
            self.fieldset,
            self._collection.ptype,
            funcname="search_kernel",
            funcvars=["particle", "fieldset", "time", "x"],
            funccode=f_str,
        )
        self.execute(pyfunc=k, runtime=0)

        if isinstance(particle_val, str):
            particle_val = self._collection._data[particle_val]
        else:
            particle_val = particle_val if particle_val else np.ones(self.size)
        density = np.zeros((field.grid.lat.size, field.grid.lon.size), dtype=np.float32)

        for i, p in enumerate(self):
            try:  # breaks if either p.xi, p.yi, p.zi, p.ti do not exist (in scipy) or field not in fieldset
                if p.ti[field.igrid] < 0:  # xi, yi, zi, ti, not initialised
                    raise('error')
                xi = p.xi[field.igrid]
                yi = p.yi[field.igrid]
            except:
                _, _, _, xi, yi, _ = field.search_indices(p.lon, p.lat, p.depth, 0, 0, search2D=True)
            density[yi, xi] += particle_val[i]

        if relative:
            density /= np.sum(particle_val)

        if area_scale:
            density /= field.cell_areas()

        return density

    def Kernel(self, pyfunc, c_include="", delete_cfiles=True):
        """Wrapper method to convert a `pyfunc` into a :class:`parcels.kernel.Kernel` object
        based on `fieldset` and `ptype` of the ParticleSet

        :param delete_cfiles: Boolean whether to delete the C-files after compilation in JIT mode (default is True)
        """
        return Kernel(self.fieldset, self.collection.ptype, pyfunc=pyfunc, c_include=c_include,
                      delete_cfiles=delete_cfiles)

    def InteractionKernel(self, pyfunc_inter):
        if pyfunc_inter is None:
            return None
        return InteractionKernelSOA(self.fieldset, self.collection.ptype, pyfunc=pyfunc_inter)

    def ParticleFile(self, *args, **kwargs):
        """Wrapper method to initialise a :class:`parcels.particlefile.ParticleFile`
        object from the ParticleSet"""
        return ParticleFile(*args, particleset=self, **kwargs)

    def set_variable_write_status(self, var, write_status):
        """
        Method to set the write status of a Variable
        :param var: Name of the variable (string)
        :param write_status: Write status of the variable (True, False or 'once')
        """
        self._collection.set_variable_write_status(var, write_status)

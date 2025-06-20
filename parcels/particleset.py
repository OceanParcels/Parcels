import sys
import warnings
from collections.abc import Iterable
from datetime import date, datetime, timedelta

import numpy as np
import xarray as xr
from scipy.spatial import KDTree
from tqdm import tqdm

from parcels._core.utils.time import TimeInterval
from parcels._reprs import particleset_repr
from parcels.application_kernels.advection import AdvectionRK4
from parcels.grid import GridType
from parcels.interaction.interactionkernel import InteractionKernel
from parcels.kernel import Kernel
from parcels.particle import Particle, Variable
from parcels.particledata import ParticleData, ParticleDataIterator
from parcels.particlefile import ParticleFile
from parcels.tools.converters import _get_cftime_calendars, convert_to_flat_array
from parcels.tools.loggers import logger
from parcels.tools.statuscodes import StatusCode
from parcels.tools.warnings import ParticleSetWarning

__all__ = ["ParticleSet"]


def _convert_to_reltime(time):
    """Check to determine if the value of the time parameter needs to be converted to a relative value (relative to the time_origin)."""
    if isinstance(time, np.datetime64) or (hasattr(time, "calendar") and time.calendar in _get_cftime_calendars()):
        return True
    return False


class ParticleSet:
    """Class for storing particle and executing kernel over them.

    Please note that this currently only supports fixed size particle sets, meaning that the particle set only
    holds the particles defined on construction. Individual particles can neither be added nor deleted individually,
    and individual particles can only be deleted as a set procedurally (i.e. by 'particle.delete()'-call during
    kernel execution).

    Parameters
    ----------
    fieldset :
        mod:`parcels.fieldset.FieldSet` object from which to sample velocity.
    pclass : parcels.particle.Particle
        Optional object that inherits from :mod:`parcels.particle.Particle` object that defines custom particle
    lon :
        List of initial longitude values for particles
    lat :
        List of initial latitude values for particles
    depth :
        Optional list of initial depth values for particles. Default is 0m
    time :
        Optional list of initial time values for particles. Default is fieldset.U.grid.time[0]
    repeatdt : datetime.timedelta or float, optional
        Optional interval on which to repeat the release of the ParticleSet. Either timedelta object, or float in seconds.
    lonlatdepth_dtype :
        Floating precision for lon, lat, depth particle coordinates.
        It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
        and np.float64 if the interpolation method is 'cgrid_velocity'
    pid_orig :
        Optional list of (offsets for) the particle IDs
    partition_function :
        Function to use for partitioning particles over processors. Default is to use kMeans
    periodic_domain_zonal :
        Zonal domain size, used to apply zonally periodic boundaries for particle-particle
        interaction. If None, no zonally periodic boundaries are applied

        Other Variables can be initialised using further arguments (e.g. v=... for a Variable named 'v')
    """

    def __init__(
        self,
        fieldset,
        pclass=Particle,
        lon=None,
        lat=None,
        depth=None,
        time=None,
        lonlatdepth_dtype=None,
        pid_orig=None,
        **kwargs,
    ):
        self.particledata = None
        self._repeat_starttime = None
        self._repeatlon = None
        self._repeatlat = None
        self._repeatdepth = None
        self._repeatpclass = None
        self._repeatkwargs = None
        self._kernel = None
        self._interaction_kernel = None

        self.fieldset = fieldset
        self._pclass = pclass

        # ==== first: create a new subclass of the pclass that includes the required variables ==== #
        # ==== see dynamic-instantiation trick here: https://www.python-course.eu/python3_classes_and_type.php ==== #
        class_name = pclass.__name__
        array_class = None
        if class_name not in dir():

            def ArrayClass_init(self, *args, **kwargs):
                fieldset = kwargs.get("fieldset", None)
                ngrids = kwargs.get("ngrids", None)
                if type(self).ngrids.initial < 0:
                    numgrids = ngrids
                    if numgrids is None and fieldset is not None:
                        numgrids = fieldset.gridset_size
                    assert numgrids is not None, "Neither fieldsets nor number of grids are specified - exiting."
                    type(self).ngrids.initial = numgrids
                self.ngrids = type(self).ngrids.initial
                if self.ngrids >= 0:
                    self.ei = np.zeros(self.ngrids, dtype=np.int32)
                super(type(self), self).__init__(*args, **kwargs)

            array_class_vdict = {
                "ngrids": Variable("ngrids", dtype=np.int32, to_write=False, initial=-1),
                "ei": Variable("ei", dtype=np.int32, to_write=False),
                "__init__": ArrayClass_init,
            }
            array_class = type(class_name, (pclass,), array_class_vdict)
        else:
            array_class = locals()[class_name]
        # ==== dynamic re-classing completed ==== #
        _pclass = array_class

        lon = np.empty(shape=0) if lon is None else convert_to_flat_array(lon)
        lat = np.empty(shape=0) if lat is None else convert_to_flat_array(lat)

        if isinstance(pid_orig, (type(None), bool)):
            pid_orig = np.arange(lon.size)

        if depth is None:
            mindepth = 0
            for field in self.fieldset.fields.values():
                if field.grid.depth is not None:
                    mindepth = min(mindepth, field.grid.depth[0])
            depth = np.ones(lon.size) * mindepth
        else:
            depth = convert_to_flat_array(depth)
        assert lon.size == lat.size and lon.size == depth.size, "lon, lat, depth don't all have the same lenghts"

        time = convert_to_flat_array(time)
        time = np.repeat(time, lon.size) if time.size == 1 else time

        if time.size > 0 and type(time[0]) in [datetime, date]:
            time = np.array([np.datetime64(t) for t in time])
        if time.size > 0 and isinstance(time[0], np.timedelta64) and not self.time_origin:
            raise NotImplementedError("If fieldset.time_origin is not a date, time of a particle must be a double")

        time = np.array([self.time_origin.reltime(t) if _convert_to_reltime(t) else t for t in time])
        assert lon.size == time.size, "time and positions (lon, lat, depth) do not have the same lengths."
        if fieldset.time_interval:
            _warn_particle_times_outside_fieldset_time_bounds(time, fieldset.time_interval)

        if lonlatdepth_dtype is None:
            lonlatdepth_dtype = self.lonlatdepth_dtype_from_field_interp_method(fieldset.U)
        assert lonlatdepth_dtype in [
            np.float32,
            np.float64,
        ], "lon lat depth precision should be set to either np.float32 or np.float64"

        for kwvar in kwargs:
            if kwvar not in ["partition_function"]:
                kwargs[kwvar] = convert_to_flat_array(kwargs[kwvar])
                assert (
                    lon.size == kwargs[kwvar].size
                ), f"{kwvar} and positions (lon, lat, depth) don't have the same lengths."

        self.particledata = ParticleData(
            _pclass,
            lon=lon,
            lat=lat,
            depth=depth,
            time=time,
            lonlatdepth_dtype=lonlatdepth_dtype,
            pid_orig=pid_orig,
            ngrid=fieldset.gridset_size,
            **kwargs,
        )

        self._kernel = None

    def __del__(self):
        if self.particledata is not None and isinstance(self.particledata, ParticleData):
            del self.particledata
        self.particledata = None

    def __iter__(self):
        return iter(self.particledata)

    def __getattr__(self, name):
        """
        Access a single property of all particles.

        Parameters
        ----------
        name : str
            Name of the property
        """
        for v in self.particledata.ptype.variables:
            if v.name == name:
                return getattr(self.particledata, name)
        if name in self.__dict__ and name[0] != "_":
            return self.__dict__[name]
        else:
            return False

    def __getitem__(self, index):
        """Get a single particle by index."""
        return self.particledata.get_single_by_index(index)

    @staticmethod
    def lonlatdepth_dtype_from_field_interp_method(field):
        if field.interp_method == "cgrid_velocity":
            return np.float64
        return np.float32

    @property
    def size(self):
        # ==== to change at some point - len and size are different things ==== #
        return len(self.particledata)

    @property
    def pclass(self):
        return self._pclass

    def __repr__(self):
        return particleset_repr(self)

    def __len__(self):
        return len(self.particledata)

    def __sizeof__(self):
        return sys.getsizeof(self.particledata)

    def add(self, particles):
        """Add particles to the ParticleSet. Note that this is an
        incremental add, the particles will be added to the ParticleSet
        on which this function is called.

        Parameters
        ----------
        particles :
            Another ParticleSet containing particles to add to this one.

        Returns
        -------
        type
            The current ParticleSet

        """
        if isinstance(particles, type(self)):
            particles = particles.particledata
        self.particledata += particles
        # Adding particles invalidates the neighbor search structure.
        self._dirty_neighbor = True
        return self

    def __iadd__(self, particles):
        """Add particles to the ParticleSet.

        Note that this is an incremental add, the particles will be added to the ParticleSet
        on which this function is called.

        Parameters
        ----------
        particles :
            Another ParticleSet containing particles to add to this one.

        Returns
        -------
        type
            The current ParticleSet
        """
        self.add(particles)
        return self

    def remove_indices(self, indices):
        """Method to remove particles from the ParticleSet, based on their `indices`."""
        # Removing particles invalidates the neighbor search structure.
        self._dirty_neighbor = True
        if type(indices) in [int, np.int32, np.intp]:
            self.particledata.remove_single_by_index(indices)
        else:
            self.particledata.remove_multi_by_indices(indices)

    def remove_booleanvector(self, indices):
        """Method to remove particles from the ParticleSet, based on an array of booleans."""
        # Removing particles invalidates the neighbor search structure.
        self._dirty_neighbor = True
        self.remove_indices(np.where(indices)[0])

    def _active_particles_mask(self, time, dt):
        active_indices = (time - self.particledata.data["time"]) / dt >= 0
        non_err_indices = np.isin(self.particledata.data["state"], [StatusCode.Success, StatusCode.Evaluate])
        active_indices = np.logical_and(active_indices, non_err_indices)
        self._active_particle_idx = np.where(active_indices)[0]
        return active_indices

    def _compute_neighbor_tree(self, time, dt):
        active_mask = self._active_particles_mask(time, dt)

        self._values = np.vstack(
            (
                self.particledata.data["depth"],
                self.particledata.data["lat"],
                self.particledata.data["lon"],
            )
        )
        if self._dirty_neighbor:
            self._neighbor_tree.rebuild(self._values, active_mask=active_mask)
            self._dirty_neighbor = False
        else:
            self._neighbor_tree.update_values(self._values, new_active_mask=active_mask)

    def _neighbors_by_index(self, particle_idx):
        neighbor_idx, distances = self._neighbor_tree.find_neighbors_by_idx(particle_idx)
        neighbor_idx = self._active_particle_idx[neighbor_idx]
        mask = neighbor_idx != particle_idx
        neighbor_idx = neighbor_idx[mask]
        if "horiz_dist" in self.particledata._ptype.variables:
            self.particledata.data["vert_dist"][neighbor_idx] = distances[0, mask]
            self.particledata.data["horiz_dist"][neighbor_idx] = distances[1, mask]
        return ParticleDataIterator(self.particledata, subset=neighbor_idx)

    def _neighbors_by_coor(self, coor):
        neighbor_idx = self._neighbor_tree.find_neighbors_by_coor(coor)
        neighbor_ids = self.particledata.data["id"][neighbor_idx]
        return neighbor_ids

    # TODO: This method is only tested in tutorial notebook. Add unit test?
    def populate_indices(self):
        """Pre-populate guesses of particle ei (element id) indices using a kdtree.

        This is only intended for curvilinear grids, where the initial index search
        may be quite expensive.
        """
        for i, grid in enumerate(self.fieldset.gridset.grids):
            if grid._gtype not in [GridType.CurvilinearZGrid, GridType.CurvilinearSGrid]:
                continue

            tree_data = np.stack((grid.lon.flat, grid.lat.flat), axis=-1)
            IN = np.all(~np.isnan(tree_data), axis=1)
            tree = KDTree(tree_data[IN, :])
            # stack all the particle positions for a single query
            pts = np.stack((self.particledata.data["lon"], self.particledata.data["lat"]), axis=-1)
            # query datatype needs to match tree datatype
            _, idx_nan = tree.query(pts.astype(tree_data.dtype))

            idx = np.where(IN)[0][idx_nan]

            self.particledata.data["ei"][:, i] = idx  # assumes that we are in the surface layer (zi=0)

    @classmethod
    def from_list(
        cls, fieldset, pclass, lon, lat, depth=None, time=None, repeatdt=None, lonlatdepth_dtype=None, **kwargs
    ):
        """Initialise the ParticleSet from lists of lon and lat.

        Parameters
        ----------
        fieldset :
            mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        pclass :
            Particle class. May be a parcels.particle.Particle class as defined in parcels, or a subclass defining a custom particle.
        lon :
            List of initial longitude values for particles
        lat :
            List of initial latitude values for particles
        depth :
            Optional list of initial depth values for particles. Default is 0m
        time :
            Optional list of start time values for particles. Default is fieldset.U.time[0]
        repeatdt :
            Optional interval (in seconds) on which to repeat the release of the ParticleSet (Default value = None)
        lonlatdepth_dtype :
            Floating precision for lon, lat, depth particle coordinates.
            It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
            and np.float64 if the interpolation method is 'cgrid_velocity'
            Other Variables can be initialised using further arguments (e.g. v=... for a Variable named 'v')
        **kwargs :
            Keyword arguments passed to the particleset constructor.
        """
        return cls(
            fieldset=fieldset,
            pclass=pclass,
            lon=lon,
            lat=lat,
            depth=depth,
            time=time,
            repeatdt=repeatdt,
            lonlatdepth_dtype=lonlatdepth_dtype,
            **kwargs,
        )

    @classmethod
    def from_line(
        cls,
        fieldset,
        pclass,
        start,
        finish,
        size,
        depth=None,
        time=None,
        repeatdt=None,
        lonlatdepth_dtype=None,
        **kwargs,
    ):
        """Create a particleset in the shape of a line (according to a cartesian grid).

        Initialise the ParticleSet from start/finish coordinates with equidistant spacing
        Note that this method uses simple numpy.linspace calls and does not take into account
        great circles, so may not be a exact on a globe

        Parameters
        ----------
        fieldset :
            mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        pclass :
            Particle class. May be a parcels.particle.Particle as defined in parcels, or a subclass defining a custom particle.
        start :
            Start point (longitude, latitude) for initialisation of particles on a straight line.
        finish :
            End point (longitude, latitude) for initialisation of particles on a straight line.
        size :
            Initial size of particle set
        depth :
            Optional list of initial depth values for particles. Default is 0m
        time :
            Optional start time value for particles. Default is fieldset.U.time[0]
        repeatdt :
            Optional interval (in seconds) on which to repeat the release of the ParticleSet (Default value = None)
        lonlatdepth_dtype :
            Floating precision for lon, lat, depth particle coordinates.
            It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
            and np.float64 if the interpolation method is 'cgrid_velocity'
        """
        lon = np.linspace(start[0], finish[0], size)
        lat = np.linspace(start[1], finish[1], size)
        if type(depth) in [int, float]:
            depth = [depth] * size
        return cls(
            fieldset=fieldset,
            pclass=pclass,
            lon=lon,
            lat=lat,
            depth=depth,
            time=time,
            repeatdt=repeatdt,
            lonlatdepth_dtype=lonlatdepth_dtype,
            **kwargs,
        )

    @classmethod
    def _monte_carlo_sample(cls, start_field, size, mode="monte_carlo"):
        """Converts a starting field into a monte-carlo sample of lons and lats.

        Parameters
        ----------
        start_field : parcels.field.Field
            mod:`parcels.fieldset.Field` object for initialising particles stochastically (horizontally)  according to the presented density field.
        size :

        mode :
             (Default value = 'monte_carlo')

        Returns
        -------
        list of float
            A list of longitude values.
        list of float
            A list of latitude values.
        """
        if mode == "monte_carlo":
            data = start_field.data if isinstance(start_field.data, np.ndarray) else np.array(start_field.data)
            if start_field.interp_method == "cgrid_tracer":
                p_interior = np.squeeze(data[0, 1:, 1:])
            else:  # if A-grid
                d = data
                p_interior = (d[0, :-1, :-1] + d[0, 1:, :-1] + d[0, :-1, 1:] + d[0, 1:, 1:]) / 4.0
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
            if grid._gtype in [GridType.RectilinearZGrid, GridType.RectilinearSGrid]:
                lon = grid.lon[i] + xsi * (grid.lon[i + 1] - grid.lon[i])
                lat = grid.lat[j] + eta * (grid.lat[j + 1] - grid.lat[j])
            else:
                lons = np.array([grid.lon[j, i], grid.lon[j, i + 1], grid.lon[j + 1, i + 1], grid.lon[j + 1, i]])
                if grid.mesh == "spherical":
                    lons[1:] = np.where(lons[1:] - lons[0] > 180, lons[1:] - 360, lons[1:])
                    lons[1:] = np.where(-lons[1:] + lons[0] > 180, lons[1:] + 360, lons[1:])
                lon = (
                    (1 - xsi) * (1 - eta) * lons[0]
                    + xsi * (1 - eta) * lons[1]
                    + xsi * eta * lons[2]
                    + (1 - xsi) * eta * lons[3]
                )
                lat = (
                    (1 - xsi) * (1 - eta) * grid.lat[j, i]
                    + xsi * (1 - eta) * grid.lat[j, i + 1]
                    + xsi * eta * grid.lat[j + 1, i + 1]
                    + (1 - xsi) * eta * grid.lat[j + 1, i]
                )
            return list(lat), list(lon)
        else:
            raise NotImplementedError(f'Mode {mode} not implemented. Please use "monte carlo" algorithm instead.')

    @classmethod
    def from_field(
        cls,
        fieldset,
        pclass,
        start_field,
        size,
        mode="monte_carlo",
        depth=None,
        time=None,
        repeatdt=None,
        lonlatdepth_dtype=None,
    ):
        """Initialise the ParticleSet randomly drawn according to distribution from a field.

        Parameters
        ----------
        fieldset : parcels.fieldset.FieldSet
            mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        pclass :
            Particle class. May be a parcels.particle.Particle class as defined in parcels, or a subclass defining a custom particle.
        start_field : parcels.field.Field
            Field for initialising particles stochastically (horizontally)  according to the presented density field.
        size :
            Initial size of particle set
        mode :
            Type of random sampling. Currently only 'monte_carlo' is implemented (Default value = 'monte_carlo')
        depth :
            Optional list of initial depth values for particles. Default is 0m
        time :
            Optional start time value for particles. Default is fieldset.U.time[0]
        repeatdt :
            Optional interval (in seconds) on which to repeat the release of the ParticleSet (Default value = None)
        lonlatdepth_dtype :
            Floating precision for lon, lat, depth particle coordinates.
            It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
            and np.float64 if the interpolation method is 'cgrid_velocity'
        """
        lat, lon = cls._monte_carlo_sample(start_field, size, mode)

        return cls(
            fieldset=fieldset,
            pclass=pclass,
            lon=lon,
            lat=lat,
            depth=depth,
            time=time,
            lonlatdepth_dtype=lonlatdepth_dtype,
            repeatdt=repeatdt,
        )

    @classmethod
    def from_particlefile(
        cls, fieldset, pclass, filename, restart=True, restarttime=None, repeatdt=None, lonlatdepth_dtype=None, **kwargs
    ):
        """Initialise the ParticleSet from a zarr ParticleFile.
        This creates a new ParticleSet based on locations of all particles written
        in a zarr ParticleFile at a certain time. Particle IDs are preserved if restart=True

        Parameters
        ----------
        fieldset : parcels.fieldset.FieldSet
            mod:`parcels.fieldset.FieldSet` object from which to sample velocity
        pclass :
            Particle class. May be a parcels.particle.Particle class as defined in parcels, or a subclass defining a custom particle.
        filename : str
            Name of the particlefile from which to read initial conditions
        restart : bool
            BSignal if pset is used for a restart (default is True).
            In that case, Particle IDs are preserved.
        restarttime :
            time at which the Particles will be restarted. Default is the last time written.
            Alternatively, restarttime could be a time value (including np.datetime64) or
            a callable function such as np.nanmin. The last is useful when running with dt < 0.
        repeatdt : datetime.timedelta or float, optional
            Optional interval on which to repeat the release of the ParticleSet. Either timedelta object, or float in seconds.
        lonlatdepth_dtype :
            Floating precision for lon, lat, depth particle coordinates.
            It is either np.float32 or np.float64. Default is np.float32 if fieldset.U.interp_method is 'linear'
            and np.float64 if the interpolation method is 'cgrid_velocity'
        **kwargs :
            Keyword arguments passed to the particleset constructor.
        """
        if repeatdt is not None:
            warnings.warn(
                f"Note that the `repeatdt` argument is not retained from {filename}, and that "
                "setting a new repeatdt will start particles from the _new_ particle "
                "locations.",
                ParticleSetWarning,
                stacklevel=2,
            )

        pfile = xr.open_zarr(str(filename))
        pfile_vars = [v for v in pfile.data_vars]

        vars = {}
        to_write = {}
        for v in pclass.getPType().variables:
            if v.name in pfile_vars:
                vars[v.name] = np.ma.filled(pfile.variables[v.name], np.nan)
            elif (
                v.name
                not in [
                    "ei",
                    "dt",
                    "depth",
                    "id",
                    "obs_written",
                    "state",
                    "lon_nextloop",
                    "lat_nextloop",
                    "depth_nextloop",
                    "time_nextloop",
                ]
                and v.to_write
            ):
                raise RuntimeError(f"Variable {v.name} is in pclass but not in the particlefile")
            to_write[v.name] = v.to_write
        vars["depth"] = np.ma.filled(pfile.variables["z"], np.nan)
        vars["id"] = np.ma.filled(pfile.variables["trajectory"], np.nan)

        for v in ["lon", "lat", "depth", "time"]:
            to_write[v] = True

        if isinstance(vars["time"][0, 0], np.timedelta64):
            vars["time"] = np.array([t / np.timedelta64(1, "s") for t in vars["time"]])

        if restarttime is None:
            restarttime = np.nanmax(vars["time"])
        elif callable(restarttime):
            restarttime = restarttime(vars["time"])
        else:
            restarttime = restarttime

        inds = np.where(vars["time"] == restarttime)
        for v in vars:
            if to_write[v] is True:
                vars[v] = vars[v][inds]
            elif to_write[v] == "once":
                vars[v] = vars[v][inds[0]]
            if v not in ["lon", "lat", "depth", "time", "id"]:
                kwargs[v] = vars[v]

        if restart:
            pclass.setLastID(0)  # reset to zero offset
        else:
            vars["id"] = None

        return cls(
            fieldset=fieldset,
            pclass=pclass,
            lon=vars["lon"],
            lat=vars["lat"],
            depth=vars["depth"],
            time=vars["time"],
            pid_orig=vars["id"],
            lonlatdepth_dtype=lonlatdepth_dtype,
            repeatdt=repeatdt,
            **kwargs,
        )

    def Kernel(self, pyfunc):
        """Wrapper method to convert a `pyfunc` into a :class:`parcels.kernel.Kernel` object.

        Conversion is based on `fieldset` and `ptype` of the ParticleSet.

        Parameters
        ----------
        pyfunc : function or list of functions
            Python function to convert into kernel. If a list of functions is provided,
            the functions will be converted to kernels and combined into a single kernel.
        """
        if isinstance(pyfunc, list):
            return Kernel.from_list(
                self.fieldset,
                self.particledata.ptype,
                pyfunc,
            )
        return Kernel(
            self.fieldset,
            self.particledata.ptype,
            pyfunc=pyfunc,
        )

    def InteractionKernel(self, pyfunc_inter):
        if pyfunc_inter is None:
            return None
        return InteractionKernel(self.fieldset, self.particledata.ptype, pyfunc=pyfunc_inter)

    def ParticleFile(self, *args, **kwargs):
        """Wrapper method to initialise a :class:`parcels.particlefile.ParticleFile` object from the ParticleSet."""
        return ParticleFile(*args, particleset=self, **kwargs)

    def data_indices(self, variable_name, compare_values, invert=False):
        """Get the indices of all particles where the value of `variable_name` equals (one of) `compare_values`.

        Parameters
        ----------
        variable_name : str
            Name of the variable to check.
        compare_values :
            Value or list of values to compare to.
        invert :
            Whether to invert the selection. I.e., when True,
            return all indices that do not equal (one of)
            `compare_values`. (Default value = False)

        Returns
        -------
        np.ndarray
            Numpy array of indices that satisfy the test.

        """
        compare_values = (
            np.array([compare_values]) if type(compare_values) not in [list, dict, np.ndarray] else compare_values
        )
        return np.where(np.isin(self.particledata.data[variable_name], compare_values, invert=invert))[0]

    @property
    def _error_particles(self):
        """Get an iterator over all particles that are in an error state.

        Returns
        -------
        iterator
            ParticleDataIterator over error particles.
        """
        error_indices = self.data_indices("state", [StatusCode.Success, StatusCode.Evaluate], invert=True)
        return ParticleDataIterator(self.particledata, subset=error_indices)

    @property
    def _num_error_particles(self):
        """Get the number of particles that are in an error state.

        Returns
        -------
        int
            Number of error particles.
        """
        return np.sum(np.isin(self.particledata.data["state"], [StatusCode.Success, StatusCode.Evaluate], invert=True))

    def set_variable_write_status(self, var, write_status):
        """Method to set the write status of a Variable.

        Parameters
        ----------
        var :
            Name of the variable (string)
        write_status :
            Write status of the variable (True, False or 'once')
        """
        self.particledata.set_variable_write_status(var, write_status)

    def execute(
        self,
        endtime: timedelta | datetime,
        dt: np.float64 | np.float32 | timedelta,
        pyfunc=AdvectionRK4,
        output_file=None,
        verbose_progress=True,
    ):
        """Execute a given kernel function over the particle set for multiple timesteps.

        Optionally also provide sub-timestepping
        for particle output.

        Parameters
        ----------
        pyfunc :
            Kernel function to execute. This can be the name of a
            defined Python function or a :class:`parcels.kernel.Kernel` object.
            Kernels can be concatenated using the + operator (Default value = AdvectionRK4)
        endtime (datetime.datetime or np.timedelta64): :
            End time for the timestepping loop. If a timedelta is provided, it is interpreted as the total simulation time.
            If a datetime is provided, it is interpreted as the end time of the simulation.
        dt (timedelta):
            Timestep interval (in seconds) to be passed to the kernel.
            It is either a timedelta object or a double.
            Use a negative value for a backward-in-time simulation. (Default value = 1 second)
        output_file :
            mod:`parcels.particlefile.ParticleFile` object for particle output (Default value = None)
        verbose_progress : bool
            Boolean for providing a progress bar for the kernel execution loop. (Default value = True)

        Notes
        -----
        ``ParticleSet.execute()`` acts as the main entrypoint for simulations, and provides the simulation time-loop. This method encapsulates the logic controlling the switching between kernel execution, output file writing, reading in fields for new timesteps, adding new particles to the simulation domain, stopping the simulation, and executing custom functions (``postIterationCallbacks`` provided by the user).
        """
        # check if particleset is empty. If so, return immediately
        if len(self) == 0:
            return

        # check if pyfunc has changed since last generation. If so, regenerate
        if self._kernel is None or (self._kernel.pyfunc is not pyfunc and self._kernel is not pyfunc):
            # Generate and store Kernel
            if isinstance(pyfunc, Kernel):
                self._kernel = pyfunc
            else:
                self._kernel = self.Kernel(pyfunc)

        if output_file:
            output_file.metadata["parcels_kernels"] = self._kernel.name

        # The fieldset time intervale defines the extent of time that is allowed to be
        # simulated. If `fieldset.time_interval` is not None, it will be used to determine the endtime (the min of endtime or fieldset.time_interval[1]).
        # If `fieldset.time_interval` is None, the endtime will be determined by the
        # `endtime` parameter or the fieldset's time dimension.
        # Time parameters for the main for loop are converted to floats, since the interpolation kernels expect float objects for time
        # The initial time (in float point) representation is t0=0.0 and time is interpreted as relative to the start of the time interval
        fieldset_timeinterval = self.fieldset.time_interval

        if fieldset_timeinterval is None:
            if isinstance(endtime, datetime):
                raise NotImplementedError(
                    "If fieldset.time_interval is None, endtime must be a timedelta not a datetime"
                )
            duration = endtime.total_seconds()  # converts timedelta to seconds as float64

        else:
            # Get the particle time interval
            if isinstance(endtime, datetime):
                simulation_endtime = min(fieldset_timeinterval[1], endtime)
                if simulation_endtime < fieldset_timeinterval[1]:
                    print(
                        f"Simulation endtime is limited by fieldset.time_interval. End time adjusted to {simulation_endtime}"
                    )
                duration = (simulation_endtime - fieldset_timeinterval[0]).total_seconds()

            else:
                duration = endtime.total_seconds()

        if isinstance(dt, timedelta):
            dt = dt.total_seconds()  # convert to seconds as float64

        outputdt = output_file.outputdt if output_file else None

        self.particledata._data["dt"][:] = dt

        # Set up pbar
        if output_file:
            logger.info(f"Output files are stored in {output_file.fname}.")

        if verbose_progress:
            pbar = tqdm(total=abs(duration), file=sys.stdout)

        if output_file:
            next_output = outputdt
        else:
            next_output = np.inf * np.sign(dt)

        tol = 1e-12
        time = 0.0
        while time < duration and dt > 0:  # Forward in time only for now
            # Check if we can fast-forward to the next time needed for the particles
            # if dt > 0:
            #     skip_kernel = True if duration > (time + dt) else False
            # else:
            #     skip_kernel = True if max(self.time) < (time + dt) else False
            t0 = time
            next_time = t0 + dt
            res = self._kernel.execute(self, endtime=next_time, dt=dt)
            if res == StatusCode.StopAllExecution:
                return StatusCode.StopAllExecution

            # End of interaction specific code
            time = next_time

            if abs(time - next_output) < tol:
                if output_file:
                    output_file.write(self, t0)
                if np.isfinite(outputdt):
                    next_output += outputdt * np.sign(dt)

            if verbose_progress:
                pbar.update(abs(dt))

        if verbose_progress:
            pbar.close()


def _warn_outputdt_release_desync(outputdt: float, starttime: float, release_times: Iterable[float]):
    """Gives the user a warning if the release time isn't a multiple of outputdt."""
    if any((np.isfinite(t) and (t - starttime) % outputdt != 0) for t in release_times):
        warnings.warn(
            "Some of the particles have a start time difference that is not a multiple of outputdt. "
            "This could cause the first output of some of the particles that start later "
            "in the simulation to be at a different time than expected.",
            ParticleSetWarning,
            stacklevel=2,
        )


def _warn_particle_times_outside_fieldset_time_bounds(release_times: np.ndarray, time: np.ndarray | TimeInterval):
    if np.any(release_times):
        if np.any(release_times < time[0]):
            warnings.warn(
                "Some particles are set to be released outside the FieldSet's executable time domain.",
                ParticleSetWarning,
                stacklevel=2,
            )
        if np.any(release_times > time[-1]):
            warnings.warn(
                "Some particles are set to be released after the fieldset's last time and the fields are not constant in time.",
                ParticleSetWarning,
                stacklevel=2,
            )

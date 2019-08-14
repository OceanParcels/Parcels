from parcels.field import Field, VectorField, SummedField, NestedField
from parcels.gridset import GridSet
from parcels.grid import Grid
from parcels.tools.loggers import logger
from parcels.tools.converters import TimeConverter
import numpy as np
from os import path
from glob import glob
from copy import deepcopy


__all__ = ['FieldSet']


class FieldSet(object):
    """FieldSet class that holds hydrodynamic data needed to execute particles

    :param U: :class:`parcels.field.Field` object for zonal velocity component
    :param V: :class:`parcels.field.Field` object for meridional velocity component
    :param fields: Dictionary of additional :class:`parcels.field.Field` objects
    """
    def __init__(self, U, V, fields=None):
        self.gridset = GridSet()
        if U:
            self.add_field(U, 'U')
            self.time_origin = self.U.grid.time_origin if isinstance(self.U, Field) else self.U[0].grid.time_origin
        if V:
            self.add_field(V, 'V')

        # Add additional fields as attributes
        if fields:
            for name, field in fields.items():
                self.add_field(field, name)

        self.compute_on_defer = None

    @staticmethod
    def checkvaliddimensionsdict(dims):
        for d in dims:
            if d not in ['lon', 'lat', 'depth', 'time']:
                raise NameError('%s is not a valid key in the dimensions dictionary' % d)

    @classmethod
    def from_data(cls, data, dimensions, transpose=False, mesh='spherical',
                  allow_time_extrapolation=None, time_periodic=False, **kwargs):
        """Initialise FieldSet object from raw data

        :param data: Dictionary mapping field names to numpy arrays.
               Note that at least a 'U' and 'V' numpy array need to be given, and that
               the built-in Advection kernels assume that U and V are in m/s

               1. If data shape is [xdim, ydim], [xdim, ydim, zdim], [xdim, ydim, tdim] or [xdim, ydim, zdim, tdim],
                  whichever is relevant for the dataset, use the flag transpose=True
               2. If data shape is [ydim, xdim], [zdim, ydim, xdim], [tdim, ydim, xdim] or [tdim, zdim, ydim, xdim],
                  use the flag transpose=False (default value)
               3. If data has any other shape, you first need to reorder it
        :param dimensions: Dictionary mapping field dimensions (lon,
               lat, depth, time) to numpy arrays.
               Note that dimensions can also be a dictionary of dictionaries if
               dimension names are different for each variable
               (e.g. dimensions['U'], dimensions['V'], etc).
        :param transpose: Boolean whether to transpose data on read-in
        :param mesh: String indicating the type of mesh coordinates and
               units used during velocity interpolation, see also https://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_unitconverters.ipynb:

               1. spherical (default): Lat and lon in degree, with a
                  correction for zonal velocity U near the poles.
               2. flat: No conversion, lat/lon are assumed to be in m.
        :param allow_time_extrapolation: boolean whether to allow for extrapolation
               (i.e. beyond the last available time snapshot)
               Default is False if dimensions includes time, else True
        :param time_periodic: boolean whether to loop periodically over the time component of the FieldSet
               This flag overrides the allow_time_interpolation and sets it to False
        """

        fields = {}
        for name, datafld in data.items():
            # Use dimensions[name] if dimensions is a dict of dicts
            dims = dimensions[name] if name in dimensions else dimensions
            cls.checkvaliddimensionsdict(dims)

            if allow_time_extrapolation is None:
                allow_time_extrapolation = False if 'time' in dims else True

            lon = dims['lon']
            lat = dims['lat']
            depth = np.zeros(1, dtype=np.float32) if 'depth' not in dims else dims['depth']
            time = np.zeros(1, dtype=np.float64) if 'time' not in dims else dims['time']
            time = np.array(time) if not isinstance(time, np.ndarray) else time
            if isinstance(time[0], np.datetime64):
                time_origin = TimeConverter(time[0])
                time = np.array([time_origin.reltime(t) for t in time])
            else:
                time_origin = TimeConverter(0)
            grid = Grid.create_grid(lon, lat, depth, time, time_origin=time_origin, mesh=mesh)
            if 'creation_log' not in kwargs.keys():
                kwargs['creation_log'] = 'from_data'

            fields[name] = Field(name, datafld, grid=grid, transpose=transpose,
                                 allow_time_extrapolation=allow_time_extrapolation, time_periodic=time_periodic, **kwargs)
        u = fields.pop('U', None)
        v = fields.pop('V', None)
        return cls(u, v, fields=fields)

    def add_field(self, field, name=None):
        """Add a :class:`parcels.field.Field` object to the FieldSet

        :param field: :class:`parcels.field.Field` object to be added
        :param name: Name of the :class:`parcels.field.Field` object to be added
        """
        name = field.name if name is None else name
        if hasattr(self, name):  # check if Field with same name already exists when adding new Field
            raise RuntimeError("FieldSet already has a Field with name '%s'" % name)
        if isinstance(field, SummedField):
            setattr(self, name, field)
            field.name = name
            for fld in field:
                self.gridset.add_grid(fld)
                fld.fieldset = self
        elif isinstance(field, NestedField):
            setattr(self, name, field)
            for fld in field:
                self.gridset.add_grid(fld)
                fld.fieldset = self
        elif isinstance(field, list):
            raise NotImplementedError('FieldLists have been replaced by SummedFields. Use the + operator instead of []')
        else:
            setattr(self, name, field)
            self.gridset.add_grid(field)
            field.fieldset = self

    def add_vector_field(self, vfield):
        """Add a :class:`parcels.field.VectorField` object to the FieldSet

        :param vfield: :class:`parcels.field.VectorField` object to be added
        """
        setattr(self, vfield.name, vfield)
        vfield.fieldset = self
        if isinstance(vfield, NestedField):
            for f in vfield:
                f.fieldset = self

    def check_complete(self):
        assert self.U, 'FieldSet does not have a Field named "U"'
        assert self.V, 'FieldSet does not have a Field named "V"'
        for attr, value in vars(self).items():
            if type(value) is Field:
                assert value.name == attr, 'Field %s.name (%s) is not consistent' % (value.name, attr)

        for g in self.gridset.grids:
            g.check_zonal_periodic()
            if len(g.time) == 1:
                continue
            assert isinstance(g.time_origin, type(self.time_origin)), 'time origins of different grids must be have the same type'
            g.time = g.time + self.time_origin.reltime(g.time_origin)
            if g.defer_load:
                g.time_full = g.time_full + self.time_origin.reltime(g.time_origin)
            g.time_origin = self.time_origin
        if not hasattr(self, 'UV'):
            if isinstance(self.U, SummedField):
                self.add_vector_field(SummedField('UV', self.U, self.V))
            elif isinstance(self.U, NestedField):
                self.add_vector_field(NestedField('UV', self.U, self.V))
            else:
                self.add_vector_field(VectorField('UV', self.U, self.V))
        if not hasattr(self, 'UVW') and hasattr(self, 'W'):
            if isinstance(self.U, SummedField):
                self.add_vector_field(SummedField('UVW', self.U, self.V, self.W))
            elif isinstance(self.U, NestedField):
                self.add_vector_field(NestedField('UVW', self.U, self.V, self.W))
            else:
                self.add_vector_field(VectorField('UVW', self.U, self.V, self.W))

        ccode_fieldnames = []
        counter = 1
        for fld in self.get_fields():
            if fld.name not in ccode_fieldnames:
                fld.ccode_name = fld.name
            else:
                fld.ccode_name = fld.name + str(counter)
                counter += 1
            ccode_fieldnames.append(fld.ccode_name)

    @classmethod
    def parse_wildcards(cls, paths, filenames, var):
        if not isinstance(paths, list):
            paths = sorted(glob(str(paths)))
        if len(paths) == 0:
            notfound_paths = filenames[var] if isinstance(filenames, dict) and var in filenames else filenames
            raise IOError("FieldSet files not found: %s" % str(notfound_paths))
        for fp in paths:
            if not path.exists(fp):
                raise IOError("FieldSet file not found: %s" % str(fp))
        return paths

    @classmethod
    def from_netcdf(cls, filenames, variables, dimensions, indices=None, fieldtype=None,
                    mesh='spherical', timestamps=None, allow_time_extrapolation=None, time_periodic=False, deferred_load=True, **kwargs):
        """Initialises FieldSet object from NetCDF files

        :param filenames: Dictionary mapping variables to file(s). The
               filepath may contain wildcards to indicate multiple files
               or be a list of file.
               filenames can be a list [files], a dictionary {var:[files]},
               a dictionary {dim:[files]} (if lon, lat, depth and/or data not stored in same files as data),
               or a dictionary of dictionaries {var:{dim:[files]}}.
               time values are in filenames[data]
        :param variables: Dictionary mapping variables to variable names in the netCDF file(s).
               Note that the built-in Advection kernels assume that U and V are in m/s
        :param dimensions: Dictionary mapping data dimensions (lon,
               lat, depth, time, data) to dimensions in the netCF file(s).
               Note that dimensions can also be a dictionary of dictionaries if
               dimension names are different for each variable
               (e.g. dimensions['U'], dimensions['V'], etc).
        :param indices: Optional dictionary of indices for each dimension
               to read from file(s), to allow for reading of subset of data.
               Default is to read the full extent of each dimension.
               Note that negative indices are not allowed.
        :param fieldtype: Optional dictionary mapping fields to fieldtypes to be used for UnitConverter.
               (either 'U', 'V', 'Kh_zonal', 'Kh_meridional' or None)
        :param mesh: String indicating the type of mesh coordinates and
               units used during velocity interpolation, see also https://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_unitconverters.ipynb:

               1. spherical (default): Lat and lon in degree, with a
                  correction for zonal velocity U near the poles.
               2. flat: No conversion, lat/lon are assumed to be in m.
        :param timestamps: A numpy array containing the timestamps for each of the files in filenames.
               Default is None if dimensions includes time.
        :param allow_time_extrapolation: boolean whether to allow for extrapolation
               (i.e. beyond the last available time snapshot)
               Default is False if dimensions includes time, else True
        :param time_periodic: boolean whether to loop periodically over the time component of the FieldSet
               This flag overrides the allow_time_interpolation and sets it to False
        :param deferred_load: boolean whether to only pre-load data (in deferred mode) or
               fully load them (default: True). It is advised to deferred load the data, since in
               that case Parcels deals with a better memory management during particle set execution.
               deferred_load=False is however sometimes necessary for plotting the fields.
        :param netcdf_engine: engine to use for netcdf reading in xarray. Default is 'netcdf',
               but in cases where this doesn't work, setting netcdf_engine='scipy' could help
        """
        # Ensure that times are not provided both in netcdf file and in 'timestamps'.
        if timestamps is not None and 'time' in dimensions:
            logger.warning_once("Time already provided, defaulting to dimensions['time'] over timestamps.")
            timestamps = None

        # Typecast timestamps to numpy array & correct shape.
        if timestamps is not None:
            if isinstance(timestamps, list):
                timestamps = np.array(timestamps)
            timestamps = np.reshape(timestamps, [timestamps.size, 1])

        fields = {}
        if 'creation_log' not in kwargs.keys():
            kwargs['creation_log'] = 'from_netcdf'
        for var, name in variables.items():
            # Resolve all matching paths for the current variable
            paths = filenames[var] if type(filenames) is dict and var in filenames else filenames
            if type(paths) is not dict:
                paths = cls.parse_wildcards(paths, filenames, var)
            else:
                for dim, p in paths.items():
                    paths[dim] = cls.parse_wildcards(p, filenames, var)

            # Use dimensions[var] and indices[var] if either of them is a dict of dicts
            dims = dimensions[var] if var in dimensions else dimensions
            cls.checkvaliddimensionsdict(dims)
            inds = indices[var] if (indices and var in indices) else indices
            fieldtype = fieldtype[var] if (fieldtype and var in fieldtype) else fieldtype

            grid = None
            # check if grid has already been processed (i.e. if other fields have same filenames, dimensions and indices)
            for procvar, _ in fields.items():
                procdims = dimensions[procvar] if procvar in dimensions else dimensions
                procinds = indices[procvar] if (indices and procvar in indices) else indices
                procpaths = filenames[procvar] if isinstance(filenames, dict) and procvar in filenames else filenames
                nowpaths = filenames[var] if isinstance(filenames, dict) and var in filenames else filenames
                if procdims == dims and procinds == inds and procpaths == nowpaths:
                    sameGrid = False
                    if ((not isinstance(filenames, dict)) or filenames[procvar] == filenames[var]):
                        sameGrid = True
                    elif isinstance(filenames[procvar], dict):
                        sameGrid = True
                        for dim in ['lon', 'lat', 'depth']:
                            if dim in dimensions:
                                sameGrid *= filenames[procvar][dim] == filenames[var][dim]
                    if sameGrid:
                        grid = fields[procvar].grid
                        kwargs['dataFiles'] = fields[procvar].dataFiles
                        break
            fields[var] = Field.from_netcdf(paths, (var, name), dims, inds, grid=grid, mesh=mesh, timestamps=timestamps,
                                            allow_time_extrapolation=allow_time_extrapolation,
                                            time_periodic=time_periodic, deferred_load=deferred_load,
                                            fieldtype=fieldtype, **kwargs)
        u = fields.pop('U', None)
        v = fields.pop('V', None)
        return cls(u, v, fields=fields)

    @classmethod
    def from_nemo(cls, filenames, variables, dimensions, indices=None, mesh='spherical',
                  allow_time_extrapolation=None, time_periodic=False,
                  tracer_interp_method='cgrid_tracer', **kwargs):
        """Initialises FieldSet object from NetCDF files of Curvilinear NEMO fields.

        :param filenames: Dictionary mapping variables to file(s). The
               filepath may contain wildcards to indicate multiple files,
               or be a list of file.
               filenames can be a list [files], a dictionary {var:[files]},
               a dictionary {dim:[files]} (if lon, lat, depth and/or data not stored in same files as data),
               or a dictionary of dictionaries {var:{dim:[files]}}
               time values are in filenames[data]
        :param variables: Dictionary mapping variables to variable names in the netCDF file(s).
               Note that the built-in Advection kernels assume that U and V are in m/s
        :param dimensions: Dictionary mapping data dimensions (lon,
               lat, depth, time, data) to dimensions in the netCF file(s).
               Note that dimensions can also be a dictionary of dictionaries if
               dimension names are different for each variable.
               Watch out: NEMO is discretised on a C-grid:
               U and V velocities are not located on the same nodes (see https://www.nemo-ocean.eu/doc/node19.html ).
                _________________V[k,j+1,i+1]________________
               |                                             |
               |                                             |
               U[k,j+1,i]   W[k:k+2,j+1,i+1], T[k,j+1,i+1]   U[k,j+1,i+1]
               |                                             |
               |                                             |
               |_________________V[k,j,i+1]__________________|
               To interpolate U, V velocities on the C-grid, Parcels needs to read the f-nodes,
               which are located on the corners of the cells.
               (for indexing details: https://www.nemo-ocean.eu/doc/img360.png )
               In 3D, the depth is the one corresponding to W nodes
        :param indices: Optional dictionary of indices for each dimension
               to read from file(s), to allow for reading of subset of data.
               Default is to read the full extent of each dimension.
               Note that negative indices are not allowed.
        :param fieldtype: Optional dictionary mapping fields to fieldtypes to be used for UnitConverter.
               (either 'U', 'V', 'Kh_zonal', 'Kh_meridional' or None)
        :param mesh: String indicating the type of mesh coordinates and
               units used during velocity interpolation, see also https://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_unitconverters.ipynb:

               1. spherical (default): Lat and lon in degree, with a
                  correction for zonal velocity U near the poles.
               2. flat: No conversion, lat/lon are assumed to be in m.
        :param allow_time_extrapolation: boolean whether to allow for extrapolation
               (i.e. beyond the last available time snapshot)
               Default is False if dimensions includes time, else True
        :param time_periodic: boolean whether to loop periodically over the time component of the FieldSet
               This flag overrides the allow_time_interpolation and sets it to False
        :param tracer_interp_method: Method for interpolation of tracer fields. It is recommended to use 'cgrid_tracer' (default)
               Note that in the case of from_nemo() and from_cgrid(), the velocity fields are default to 'cgrid_velocity'

        """

        if 'creation_log' not in kwargs.keys():
            kwargs['creation_log'] = 'from_nemo'
        fieldset = cls.from_c_grid_dataset(filenames, variables, dimensions, mesh=mesh, indices=indices, time_periodic=time_periodic,
                                           allow_time_extrapolation=allow_time_extrapolation, tracer_interp_method=tracer_interp_method, **kwargs)
        if hasattr(fieldset, 'W'):
            fieldset.W.set_scaling_factor(-1.)
        return fieldset

    @classmethod
    def from_c_grid_dataset(cls, filenames, variables, dimensions, indices=None, mesh='spherical',
                            allow_time_extrapolation=None, time_periodic=False,
                            tracer_interp_method='cgrid_tracer', **kwargs):
        """Initialises FieldSet object from NetCDF files of Curvilinear NEMO fields.

        :param filenames: Dictionary mapping variables to file(s). The
               filepath may contain wildcards to indicate multiple files,
               or be a list of file.
               filenames can be a list [files], a dictionary {var:[files]},
               a dictionary {dim:[files]} (if lon, lat, depth and/or data not stored in same files as data),
               or a dictionary of dictionaries {var:{dim:[files]}}
               time values are in filenames[data]
        :param variables: Dictionary mapping variables to variable
               names in the netCDF file(s).
        :param dimensions: Dictionary mapping data dimensions (lon,
               lat, depth, time, data) to dimensions in the netCF file(s).
               Note that dimensions can also be a dictionary of dictionaries if
               dimension names are different for each variable.
               Watch out: NEMO is discretised on a C-grid:
               U and V velocities are not located on the same nodes (see https://www.nemo-ocean.eu/doc/node19.html ).
                _________________V[k,j+1,i+1]________________
               |                                             |
               |                                             |
               U[k,j+1,i]   W[k:k+2,j+1,i+1], T[k,j+1,i+1]   U[k,j+1,i+1]
               |                                             |
               |                                             |
               |_________________V[k,j,i+1]__________________|
               To interpolate U, V velocities on the C-grid, Parcels needs to read the f-nodes,
               which are located on the corners of the cells.
               (for indexing details: https://www.nemo-ocean.eu/doc/img360.png )
               In 3D, the depth is the one corresponding to W nodes
        :param indices: Optional dictionary of indices for each dimension
               to read from file(s), to allow for reading of subset of data.
               Default is to read the full extent of each dimension.
               Note that negative indices are not allowed.
        :param fieldtype: Optional dictionary mapping fields to fieldtypes to be used for UnitConverter.
               (either 'U', 'V', 'Kh_zonal', 'Kh_meridional' or None)
        :param mesh: String indicating the type of mesh coordinates and
               units used during velocity interpolation:

               1. spherical (default): Lat and lon in degree, with a
                  correction for zonal velocity U near the poles.
               2. flat: No conversion, lat/lon are assumed to be in m.
        :param allow_time_extrapolation: boolean whether to allow for extrapolation
               (i.e. beyond the last available time snapshot)
               Default is False if dimensions includes time, else True
        :param time_periodic: boolean whether to loop periodically over the time component of the FieldSet
               This flag overrides the allow_time_interpolation and sets it to False
        :param tracer_interp_method: Method for interpolation of tracer fields. It is recommended to use 'cgrid_tracer' (default)
               Note that in the case of from_nemo() and from_cgrid(), the velocity fields are default to 'cgrid_velocity'

        """

        if 'U' in dimensions and 'V' in dimensions and dimensions['U'] != dimensions['V']:
            raise RuntimeError("On a c-grid discretisation like NEMO, U and V should have the same dimensions")
        if 'U' in dimensions and 'W' in dimensions and dimensions['U'] != dimensions['W']:
            raise RuntimeError("On a c-grid discretisation like NEMO, U, V and W should have the same dimensions")

        interp_method = {}
        for v in variables:
            if v in ['U', 'V', 'W']:
                interp_method[v] = 'cgrid_velocity'
            else:
                interp_method[v] = tracer_interp_method
        if 'creation_log' not in kwargs.keys():
            kwargs['creation_log'] = 'from_c_grid_dataset'

        return cls.from_netcdf(filenames, variables, dimensions, mesh=mesh, indices=indices, time_periodic=time_periodic,
                               allow_time_extrapolation=allow_time_extrapolation, interp_method=interp_method, **kwargs)

    @classmethod
    def from_pop(cls, filenames, variables, dimensions, indices=None, mesh='spherical',
                 allow_time_extrapolation=None, time_periodic=False,
                 tracer_interp_method='bgrid_tracer', **kwargs):
        """Initialises FieldSet object from NetCDF files of POP fields.
            It is assumed that the velocities in the POP fields is in cm/s.

        :param filenames: Dictionary mapping variables to file(s). The
               filepath may contain wildcards to indicate multiple files,
               or be a list of file.
               filenames can be a list [files], a dictionary {var:[files]},
               a dictionary {dim:[files]} (if lon, lat, depth and/or data not stored in same files as data),
               or a dictionary of dictionaries {var:{dim:[files]}}
               time values are in filenames[data]
        :param variables: Dictionary mapping variables to variable names in the netCDF file(s).
               Note that the built-in Advection kernels assume that U and V are in m/s
        :param dimensions: Dictionary mapping data dimensions (lon,
               lat, depth, time, data) to dimensions in the netCF file(s).
               Note that dimensions can also be a dictionary of dictionaries if
               dimension names are different for each variable.
               Watch out: POP is discretised on a B-grid:
               U and V velocity nodes are not located as W velocity and T tracer nodes (see http://www.cesm.ucar.edu/models/cesm1.0/pop2/doc/sci/POPRefManual.pdf ).
               U[k,j+1,i],V[k,j+1,i] ____________________U[k,j+1,i+1],V[k,j+1,i+1]
               |                                         |
               |      W[k:k+2,j+1,i+1],T[k,j+1,i+1]      |
               |                                         |
               U[k,j,i],V[k,j,i] ________________________U[k,j,i+1],V[k,j,i+1]
               In 2D: U and V nodes are on the cell vertices and interpolated bilinearly as a A-grid.
                      T node is at the cell centre and interpolated constant per cell as a C-grid.
               In 3D: U and V nodes are at the midlle of the cell vertical edges,
                      They are interpolated bilinearly (independently of z) in the cell.
                      W nodes are at the centre of the horizontal interfaces.
                      They are interpolated linearly (as a function of z) in the cell.
                      T node is at the cell centre, and constant per cell.
        :param indices: Optional dictionary of indices for each dimension
               to read from file(s), to allow for reading of subset of data.
               Default is to read the full extent of each dimension.
               Note that negative indices are not allowed.
        :param fieldtype: Optional dictionary mapping fields to fieldtypes to be used for UnitConverter.
               (either 'U', 'V', 'Kh_zonal', 'Kh_meridional' or None)
        :param mesh: String indicating the type of mesh coordinates and
               units used during velocity interpolation, see also https://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_unitconverters.ipynb:

               1. spherical (default): Lat and lon in degree, with a
                  correction for zonal velocity U near the poles.
               2. flat: No conversion, lat/lon are assumed to be in m.
        :param allow_time_extrapolation: boolean whether to allow for extrapolation
               (i.e. beyond the last available time snapshot)
               Default is False if dimensions includes time, else True
        :param time_periodic: boolean whether to loop periodically over the time component of the FieldSet
               This flag overrides the allow_time_interpolation and sets it to False
        :param tracer_interp_method: Method for interpolation of tracer fields. It is recommended to use 'bgrid_tracer' (default)
               Note that in the case of from_pop() and from_bgrid(), the velocity fields are default to 'bgrid_velocity'

        """

        if 'creation_log' not in kwargs.keys():
            kwargs['creation_log'] = 'from_pop'
        fieldset = cls.from_b_grid_dataset(filenames, variables, dimensions, mesh=mesh, indices=indices, time_periodic=time_periodic,
                                           allow_time_extrapolation=allow_time_extrapolation, tracer_interp_method=tracer_interp_method, **kwargs)
        if hasattr(fieldset, 'U'):
            fieldset.U.set_scaling_factor(0.01)  # cm/s to m/s
        if hasattr(fieldset, 'V'):
            fieldset.V.set_scaling_factor(0.01)  # cm/s to m/s
        if hasattr(fieldset, 'W'):
            fieldset.W.set_scaling_factor(-0.01)  # cm/s to m/s and change the W direction
        return fieldset

    @classmethod
    def from_b_grid_dataset(cls, filenames, variables, dimensions, indices=None, mesh='spherical',
                            allow_time_extrapolation=None, time_periodic=False,
                            tracer_interp_method='bgrid_tracer', **kwargs):
        """Initialises FieldSet object from NetCDF files of Bgrid fields.

        :param filenames: Dictionary mapping variables to file(s). The
               filepath may contain wildcards to indicate multiple files,
               or be a list of file.
               filenames can be a list [files], a dictionary {var:[files]},
               a dictionary {dim:[files]} (if lon, lat, depth and/or data not stored in same files as data),
               or a dictionary of dictionaries {var:{dim:[files]}}
               time values are in filenames[data]
        :param variables: Dictionary mapping variables to variable
               names in the netCDF file(s).
        :param dimensions: Dictionary mapping data dimensions (lon,
               lat, depth, time, data) to dimensions in the netCF file(s).
               Note that dimensions can also be a dictionary of dictionaries if
               dimension names are different for each variable.
               U and V velocity nodes are not located as W velocity and T tracer nodes (see http://www.cesm.ucar.edu/models/cesm1.0/pop2/doc/sci/POPRefManual.pdf ).
               U[k,j+1,i],V[k,j+1,i] ____________________U[k,j+1,i+1],V[k,j+1,i+1]
               |                                         |
               |      W[k:k+2,j+1,i+1],T[k,j+1,i+1]      |
               |                                         |
               U[k,j,i],V[k,j,i] ________________________U[k,j,i+1],V[k,j,i+1]
               In 2D: U and V nodes are on the cell vertices and interpolated bilinearly as a A-grid.
                      T node is at the cell centre and interpolated constant per cell as a C-grid.
               In 3D: U and V nodes are at the midlle of the cell vertical edges,
                      They are interpolated bilinearly (independently of z) in the cell.
                      W nodes are at the centre of the horizontal interfaces.
                      They are interpolated linearly (as a function of z) in the cell.
                      T node is at the cell centre, and constant per cell.
        :param indices: Optional dictionary of indices for each dimension
               to read from file(s), to allow for reading of subset of data.
               Default is to read the full extent of each dimension.
               Note that negative indices are not allowed.
        :param fieldtype: Optional dictionary mapping fields to fieldtypes to be used for UnitConverter.
               (either 'U', 'V', 'Kh_zonal', 'Kh_meridional' or None)
        :param mesh: String indicating the type of mesh coordinates and
               units used during velocity interpolation:

               1. spherical (default): Lat and lon in degree, with a
                  correction for zonal velocity U near the poles.
               2. flat: No conversion, lat/lon are assumed to be in m.
        :param allow_time_extrapolation: boolean whether to allow for extrapolation
               (i.e. beyond the last available time snapshot)
               Default is False if dimensions includes time, else True
        :param time_periodic: boolean whether to loop periodically over the time component of the FieldSet
               This flag overrides the allow_time_interpolation and sets it to False
        :param tracer_interp_method: Method for interpolation of tracer fields. It is recommended to use 'bgrid_tracer' (default)
               Note that in the case of from_pop() and from_bgrid(), the velocity fields are default to 'bgrid_velocity'

        """

        if 'U' in dimensions and 'V' in dimensions and dimensions['U'] != dimensions['V']:
            raise RuntimeError("On a B-grid discretisation, U and V should have the same dimensions")
        if 'U' in dimensions and 'W' in dimensions and dimensions['U'] != dimensions['W']:
            raise RuntimeError("On a B-grid discretisation, U, V and W should have the same dimensions")

        interp_method = {}
        for v in variables:
            if v in ['U', 'V']:
                interp_method[v] = 'bgrid_velocity'
            elif v in ['W']:
                interp_method[v] = 'bgrid_w_velocity'
            else:
                interp_method[v] = tracer_interp_method
        if 'creation_log' not in kwargs.keys():
            kwargs['creation_log'] = 'from_b_grid_dataset'

        return cls.from_netcdf(filenames, variables, dimensions, mesh=mesh, indices=indices, time_periodic=time_periodic,
                               allow_time_extrapolation=allow_time_extrapolation, interp_method=interp_method, **kwargs)

    @classmethod
    def from_parcels(cls, basename, uvar='vozocrtx', vvar='vomecrty', indices=None, extra_fields=None,
                     allow_time_extrapolation=None, time_periodic=False, deferred_load=True, **kwargs):
        """Initialises FieldSet data from NetCDF files using the Parcels FieldSet.write() conventions.

        :param basename: Base name of the file(s); may contain
               wildcards to indicate multiple files.
        :param indices: Optional dictionary of indices for each dimension
               to read from file(s), to allow for reading of subset of data.
               Default is to read the full extent of each dimension.
               Note that negative indices are not allowed.
        :param fieldtype: Optional dictionary mapping fields to fieldtypes to be used for UnitConverter.
               (either 'U', 'V', 'Kh_zonal', 'Kh_meridional' or None)
        :param extra_fields: Extra fields to read beyond U and V
        :param allow_time_extrapolation: boolean whether to allow for extrapolation
               (i.e. beyond the last available time snapshot)
               Default is False if dimensions includes time, else True
        :param time_periodic: boolean whether to loop periodically over the time component of the FieldSet
               This flag overrides the allow_time_interpolation and sets it to False
        :param deferred_load: boolean whether to only pre-load data (in deferred mode) or
               fully load them (default: True). It is advised to deferred load the data, since in
               that case Parcels deals with a better memory management during particle set execution.
               deferred_load=False is however sometimes necessary for plotting the fields.
        """

        if extra_fields is None:
            extra_fields = {}
        if 'creation_log' not in kwargs.keys():
            kwargs['creation_log'] = 'from_parcels'

        dimensions = {}
        default_dims = {'lon': 'nav_lon', 'lat': 'nav_lat',
                        'depth': 'depth', 'time': 'time_counter'}
        extra_fields.update({'U': uvar, 'V': vvar})
        for vars in extra_fields:
            dimensions[vars] = deepcopy(default_dims)
            dimensions[vars]['depth'] = 'depth%s' % vars.lower()
        filenames = dict([(v, str("%s%s.nc" % (basename, v)))
                          for v in extra_fields.keys()])
        return cls.from_netcdf(filenames, indices=indices, variables=extra_fields,
                               dimensions=dimensions, allow_time_extrapolation=allow_time_extrapolation,
                               time_periodic=time_periodic, deferred_load=deferred_load, **kwargs)

    @classmethod
    def from_xarray_dataset(cls, ds, variables, dimensions, indices=None, mesh='spherical', allow_time_extrapolation=None,
                            time_periodic=False, deferred_load=True, **kwargs):
        """Initialises FieldSet data from xarray Datasets.

        :param ds: xarray Dataset.
               Note that the built-in Advection kernels assume that U and V are in m/s
        :param variables: Dictionary mapping parcels variable names to data variables in the xarray Dataset.
        :param dimensions: Dictionary mapping data dimensions (lon,
               lat, depth, time, data) to dimensions in the xarray Dataset.
               Note that dimensions can also be a dictionary of dictionaries if
               dimension names are different for each variable
               (e.g. dimensions['U'], dimensions['V'], etc).
        :param indices: Optional dictionary of indices for each dimension
               to read from file(s), to allow for reading of subset of data.
               Default is to read the full extent of each dimension.
        :param fieldtype: Optional dictionary mapping fields to fieldtypes to be used for UnitConverter.
               (either 'U', 'V', 'Kh_zonal', 'Kh_meridional' or None)
        :param mesh: String indicating the type of mesh coordinates and
               units used during velocity interpolation, see also https://nbviewer.jupyter.org/github/OceanParcels/parcels/blob/master/parcels/examples/tutorial_unitconverters.ipynb:

               1. spherical (default): Lat and lon in degree, with a
                  correction for zonal velocity U near the poles.
               2. flat: No conversion, lat/lon are assumed to be in m.
        :param allow_time_extrapolation: boolean whether to allow for extrapolation
               (i.e. beyond the last available time snapshot)
               Default is False if dimensions includes time, else True
        :param time_periodic: boolean whether to loop periodically over the time component of the FieldSet
               This flag overrides the allow_time_interpolation and sets it to False
        :param deferred_load: boolean whether to only pre-load data (in deferred mode) or
               fully load them (default: True). It is advised to deferred load the data, since in
               that case Parcels deals with a better memory management during particle set execution.
               deferred_load=False is however sometimes necessary for plotting the fields.
        """

        fields = {}
        if 'creation_log' not in kwargs.keys():
            kwargs['creation_log'] = 'from_xarray_dataset'
        for var, name in variables.items():

            # Use dimensions[var] and indices[var] if either of them is a dict of dicts
            dims = dimensions[var] if var in dimensions else dimensions
            inds = indices[var] if (indices and var in indices) else indices

            fields[var] = Field.from_netcdf(None, ds[name], dimensions=dims, indices=inds, grid=None, mesh=mesh,
                                            allow_time_extrapolation=allow_time_extrapolation, var_name=var,
                                            time_periodic=time_periodic, deferred_load=deferred_load, **kwargs)
        u = fields.pop('U', None)
        v = fields.pop('V', None)
        return cls(u, v, fields=fields)

    def get_fields(self):
        """Returns a list of all the :class:`parcels.field.Field` and :class:`parcels.field.VectorField`
        objects associated with this FieldSet"""
        fields = []
        for v in self.__dict__.values():
            if type(v) in [Field, VectorField]:
                if v not in fields:
                    fields.append(v)
            elif type(v) in [NestedField, SummedField]:
                if v not in fields:
                    fields.append(v)
                for v2 in v:
                    if v2 not in fields:
                        fields.append(v2)
        return fields

    def add_constant(self, name, value):
        """Add a constant to the FieldSet. Note that all constants are
        stored as 32-bit floats. While constants can be updated during
        execution in SciPy mode, they can not be updated in JIT mode.

        :param name: Name of the constant
        :param value: Value of the constant (stored as 32-bit float)
        """
        setattr(self, name, value)

    def add_periodic_halo(self, zonal=False, meridional=False, halosize=5):
        """Add a 'halo' to all :class:`parcels.field.Field` objects in a FieldSet,
        through extending the Field (and lon/lat) by copying a small portion
        of the field on one side of the domain to the other.

        :param zonal: Create a halo in zonal direction (boolean)
        :param meridional: Create a halo in meridional direction (boolean)
        :param halosize: size of the halo (in grid points). Default is 5 grid points
        """

        for grid in self.gridset.grids:
            grid.add_periodic_halo(zonal, meridional, halosize)
        for attr, value in iter(self.__dict__.items()):
            if isinstance(value, Field):
                value.add_periodic_halo(zonal, meridional, halosize)

    def write(self, filename):
        """Write FieldSet to NetCDF file using NEMO convention

        :param filename: Basename of the output fileset"""
        logger.info("Generating NEMO FieldSet output with basename: %s" % filename)

        if hasattr(self, 'U'):
            self.U.write(filename, varname='vozocrtx')
        if hasattr(self, 'V'):
            self.V.write(filename, varname='vomecrty')

        for v in self.get_fields():
            if (v.name != 'U') and (v.name != 'V'):
                v.write(filename)

    def advancetime(self, fieldset_new):
        """Replace oldest time on FieldSet with new FieldSet
        :param fieldset_new: FieldSet snapshot with which the oldest time has to be replaced"""

        logger.warning_once("Fieldset.advancetime() is deprecated.\n \
                             Parcels deals automatically with loading only 3 time steps simustaneously\
                             such that the total allocated memory remains limited.")

        advance = 0
        for gnew in fieldset_new.gridset.grids:
            gnew.advanced = False

        for fnew in fieldset_new.get_fields():
            if isinstance(fnew, VectorField):
                continue
            f = getattr(self, fnew.name)
            gnew = fnew.grid
            if not gnew.advanced:
                g = f.grid
                advance2 = g.advancetime(gnew)
                if advance2*advance < 0:
                    raise RuntimeError("Some Fields of the Fieldset are advanced forward and other backward")
                advance = advance2
                gnew.advanced = True
            f.advancetime(fnew, advance == 1)

    def computeTimeChunk(self, time, dt):
        """Load a chunk of three data time steps into the FieldSet.
        This is used when FieldSet uses data imported from netcdf,
        with default option deferred_load. The loaded time steps are at or immediatly before time
        and the two time steps immediately following time if dt is positive (and inversely for negative dt)
        :param time: Time around which the FieldSet chunks are to be loaded. Time is provided as a double, relatively to Fieldset.time_origin
        :param dt: time step of the integration scheme
        """
        signdt = np.sign(dt)
        nextTime = np.infty if dt > 0 else -np.infty

        for g in self.gridset.grids:
            g.update_status = 'not_updated'
        for f in self.get_fields():
            if type(f) in [VectorField, NestedField, SummedField] or not f.grid.defer_load:
                continue
            if f.grid.update_status == 'not_updated':
                nextTime_loc = f.grid.computeTimeChunk(f, time, signdt)
            nextTime = min(nextTime, nextTime_loc) if signdt >= 0 else max(nextTime, nextTime_loc)

        # load in new data
        for f in self.get_fields():
            if type(f) in [VectorField, NestedField, SummedField] or not f.grid.defer_load or f.is_gradient or f.dataFiles is None:
                continue
            g = f.grid
            if g.update_status == 'first_updated':  # First load of data
                data = np.empty((g.tdim, g.zdim, g.ydim-2*g.meridional_halo, g.xdim-2*g.zonal_halo), dtype=np.float32)
                f.loaded_time_indices = range(3)
                for tind in f.loaded_time_indices:
                    f.computeTimeChunk(data, tind)
                f.data = f.reshape(data)
            elif g.update_status == 'updated':
                data = np.empty((g.tdim, g.zdim, g.ydim-2*g.meridional_halo, g.xdim-2*g.zonal_halo), dtype=np.float32)
                if signdt >= 0:
                    f.data[:2, :] = f.data[1:, :]
                    f.loaded_time_indices = [2]
                else:
                    f.data[1:, :] = f.data[:2, :]
                    f.loaded_time_indices = [0]
                f.computeTimeChunk(data, f.loaded_time_indices[0])
                f.data[f.loaded_time_indices[0], :] = f.reshape(data)[f.loaded_time_indices[0], :]
            else:
                f.loaded_time_indices = []

            # do built-in computations on data
            for tind in f.loaded_time_indices:
                if f._scaling_factor:
                    f.data[tind, :] *= f._scaling_factor
                f.data[tind, :] = np.where(np.isnan(f.data[tind, :]), 0, f.data[tind, :])
                if f.vmin is not None:
                    f.data[tind, :] = np.where(f.data[tind, :] < f.vmin, 0, f.data[tind, :])
                if f.vmax is not None:
                    f.data[tind, :] = np.where(f.data[tind, :] > f.vmax, 0, f.data[tind, :])
                if f.gradientx is not None:
                    f.gradient(update=True, tindex=tind)

        # do user-defined computations on fieldset data
        if self.compute_on_defer:
            self.compute_on_defer(self)

        if abs(nextTime) == np.infty or np.isnan(nextTime):  # Second happens when dt=0
            return nextTime
        else:
            nSteps = int((nextTime - time) / dt)
            if nSteps == 0:
                return nextTime
            else:
                return time + nSteps * dt

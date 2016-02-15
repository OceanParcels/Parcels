from netCDF4 import Dataset
from parcels.field import Field
from parcels.particle import ParticleSet
from py import path
from glob import glob


__all__ = ['NEMOGrid']


class NEMOGrid(object):
    """Grid class used to generate and read NEMO output files

    :param U: :class:`Field` for zonal velocity component
    :param V: :class:`Field` for meridional velocity component
    :param depth: Depth coordinates of the grid
    :param time: Time coordinates of the grid
    :param fields: Dictionary of additional fields
    """
    def __init__(self, U, V, depth, time, fields={}):
        self.U = U
        self.V = V
        self.depth = depth
        self.time = time
        self.fields = fields

        # Add additional fields as attributes
        for name, field in fields.items():
            setattr(self, name, field)

    @classmethod
    def from_data(cls, data_u, lon_u, lat_u, data_v, lon_v, lat_v,
                  depth, time, field_data={}, transpose=True, **kwargs):
        """Initialise Grid object from raw data

        :param data_u: Zonal velocity data
        :param lon_u: Longitude coordinates of the U components
        :param lat_u: Latitude coordinates of the U components
        :param data_v: Meridional velocity data
        :param lon_v: Longitude coordinates of the V components
        :param lat_v: Latitude coordinates of the V components
        :param depth: Depth coordinates of the grid
        :param time: Time coordinates of the grid
        """
        # Create velocity fields
        ufield = Field('U', data_u, lon_u, lat_u, depth=depth,
                       time=time, transpose=transpose, **kwargs)
        vfield = Field('V', data_v, lon_v, lat_v, depth=depth,
                       time=time, transpose=transpose, **kwargs)
        # Create additional data fields
        fields = {}
        for name, data in field_data.items():
            fields[name] = Field(name, data, lon_v, lat_u, depth=depth,
                                 time=time, transpose=transpose, **kwargs)
        return cls(ufield, vfield, depth, time, fields=fields)

    @classmethod
    def from_file(cls, filename, uvar='vozocrtx', vvar='vomecrty',
                  extra_vars={}, **kwargs):
        """Initialises grid data from files using NEMO conventions.

        :param filename: Base name of the file(s); may contain
        wildcards to indicate multiple files.
        """
        fields = {}
        extra_vars.update({'U': uvar, 'V': vvar})
        for var, vname in extra_vars.items():
            # Resolve all matching paths for the current variable
            basepath = path.local("%s%s.nc" % (filename, var))
            paths = [path.local(fp) for fp in glob(str(basepath))]
            for fp in paths:
                if not fp.exists():
                    raise IOError("Grid file not found: %s" % str(fp))
            dsets = [Dataset(str(fp), 'r', format="NETCDF4") for fp in paths]
            fields[var] = Field.from_netcdf(var, vname, dsets, **kwargs)
        u = fields.pop('U')
        v = fields.pop('V')
        return cls(u, v, u.depth, u.time, fields=fields)

    def ParticleSet(self, *args, **kwargs):
        return ParticleSet(*args, grid=self, **kwargs)

    def eval(self, x, y):
        u = self.U.eval(x, y)
        v = self.V.eval(x, y)
        return u, v

    def write(self, filename):
        """Write flow field to NetCDF file using NEMO convention

        :param filename: Basename of the output fileset"""
        print("Generating NEMO grid output with basename: %s" % filename)

        self.U.write(filename, varname='vozocrtx')
        self.V.write(filename, varname='vomecrty')

        for f in self.fields:
            field = getattr(self, f)
            field.write(filename)



class NetCDF_Grid(object):
    """Grid class used to generate and read NetCDF files
        
        :param depth: Depth coordinates of the grid
        :param time: Time coordinates of the grid
        :param fields: Dictionary of fields variables across these coordinates
        """
    def __init__(self, depth, time, fields={}):
        self.depth = depth
        self.time = time
        self.fields = fields.keys()
        
        # Add additional fields as attributes
        for name, field in fields.items():
            setattr(self, name, field)

    @classmethod
    def from_file(cls, loc, filenames={}, vars={}, dimensions={}, **kwargs):
        """Initialises grid data from files using NEMO conventions.
            
            :param filenames: Dictionary of filenames for each variable stored within
            :param vars: Dictionary of variables and the corresponding naming convention within the netcdf file
            :param dimensions: Dictionary of dimensions used and the naming convention within the netcdf file
            """
        fields = {}
        for var, vname in vars.items(): #Cycle through files and variables loading netcdf data
            # Resolve all matching paths for the current variable
            filename = filenames[var]
            fp = path.local(loc+"/"+filename)
            if not fp.exists():
                raise IOError("Grid file not found: %s" % str(fp))
            dset = Dataset(str(fp), 'r', format="NETCDF4")
            print("Loading %s data from %s" % (var, str(fp)))
            fields[var] = Field.from_netcdf(var, vname, dset, dimensions, **kwargs)
        
        #Assumes depth and time dimensions are the same size across separate NetCDF loaded fields
        return cls(fields[fields.keys()[0]].depth, fields[fields.keys()[0]].time, fields=fields)
    
    def ParticleSet(self, *args, **kwargs):
        return ParticleSet(*args, grid=self, **kwargs)
    
    def eval(self, x, y):
        u = self.U.eval(x, y)
        v = self.V.eval(x, y)
        return u, v
    
    def write(self, filename):
        """Write only the flow field components of the grid to NetCDF file using a simple convention
            
            :param filename: Basename of the output fileset"""
        print("Generating NetCDF grid output with basename: %s" % filename)
        
        self.U.write(filename, varname='U')
        self.V.write(filename, varname='V')
        
        for f in self.fields:
            field = getattr(self, f)
            field.write(filename)



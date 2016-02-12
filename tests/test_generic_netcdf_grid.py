from netCDF4 import Dataset
from parcels import NEMOGrid, Particle, JITParticle, AdvectionRK4
from parcels.field import Field
from parcels.particle import ParticleSet
from argparse import ArgumentParser
from py import path
from glob import glob
import numpy as np
import math
import pytest

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
        
        :param filename: Base name of the file(s); may contain
        wildcards to indicate multiple files.
        """
        fields = {}
        for var, vname in vars.items():
            # Resolve all matching paths for the current variable
            filename = filenames[var]
            fp = path.local(loc+"/"+filename)
            #paths = [path.local(fp) for fp in glob(str(filename))]
            if not fp.exists():
                raise IOError("Grid file not found: %s" % str(fp))
            dset = Dataset(str(fp), 'r', format="NETCDF4")
            print("Loading %s data from %s" % (var, str(fp)))
            fields[var] = Field.from_netcdf(var, vname, dset, dimensions, **kwargs)
                #u = fields.pop('U')
        print(fields.keys())
        #v = fields.pop('V')
        return cls(fields[fields.keys()[0]].depth, fields[fields.keys()[0]].time, fields=fields)
    
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



if __name__ == "__main__":
    p = ArgumentParser(description="""
        Example of particle advection around an idealised peninsula""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing RK4 computation')
    p.add_argument('-p', '--particles', type=int, default=2,
                                  help='Number of particles to advect')
    p.add_argument('-v', '--verbose', action='store_true', default=False,
                                  help='Print particle information before and after execution')
    p.add_argument('--profiling', action='store_true', default=False,
                                  help='Print profiling information after run')
    p.add_argument('-g', '--grid', type=int, nargs=2, default=None,
                                  help='Generate grid file with given dimensions')

    # Open grid files
    filenames = {"U":"ocean_u_1993_01.TropPac.nc",
                 "V" : "ocean_v_1993_01.TropPac.nc"}
    dimensions = {'lat':'yu_ocean',
                  'long':'xu_ocean',
                  'time':'Time'}
    datadir = "/Volumes/4TB SAMSUNG/Ocean_Model_Data/OFAM_month1"
    grid = NetCDF_Grid.from_file(loc=datadir, filenames=filenames, vars={'U': 'u', 'V': 'v'}, dimensions=dimensions)

    ParticleClass = JITParticle
    npart = 10
    
    pset = grid.ParticleSet(size=npart, pclass=ParticleClass,
                            start=(180, -5), finish=(180, 5))

    hours = 25*24
    substeps = 6

    pset.execute(AdvectionRK4, timesteps=hours*substeps, dt=300.,
                       output_file=pset.ParticleFile(name="GenericGridParticle"),
                       output_steps=substeps)
          

    

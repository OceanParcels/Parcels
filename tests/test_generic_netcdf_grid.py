from parcels import Particle, JITParticle, AdvectionRK4
from parcels.grid import NetCDF_Grid
from argparse import ArgumentParser
from py import path

if __name__ == "__main__":
    p = ArgumentParser(description="""
        Example of particle advection around an idealised peninsula""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='scipy',
                   help='Execution mode for performing RK4 computation')
    p.add_argument('-p', '--particles', type=int, default=10,
                   help='Number of particles to advect')
    p.add_argument('--profiling', action='store_true', default=False,
                   help='Print profiling information after run')
    p.add_argument('-l', '--location', default=None,
                   help='Location folder path of input NetCDF files')
    p.add_argument('-f', '--files', default=['OFAM_Simple_U.nc', 'OFAM_Simple_V.nc'],
                   help='List of NetCDF files to load')
    p.add_argument('-v', '--variables', default=['U', 'V'],
                   help='List of field variables to extract, using PARCELS naming convention')
    p.add_argument('-n', '--netcdf_vars', default=['u', 'v'],
                   help='List of field variable names, as given in the NetCDF file. Order must match --variables args')
    p.add_argument('-d', '--dimensions', default=['lat', 'long', 'time'],
                   help='List of PARCELS convention named dimensions across which field variables occur')
    p.add_argument('-m', '--map_dimensions', default=['yu_ocean', 'xu_ocean', 'Time'],
                   help='List of dimensions across which field variables occur, as given in the NetCDF files, to map to the --dimensions args')
    p.add_argument('-o', '--output_file', default='NetCDF_Particles',
                   help='Name of output NetCDF file for particle data')
    args = p.parse_args()

    ParticleClass = JITParticle if args.mode == 'jit' else Particle

    if args.location is None:
        datadir = path.local().dirpath()
        datadir = str(datadir) + '/' + 'parcels-examples/OFAM_example_data'
    else:
        datadir = args.location

    # Build required dictionaries for NETCDF_Grid from cmd arguments
    filenames = {}
    for vars, var_files in zip(args.variables, args.files):
        filenames.update({vars: var_files})
    variables = {}
    for vars, net_vars in zip(args.variables, args.netcdf_vars):
        variables.update({vars: net_vars})
    dimensions = {}
    for dims, net_dims in zip(args.dimensions, args.map_dimensions):
        dimensions.update({dims: net_dims})

    grid = NetCDF_Grid.from_file(loc=datadir, filenames=filenames, vars=variables, dimensions=dimensions)

    lat_min = min(getattr(grid, grid.fields[0]).lat)
    lat_max = max(getattr(grid, grid.fields[0]).lat)
    lon_min = min(getattr(grid, grid.fields[0]).lon)
    lon_max = max(getattr(grid, grid.fields[0]).lon)

    pset = grid.ParticleSet(size=args.particles, pclass=ParticleClass,
                            start=((lon_min + lon_max) / 2, lat_min), finish=((lon_min + lon_max) / 2, lat_max))

    hours = 25 * 24
    substeps = 60

    pset.execute(AdvectionRK4, timesteps=hours * substeps, dt=300.,
                 output_file=pset.ParticleFile(name=args.output_file),
                 output_steps=substeps)

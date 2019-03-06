from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle, Variable
from parcels import AdvectionRK4, AdvectionEE, AdvectionRK45
from argparse import ArgumentParser
import numpy as np
import math  # NOQA
import pytest
from datetime import timedelta as delta


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
method = {'RK4': AdvectionRK4, 'EE': AdvectionEE, 'RK45': AdvectionRK45}


def peninsula_fieldset(xdim, ydim, mesh='flat'):
    """Construct a fieldset encapsulating the flow field around an
    idealised peninsula.

    :param xdim: Horizontal dimension of the generated fieldset
    :param xdim: Vertical dimension of the generated fieldset
    :param mesh: String indicating the type of mesh coordinates and
               units used during velocity interpolation:

               1. spherical: Lat and lon in degree, with a
                  correction for zonal velocity U near the poles.
               2. flat  (default): No conversion, lat/lon are assumed to be in m.

    The original test description can be found in Fig. 2.2.3 in:
    North, E. W., Gallego, A., Petitgas, P. (Eds). 2009. Manual of
    recommended practices for modelling physical - biological
    interactions during fish early life.
    ICES Cooperative Research Report No. 295. 111 pp.
    http://archimer.ifremer.fr/doc/00157/26792/24888.pdf

    To avoid accuracy problems with interpolation from A-grid
    to C-grid, we return NetCDF files that are on an A-grid.
    """
    # Set Parcels FieldSet variables

    # Generate the original test setup on A-grid in m
    domainsizeX, domainsizeY = (1.e5, 5.e4)
    dx, dy = domainsizeX / xdim, domainsizeY / ydim
    La = np.linspace(dx, 1.e5-dx, xdim, dtype=np.float32)
    Wa = np.linspace(dy, 5.e4-dy, ydim, dtype=np.float32)

    u0 = 1
    x0 = domainsizeX / 2
    R = 0.32 * domainsizeX / 2

    # Create the fields
    x, y = np.meshgrid(La, Wa, sparse=True, indexing='xy')
    P = (u0*R**2*y/((x-x0)**2+y**2)-u0*y) / 1e3
    U = u0-u0*R**2*((x-x0)**2-y**2)/(((x-x0)**2+y**2)**2)
    V = -2*u0*R**2*((x-x0)*y)/(((x-x0)**2+y**2)**2)

    # Set land points to NaN
    landpoints = P >= 0.
    P[landpoints] = np.nan
    U[landpoints] = np.nan
    V[landpoints] = np.nan

    # Convert from m to lat/lon for spherical meshes
    lon = La / 1852. / 60. if mesh == 'spherical' else La
    lat = Wa / 1852. / 60. if mesh == 'spherical' else Wa

    data = {'U': U, 'V': V, 'P': P}
    dimensions = {'lon': lon, 'lat': lat}
    return FieldSet.from_data(data, dimensions, mesh=mesh)


def UpdateP(particle, fieldset, time):
    particle.p = fieldset.P[time, particle.depth, particle.lat, particle.lon]


def pensinsula_example(fieldset, npart, mode='jit', degree=1,
                       verbose=False, output=True, method=AdvectionRK4):
    """Example configuration of particle flow around an idealised Peninsula

    :arg filename: Basename of the input fieldset
    :arg npart: Number of particles to intialise"""

    # First, we define a custom Particle class to which we add a
    # custom variable, the initial stream function value p.
    # We determine the particle base class according to mode.
    class MyParticle(ptype[mode]):
        # JIT compilation requires a-priori knowledge of the particle
        # data structure, so we define additional variables here.
        p = Variable('p', dtype=np.float32, initial=0.)
        p_start = Variable('p_start', dtype=np.float32, initial=fieldset.P)

    # Initialise particles
    if fieldset.U.grid.mesh == 'flat':
        x = 3000  # 3 km offset from boundary
    else:
        x = 3. * (1. / 1.852 / 60)  # 3 km offset from boundary
    y = (fieldset.U.lat[0] + x, fieldset.U.lat[-1] - x)  # latitude range, including offsets
    pset = ParticleSet.from_line(fieldset, size=npart, pclass=MyParticle,
                                 start=(x, y[0]), finish=(x, y[1]), time=0)

    if verbose:
        print("Initial particle positions:\n%s" % pset)

    # Advect the particles for 24h
    time = delta(hours=24)
    dt = delta(minutes=5)
    k_adv = pset.Kernel(method)
    k_p = pset.Kernel(UpdateP)
    out = pset.ParticleFile(name="MyParticle", outputdt=delta(hours=1)) if output else None
    print("Peninsula: Advecting %d particles for %s" % (npart, str(time)))
    pset.execute(k_adv + k_p, runtime=time, dt=dt, output_file=out)

    if verbose:
        print("Final particle positions:\n%s" % pset)

    return pset


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('mesh', ['flat', 'spherical'])
def test_peninsula_fieldset(mode, mesh):
    """Execute peninsula test from fieldset generated in memory"""
    fieldset = peninsula_fieldset(100, 50, mesh)
    pset = pensinsula_example(fieldset, 5, mode=mode, degree=1)
    # Test advection accuracy by comparing streamline values
    err_adv = np.array([abs(p.p_start - p.p) for p in pset])
    assert(err_adv <= 1.e-3).all()
    # Test Field sampling accuracy by comparing kernel against Field sampling
    err_smpl = np.array([abs(p.p - pset.fieldset.P[0., p.depth, p.lat, p.lon]) for p in pset])
    assert(err_smpl <= 1.e-3).all()


def fieldsetfile(mesh):
    """Generate fieldset files for peninsula test"""
    filename = 'peninsula'
    fieldset = peninsula_fieldset(100, 50, mesh=mesh)
    fieldset.write(filename)
    return filename


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('mesh', ['flat', 'spherical'])
def test_peninsula_file(mode, mesh):
    """Open fieldset files and execute"""
    fieldset = FieldSet.from_parcels(fieldsetfile(mesh), extra_fields={'P': 'P'}, allow_time_extrapolation=True)
    pset = pensinsula_example(fieldset, 5, mode=mode, degree=1)
    # Test advection accuracy by comparing streamline values
    err_adv = np.array([abs(p.p_start - p.p) for p in pset])
    assert(err_adv <= 1.e-3).all()
    # Test Field sampling accuracy by comparing kernel against Field sampling
    err_smpl = np.array([abs(p.p - pset.fieldset.P[0., p.depth, p.lat, p.lon]) for p in pset])
    assert(err_smpl <= 1.e-3).all()


if __name__ == "__main__":
    p = ArgumentParser(description="""
Example of particle advection around an idealised peninsula""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing RK4 computation')
    p.add_argument('-p', '--particles', type=int, default=20,
                   help='Number of particles to advect')
    p.add_argument('-d', '--degree', type=int, default=1,
                   help='Degree of spatial interpolation')
    p.add_argument('-v', '--verbose', action='store_true', default=False,
                   help='Print particle information before and after execution')
    p.add_argument('-o', '--nooutput', action='store_true', default=False,
                   help='Suppress trajectory output')
    p.add_argument('--profiling', action='store_true', default=False,
                   help='Print profiling information after run')
    p.add_argument('-f', '--fieldset', type=int, nargs=2, default=None,
                   help='Generate fieldset file with given dimensions')
    p.add_argument('-m', '--method', choices=('RK4', 'EE', 'RK45'), default='RK4',
                   help='Numerical method used for advection')
    args = p.parse_args()

    if args.fieldset is not None:
        filename = 'peninsula'
        fieldset = peninsula_fieldset(args.fieldset[0], args.fieldset[1], mesh='flat')
        fieldset.write(filename)

    # Open fieldset file set
    fieldset = FieldSet.from_parcels('peninsula', extra_fields={'P': 'P'}, allow_time_extrapolation=True)

    if args.profiling:
        from cProfile import runctx
        from pstats import Stats
        runctx("pensinsula_example(fieldset, args.particles, mode=args.mode,\
                                   degree=args.degree, verbose=args.verbose,\
                                   output=not args.nooutput, method=method[args.method])",
               globals(), locals(), "Profile.prof")
        Stats("Profile.prof").strip_dirs().sort_stats("time").print_stats(10)
    else:
        pensinsula_example(fieldset, args.particles, mode=args.mode,
                           degree=args.degree, verbose=args.verbose,
                           output=not args.nooutput, method=method[args.method])

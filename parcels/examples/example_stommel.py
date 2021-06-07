import math
from argparse import ArgumentParser
from datetime import timedelta as delta

import numpy as np
import pytest

from parcels import AdvectionEE
from parcels import AdvectionRK4
from parcels import AdvectionRK45
from parcels import FieldSet
from parcels import JITParticle
from parcels import ScipyParticle
from parcels import ParticleSetSOA, ParticleFileSOA, KernelSOA  # noqa
from parcels import ParticleSetAOS, ParticleFileAOS, KernelAOS  # noqa
from parcels import timer
from parcels import Variable

pset_modes = ['soa', 'aos']
ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
method = {'RK4': AdvectionRK4, 'EE': AdvectionEE, 'RK45': AdvectionRK45}
pset_type = {'soa': {'pset': ParticleSetSOA, 'pfile': ParticleFileSOA, 'kernel': KernelSOA},
             'aos': {'pset': ParticleSetAOS, 'pfile': ParticleFileAOS, 'kernel': KernelAOS}}


def stommel_fieldset(xdim=200, ydim=200, grid_type='A'):
    """Simulate a periodic current along a western boundary, with significantly
    larger velocities along the western edge than the rest of the region

    The original test description can be found in: N. Fabbroni, 2009,
    Numerical Simulation of Passive tracers dispersion in the sea,
    Ph.D. dissertation, University of Bologna
    http://amsdottorato.unibo.it/1733/1/Fabbroni_Nicoletta_Tesi.pdf
    """
    a = b = 10000 * 1e3
    scalefac = 0.05  # to scale for physically meaningful velocities
    dx, dy = a / xdim, b / ydim

    # Coordinates of the test fieldset (on A-grid in deg)
    lon = np.linspace(0, a, xdim, dtype=np.float32)
    lat = np.linspace(0, b, ydim, dtype=np.float32)

    # Define arrays U (zonal), V (meridional) and P (sea surface height)
    U = np.zeros((lat.size, lon.size), dtype=np.float32)
    V = np.zeros((lat.size, lon.size), dtype=np.float32)
    P = np.zeros((lat.size, lon.size), dtype=np.float32)

    beta = 2e-11
    r = 1/(11.6*86400)
    es = r/(beta*a)

    for j in range(lat.size):
        for i in range(lon.size):
            xi = lon[i] / a
            yi = lat[j] / b
            P[j, i] = (1 - math.exp(-xi / es) - xi) * math.pi * np.sin(math.pi * yi) * scalefac
            if grid_type == 'A':
                U[j, i] = -(1 - math.exp(-xi / es) - xi) * math.pi ** 2 * np.cos(math.pi * yi) * scalefac
                V[j, i] = (math.exp(-xi / es) / es - 1) * math.pi * np.sin(math.pi * yi) * scalefac
    if grid_type == 'C':
        V[:, 1:] = (P[:, 1:] - P[:, 0:-1]) / dx * a
        U[1:, :] = -(P[1:, :] - P[0:-1, :]) / dy * b

    data = {'U': U, 'V': V, 'P': P}
    dimensions = {'lon': lon, 'lat': lat}
    fieldset = FieldSet.from_data(data, dimensions, mesh='flat')
    if grid_type == 'C':
        fieldset.U.interp_method = 'cgrid_velocity'
        fieldset.V.interp_method = 'cgrid_velocity'
    return fieldset


def UpdateP(particle, fieldset, time):
    particle.p = fieldset.P[time, particle.depth, particle.lat, particle.lon]


def AgeP(particle, fieldset, time):
    particle.age += particle.dt
    if particle.age > fieldset.maxage:
        particle.delete()


def stommel_example(npart=1, mode='jit', verbose=False, method=AdvectionRK4, grid_type='A',
                    outfile="StommelParticle.nc", repeatdt=None, maxage=None, write_fields=True, pset_mode='soa'):
    timer.fieldset = timer.Timer('FieldSet', parent=timer.stommel)
    fieldset = stommel_fieldset(grid_type=grid_type)
    if write_fields:
        filename = 'stommel'
        fieldset.write(filename)
    timer.fieldset.stop()

    # Determine particle class according to mode
    timer.pset = timer.Timer('Pset', parent=timer.stommel)
    timer.psetinit = timer.Timer('Pset_init', parent=timer.pset)
    ParticleClass = JITParticle if mode == 'jit' else ScipyParticle

    class MyParticle(ParticleClass):
        p = Variable('p', dtype=np.float32, initial=0.)
        p_start = Variable('p_start', dtype=np.float32, initial=fieldset.P)
        age = Variable('age', dtype=np.float32, initial=0.)

    pset = pset_type[pset_mode]['pset'].from_line(fieldset, size=npart, pclass=MyParticle, repeatdt=repeatdt,
                                                  start=(10e3, 5000e3), finish=(100e3, 5000e3), time=0)

    if verbose:
        print("Initial particle positions:\n%s" % pset)

    # Execute for 30 days, with 1hour timesteps and 12-hourly output
    runtime = delta(days=600)
    dt = delta(hours=1)
    outputdt = delta(days=5)
    maxage = runtime.total_seconds() if maxage is None else maxage
    fieldset.add_constant('maxage', maxage)
    print("Stommel: Advecting %d particles for %s" % (npart, runtime))
    timer.psetinit.stop()
    timer.psetrun = timer.Timer('Pset_run', parent=timer.pset)
    pset.execute(method + pset.Kernel(UpdateP) + pset.Kernel(AgeP), runtime=runtime, dt=dt,
                 moviedt=None, output_file=pset.ParticleFile(name=outfile, outputdt=outputdt))

    if verbose:
        print("Final particle positions:\n%s" % pset)
    timer.psetrun.stop()
    timer.pset.stop()

    return pset


@pytest.mark.parametrize('pset_mode', pset_modes)
@pytest.mark.parametrize('grid_type', ['A', 'C'])
@pytest.mark.parametrize('mode', ['jit', 'scipy'])
def test_stommel_fieldset(pset_mode, mode, grid_type, tmpdir):
    timer.root = timer.Timer('Main')
    timer.stommel = timer.Timer('Stommel', parent=timer.root)
    outfile = tmpdir.join("StommelParticle")
    psetRK4 = stommel_example(1, mode=mode, method=method['RK4'], grid_type=grid_type, outfile=outfile, write_fields=False, pset_mode=pset_mode)
    psetRK45 = stommel_example(1, mode=mode, method=method['RK45'], grid_type=grid_type, outfile=outfile, write_fields=False, pset_mode=pset_mode)
    assert np.allclose(psetRK4.lon, psetRK45.lon, rtol=1e-3)
    assert np.allclose(psetRK4.lat, psetRK45.lat, rtol=1.1e-3)
    err_adv = np.abs(psetRK4.p_start - psetRK4.p)
    assert(err_adv <= 1.e-1).all()
    err_smpl = np.array([abs(psetRK4.p[i] - psetRK4.fieldset.P[0., psetRK4.lon[i], psetRK4.lat[i], psetRK4.depth[i]]) for i in range(psetRK4.size)])
    assert(err_smpl <= 1.e-1).all()
    timer.stommel.stop()
    timer.root.stop()
    timer.root.print_tree()


if __name__ == "__main__":
    timer.root = timer.Timer('Main')
    timer.args = timer.Timer('Args', parent=timer.root)
    p = ArgumentParser(description="""
Example of particle advection in the steady-state solution of the Stommel equation""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing computation')
    p.add_argument('-p', '--particles', type=int, default=1,
                   help='Number of particles to advect')
    p.add_argument('-v', '--verbose', action='store_true', default=False,
                   help='Print particle information before and after execution')
    p.add_argument('-m', '--method', choices=('RK4', 'EE', 'RK45'), default='RK4',
                   help='Numerical method used for advection')
    p.add_argument('-o', '--outfile', default='StommelParticle.nc',
                   help='Name of output file')
    p.add_argument('-r', '--repeatdt', default=None, type=int,
                   help='repeatdt of the ParticleSet')
    p.add_argument('-a', '--maxage', default=None, type=int,
                   help='max age of the particles (after which particles are deleted)')
    p.add_argument('-psm', '--pset_mode', choices=('soa', 'aos'), default='soa',
                   help='max age of the particles (after which particles are deleted)')
    args = p.parse_args()

    timer.args.stop()
    timer.stommel = timer.Timer('Stommel', parent=timer.root)
    stommel_example(args.particles, mode=args.mode, verbose=args.verbose, method=method[args.method],
                    outfile=args.outfile, repeatdt=args.repeatdt, maxage=args.maxage)
    timer.stommel.stop()
    timer.root.stop()
    timer.root.print_tree()

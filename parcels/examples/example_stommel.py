from parcels import FieldSet, ParticleSet, ScipyParticle, JITParticle, Variable
from parcels import AdvectionRK4, AdvectionEE, AdvectionRK45
from parcels import timer
from argparse import ArgumentParser
import numpy as np
import math
import pytest
from datetime import timedelta as delta


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
method = {'RK4': AdvectionRK4, 'EE': AdvectionEE, 'RK45': AdvectionRK45}


def stommel_fieldset(xdim=200, ydim=200):
    """Simulate a periodic current along a western boundary, with significantly
    larger velocities along the western edge than the rest of the region

    The original test description can be found in: N. Fabbroni, 2009,
    Numerical Simulation of Passive tracers dispersion in the sea,
    Ph.D. dissertation, University of Bologna
    http://amsdottorato.unibo.it/1733/1/Fabbroni_Nicoletta_Tesi.pdf
    """
    # Set NEMO fieldset variables
    depth = np.zeros(1, dtype=np.float32)
    time = np.linspace(0., 100000. * 86400., 2, dtype=np.float64)

    # Some constants
    A = 100
    eps = 0.05
    a = 10000
    b = 10000

    # Coordinates of the test fieldset (on A-grid in deg)
    lon = np.linspace(0, a, xdim, dtype=np.float32)
    lat = np.linspace(0, b, ydim, dtype=np.float32)

    # Define arrays U (zonal), V (meridional), W (vertical) and P (sea
    # surface height) all on A-grid
    U = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    V = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    P = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)

    [x, y] = np.mgrid[:lon.size, :lat.size]
    l1 = (-1 + math.sqrt(1 + 4 * math.pi**2 * eps**2)) / (2 * eps)
    l2 = (-1 - math.sqrt(1 + 4 * math.pi**2 * eps**2)) / (2 * eps)
    c1 = (1 - math.exp(l2)) / (math.exp(l2) - math.exp(l1))
    c2 = -(1 + c1)
    for t in range(time.size):
        for i in range(lon.size):
            for j in range(lat.size):
                xi = lon[i] / a
                yi = lat[j] / b
                P[i, j, t] = A * (c1*math.exp(l1*xi) + c2*math.exp(l2*xi) + 1) * math.sin(math.pi * yi)
        for i in range(lon.size-2):
            for j in range(lat.size):
                V[i+1, j, t] = (P[i+2, j, t] - P[i, j, t]) / (2 * a / (xdim-1))
        for i in range(lon.size):
            for j in range(lat.size-2):
                U[i, j+1, t] = -(P[i, j+2, t] - P[i, j, t]) / (2 * b / (ydim-1))

    data = {'U': U, 'V': V, 'P': P}
    dimensions = {'lon': lon, 'lat': lat, 'depth': depth, 'time': time}
    return FieldSet.from_data(data, dimensions, mesh='flat', transpose=True)


def UpdateP(particle, fieldset, time, dt):
    particle.p = fieldset.P[time, particle.lon, particle.lat, particle.depth]


def stommel_example(npart=1, mode='jit', verbose=False, method=AdvectionRK4):
    timer.fieldset = timer.Timer('FieldSet', parent=timer.stommel)
    fieldset = stommel_fieldset()
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

    pset = ParticleSet.from_line(fieldset, size=npart, pclass=MyParticle,
                                 start=(100, 5000), finish=(200, 5000), time=0)

    if verbose:
        print("Initial particle positions:\n%s" % pset)

    # Execute for 30 days, with 1hour timesteps and 12-hourly output
    runtime = delta(days=30)
    dt = delta(hours=1)
    outputdt = delta(hours=12)
    print("Stommel: Advecting %d particles for %s" % (npart, runtime))
    timer.psetinit.stop()
    timer.psetrun = timer.Timer('Pset_run', parent=timer.pset)
    pset.execute(method + pset.Kernel(UpdateP), runtime=runtime, dt=dt,
                 moviedt=None, output_file=pset.ParticleFile(name="StommelParticle", outputdt=outputdt))

    if verbose:
        print("Final particle positions:\n%s" % pset)
    timer.psetrun.stop()
    timer.pset.stop()

    return pset


@pytest.mark.parametrize('mode', ['jit', 'scipy'])
def test_stommel_fieldset(mode):
    timer.root = timer.Timer('Main')
    timer.stommel = timer.Timer('Stommel', parent=timer.root)
    psetRK4 = stommel_example(1, mode=mode, method=method['RK4'])
    psetRK45 = stommel_example(1, mode=mode, method=method['RK45'])
    assert np.allclose([p.lon for p in psetRK4], [p.lon for p in psetRK45], rtol=1e-3)
    assert np.allclose([p.lat for p in psetRK4], [p.lat for p in psetRK45], rtol=1e-3)
    err_adv = np.array([abs(p.p_start - p.p) for p in psetRK4])
    assert(err_adv <= 1.e-1).all()
    err_smpl = np.array([abs(p.p - psetRK4.fieldset.P[0., p.lon, p.lat, p.depth]) for p in psetRK4])
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
    args = p.parse_args()

    timer.args.stop()
    timer.stommel = timer.Timer('Stommel', parent=timer.root)
    stommel_example(args.particles, mode=args.mode,
                    verbose=args.verbose, method=method[args.method])
    timer.stommel.stop()
    timer.root.stop()
    timer.root.print_tree()

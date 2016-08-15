from parcels import Grid, ScipyParticle, JITParticle, Variable
from parcels import AdvectionRK4, AdvectionEE, AdvectionRK45
from argparse import ArgumentParser
import numpy as np
import math
import pytest
from datetime import timedelta as delta


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}
method = {'RK4': AdvectionRK4, 'EE': AdvectionEE, 'RK45': AdvectionRK45}


def stommel_grid(xdim=200, ydim=200):
    """Simulate a periodic current along a western boundary, with significantly
    larger velocities along the western edge than the rest of the region

    The original test description can be found in: N. Fabbroni, 2009,
    Numerical Simulation of Passive tracers dispersion in the sea,
    Ph.D. dissertation, University of Bologna
    http://amsdottorato.unibo.it/1733/1/Fabbroni_Nicoletta_Tesi.pdf
    """
    # Set NEMO grid variables
    depth = np.zeros(1, dtype=np.float32)
    time = np.linspace(0., 100000. * 86400., 2, dtype=np.float64)

    # Some constants
    A = 100
    eps = 0.05
    a = 10000
    b = 10000

    # Coordinates of the test grid (on A-grid in deg)
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
                V[i+1, j, t] = (P[i+2, j, t] - P[i, j, t]) / (2 * a / xdim)
        for i in range(lon.size):
            for j in range(lat.size-2):
                U[i, j+1, t] = -(P[i, j+2, t] - P[i, j, t]) / (2 * b / ydim)

    return Grid.from_data(U, lon, lat, V, lon, lat, depth, time, field_data={'P': P}, mesh='flat')


def UpdateP(particle, grid, time, dt):
    particle.p = grid.P[time, particle.lon, particle.lat]


def stommel_example(npart=1, mode='jit', verbose=False, method=AdvectionRK4):

    grid = stommel_grid()
    filename = 'stommel'
    grid.write(filename)

    # Determine particle class according to mode
    ParticleClass = JITParticle if mode == 'jit' else ScipyParticle

    class MyParticle(ParticleClass):
        p = Variable('p', dtype=np.float32, default=0.)
        p_start = Variable('p_start', dtype=np.float32, default=0.)

    pset = grid.ParticleSet(size=npart, pclass=MyParticle,
                            start=(100, 5000), finish=(200, 5000))
    for particle in pset:
        particle.p_start = grid.P[0., particle.lon, particle.lat]

    if verbose:
        print("Initial particle positions:\n%s" % pset)

    # Execute for 50 days, with 5min timesteps and hourly output
    endtime = delta(days=50)
    dt = delta(minutes=5)
    interval = delta(hours=12)
    print("Stommel: Advecting %d particles for %s" % (npart, endtime))
    pset.execute(method + pset.Kernel(UpdateP), endtime=endtime, dt=dt, interval=interval,
                 output_file=pset.ParticleFile(name="StommelParticle"), show_movie=False)

    if verbose:
        print("Final particle positions:\n%s" % pset)

    return pset


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
@pytest.mark.parametrize('meth', ['RK4', pytest.mark.xfail(reason="Issue #88")('RK45')])
def test_stommel_grid(mode, meth):
    pset = stommel_example(1, mode=mode, method=method[meth])
    err_adv = np.array([abs(p.p_start - p.p) for p in pset])
    assert(err_adv <= 1.e-1).all()
    # Test grid sampling accuracy by comparing kernel against grid sampling
    err_smpl = np.array([abs(p.p - pset.grid.P[0., p.lon, p.lat]) for p in pset])
    assert(err_smpl <= 1.e-1).all()


if __name__ == "__main__":
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

    stommel_example(args.particles, mode=args.mode,
                    verbose=args.verbose, method=method[args.method])

from parcels import Grid, Particle, JITParticle, AdvectionRK4
from argparse import ArgumentParser
import numpy as np
import pytest


def generate_vertvel_grid(zdim, wvel, xdim=20, ydim=20, tdim=1):
    depth = np.linspace(0, 500, zdim, dtype=np.float32)
    lon = np.linspace(0., 1., xdim, dtype=np.float32)
    lat = np.linspace(0., 1., ydim, dtype=np.float32)
    time = np.zeros(tdim, dtype=np.float64)
    U = np.zeros((xdim, ydim, zdim), dtype=np.float32)
    V = np.zeros((xdim, ydim, zdim), dtype=np.float32)
    W = np.zeros((xdim, ydim, zdim), dtype=np.float32)
    for z in range(zdim):
        U[:, :, z] = 0.
        V[:, :, z] = 0.
        W[:, :, z] = wvel
    return (U, V, W, lon, lat, depth, time)


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_vertvel(mode):
    wvel = -1e-4
    U, V, W, lon, lat, depth, time = generate_vertvel_grid(zdim=100, wvel=wvel)
    grid = Grid.from_data(U, lon, lat, V, lon, lat, depth, time, field_data={'W': W})

    ParticleClass = JITParticle if mode == 'jit' else Particle
    pset = grid.ParticleSet(1, pclass=ParticleClass, start=(0, 0, 0.), finish=(0, 0, 0.))

    time = 24 * 3600.
    dt = 5*60.
    k_adv = pset.Kernel(AdvectionRK4)
    pset.execute(k_adv, timesteps=int(time / dt), dt=dt)
    err_adv = np.array([abs(-p.dep - wvel*time) for p in pset])
    assert(err_adv <= 1.e-3).all()


if __name__ == "__main__":
    p = ArgumentParser(description="""Example of vertical particle advection""")
    p.add_argument('mode', choices=('scipy', 'jit'), nargs='?', default='jit',
                   help='Execution mode for performing RK4 computation')
    args = p.parse_args()

    test_vertvel(mode=args.mode)

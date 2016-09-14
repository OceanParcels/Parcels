from parcels import Grid, ScipyParticle, JITParticle
from parcels import AdvectionRK4
import numpy as np
from datetime import timedelta as delta
import pytest

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}

# Define some constants.
u_g = .04  # Geostrophic current
u_0 = .3  # Initial speed in x dirrection. v_0 = 0
gamma = 1./delta(days=2.89).total_seconds()  # Dissipitave effects due to viscousity.
gamma_g = 1./delta(days=28.9).total_seconds()
f = 1.e-4  # Coriolis parameter.


def decaying_eddy_grid(xdim=400, ydim=400):  # Define 2D flat, square grid for testing purposes.

    depth = np.zeros(1, dtype=np.float32)
    time = np.arange(0., 2. * 86400., 60.*5., dtype=np.float64)
    lon = np.linspace(0, 70000, xdim, dtype=np.float32)
    lat = np.linspace(5000, 12000, ydim, dtype=np.float32)

    U = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)
    V = np.zeros((lon.size, lat.size, time.size), dtype=np.float32)

    for t in range(time.size):
        U[:, :, t] = u_g*np.exp(-gamma_g*time[t]) + (u_0-u_g)*np.exp(-gamma*time[t])*np.cos(f*time[t])
        V[:, :, t] = -(u_0-u_g)*np.exp(-gamma*time[t])*np.sin(f*time[t])

    return Grid.from_data(U, lon, lat, V, lon, lat, depth, time, mesh='flat')


def true_values(t):  # Calculate the expected values for particle 1 at the endtime.

    x = 10000. + (u_g/gamma_g)*(1-np.exp(-gamma_g*t)) + f*((u_0-u_g)/(f**2 + gamma**2))*((gamma/f) + np.exp(-gamma*t)*(np.sin(f*t) - (gamma/f)*np.cos(f*t)))
    y = 10000. - ((u_0-u_g)/(f**2+gamma**2))*f*(1 - np.exp(-gamma*t)*(np.cos(f*t) + (gamma/f)*np.sin(f*t)))

    return [x, y]


def decaying_example(grid, mode='jit', method=AdvectionRK4):

    npart = 1
    pset = grid.ParticleSet(size=npart, pclass=ptype[mode],
                            start=(10000., 10000.),
                            finish=(20000., 20000.))

    endtime = delta(days=2)
    dt = delta(minutes=5)
    interval = delta(hours=1)

    pset.execute(method, endtime=endtime, dt=dt, interval=interval,
                 output_file=pset.ParticleFile(name="DecayingParticle"), show_movie=False)

    return pset


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_rotation_example(mode):
    grid = decaying_eddy_grid()
    pset = decaying_example(grid, mode=mode)
    vals = true_values(pset[0].time)
    assert(np.allclose(pset[0].lon, vals[0], 1e-4))    # Check advected values against calculated values.
    assert(np.allclose(pset[0].lat, vals[1], 1e-4))

if __name__ == "__main__":
    filename = 'decaying_eddy'
    grid = decaying_eddy_grid()
    grid.write(filename)

    pset = decaying_example(grid)

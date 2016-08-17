from parcels import Grid, JITParticle, ScipyParticle, Variable
from parcels import AdvectionRK4
import numpy as np
from datetime import timedelta as delta


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def DelayStart(particle, grid, time, dt):
    """An example kernel showing how particle starts can be delayed, using a starttime variable
    """
    if time > particle.starttime:
        particle.active = 1


def pensinsula_example(mode, npart=10):

    grid = Grid.from_nemo('examples/Peninsula_data/peninsula', extra_vars={'P': 'P'})

    class DelayStartParticle(ptype[mode]):
        starttime = Variable('starttime', dtype=np.float32, default=0.)

    # Initialise particles
    x = 3. * (1. / 1.852 / 60)  # 3 km offset from boundary
    y = (grid.U.lat[0] + x, grid.U.lat[-1] - x)  # latitude range, including offsets

    pset = grid.ParticleSet(npart, pclass=DelayStartParticle, start=(x, y[0]), finish=(x, y[1]))
    p = 0
    for particle in pset:
        particle.active = -1  # Using -1 value to signal that particle has not yet started
        particle.starttime = 3600 * p  # set starttime for each particle
        p += 1

    pset.execute(AdvectionRK4 + pset.Kernel(DelayStart),
                 runtime=delta(hours=24), dt=delta(minutes=5),
                 interval=delta(hours=1), show_movie=True)


if __name__ == "__main__":
    pensinsula_example('jit')

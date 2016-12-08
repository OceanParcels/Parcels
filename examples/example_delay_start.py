from parcels import Grid, JITParticle, ScipyParticle
from parcels import AdvectionRK4
import numpy as np
from datetime import timedelta as delta
import pytest


ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_delay_start_example(mode, npart=10, show_movie=False):
    """Example script that shows how to 'delay' the start of particle advection.
    This is useful for example when particles need to be started at different times

    In this example, we use pset.add statements to add one particle every hour
    in the peninsula grid. Note that the title in the movie may not show correct time"""

    grid = Grid.from_nemo('examples/Peninsula_data/peninsula', extra_vars={'P': 'P'})

    # Initialise particles as in the Peninsula example
    x = 3. * (1. / 1.852 / 60)  # 3 km offset from boundary
    y = (grid.U.lat[0] + x, grid.U.lat[-1] - x)  # latitude range, including offsets

    lat = np.linspace(y[0], y[1], npart, dtype=np.float32)
    pset = grid.ParticleSet(0, lon=[], lat=[], pclass=ptype[mode])

    delaytime = delta(hours=1)  # delay time between particle releases
    
    # Since we are going to add particles during runtime, we need "indexed" NetCDF file
    output_file = pset.ParticleFile(name="DelayParticle", type="indexed")

    for t in range(npart):
        pset.add(ptype[mode](lon=x, lat=lat[t], grid=grid))
        pset.execute(AdvectionRK4, runtime=delaytime, dt=delta(minutes=5),
                     interval=delta(hours=1), show_movie=show_movie,
                     starttime=delaytime*t, output_file=output_file)

    # Note that time on the movie is not parsed correctly
    pset.execute(AdvectionRK4, runtime=delta(hours=24)-npart*delaytime,
                 starttime=delaytime*npart, dt=delta(minutes=5), interval=delta(hours=1),
                 show_movie=show_movie, output_file=output_file)

    londist = np.array([(p.lon - x) for p in pset])
    assert(londist > 0.1).all()


if __name__ == "__main__":
    test_delay_start_example('jit', show_movie=True)

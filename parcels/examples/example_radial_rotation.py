import math
from datetime import timedelta as delta

import numpy as np
import pytest

from parcels import AdvectionRK4
from parcels import FieldSet
from parcels import JITParticle
from parcels import ParticleSet
from parcels import ScipyParticle

ptype = {'scipy': ScipyParticle, 'jit': JITParticle}


def radial_rotation_fieldset(xdim=200, ydim=200):  # Define 2D flat, square fieldset for testing purposes.

    lon = np.linspace(0, 60, xdim, dtype=np.float32)
    lat = np.linspace(0, 60, ydim, dtype=np.float32)

    x0 = 30.                                   # Define the origin to be the centre of the Field.
    y0 = 30.

    U = np.zeros((ydim, xdim), dtype=np.float32)
    V = np.zeros((ydim, xdim), dtype=np.float32)

    T = delta(days=1)
    omega = 2*np.pi/T.total_seconds()          # Define the rotational period as 1 day.

    for i in range(lon.size):
        for j in range(lat.size):

            r = np.sqrt((lon[i]-x0)**2 + (lat[j]-y0)**2)  # Define radial displacement.
            assert(r >= 0.)
            assert(r <= np.sqrt(x0**2 + y0**2))

            theta = math.atan2((lat[j]-y0), (lon[i]-x0))  # Define the polar angle.
            assert(abs(theta) <= np.pi)

            U[j, i] = r * math.sin(theta) * omega
            V[j, i] = -r * math.cos(theta) * omega

    data = {'U': U, 'V': V}
    dimensions = {'lon': lon, 'lat': lat}
    return FieldSet.from_data(data, dimensions, mesh='flat')


def true_values(age):  # Calculate the expected values for particle 2 at the endtime.

    x = 20*math.sin(2*np.pi*age/(24.*60.**2)) + 30.
    y = 20*math.cos(2*np.pi*age/(24.*60.**2)) + 30.

    return [x, y]


def rotation_example(fieldset, outfile, mode='jit', method=AdvectionRK4):

    npart = 2          # Test two particles on the rotating fieldset.
    pset = ParticleSet.from_line(fieldset, size=npart, pclass=ptype[mode],
                                 start=(30., 30.),
                                 finish=(30., 50.))  # One particle in centre, one on periphery of Field.

    runtime = delta(hours=17)
    dt = delta(minutes=5)
    outputdt = delta(hours=1)

    pset.execute(method, runtime=runtime, dt=dt, moviedt=None,
                 output_file=pset.ParticleFile(name=outfile, outputdt=outputdt))

    return pset


@pytest.mark.parametrize('mode', ['scipy', 'jit'])
def test_rotation_example(mode, tmpdir):
    fieldset = radial_rotation_fieldset()
    outfile = tmpdir.join("RadialParticle")
    pset = rotation_example(fieldset, outfile, mode=mode)
    assert(pset[0].lon == 30. and pset[0].lat == 30.)  # Particle at centre of Field remains stationary.
    vals = true_values(pset.time[1])
    assert(np.allclose(pset[1].lon, vals[0], 1e-5))    # Check advected values against calculated values.
    assert(np.allclose(pset[1].lat, vals[1], 1e-5))


if __name__ == "__main__":
    filename = 'radial_rotation'
    fieldset = radial_rotation_fieldset()
    fieldset.write(filename)

    outfile = "RadialParticle"
    rotation_example(fieldset, outfile)

import math
from datetime import timedelta

import numpy as np

import parcels


def radial_rotation_fieldset(
    xdim=200, ydim=200
):  # Define 2D flat, square fieldset for testing purposes.
    lon = np.linspace(0, 60, xdim, dtype=np.float32)
    lat = np.linspace(0, 60, ydim, dtype=np.float32)

    x0 = 30.0  # Define the origin to be the centre of the Field.
    y0 = 30.0

    U = np.zeros((ydim, xdim), dtype=np.float32)
    V = np.zeros((ydim, xdim), dtype=np.float32)

    T = timedelta(days=1)
    omega = 2 * np.pi / T.total_seconds()  # Define the rotational period as 1 day.

    for i in range(lon.size):
        for j in range(lat.size):
            r = np.sqrt(
                (lon[i] - x0) ** 2 + (lat[j] - y0) ** 2
            )  # Define radial displacement.
            assert r >= 0.0
            assert r <= np.sqrt(x0**2 + y0**2)

            theta = math.atan2((lat[j] - y0), (lon[i] - x0))  # Define the polar angle.
            assert abs(theta) <= np.pi

            U[j, i] = r * math.sin(theta) * omega
            V[j, i] = -r * math.cos(theta) * omega

    data = {"U": U, "V": V}
    dimensions = {"lon": lon, "lat": lat}
    return parcels.FieldSet.from_data(data, dimensions, mesh="flat")


def true_values(age):  # Calculate the expected values for particle 2 at the endtime.
    x = 20 * math.sin(2 * np.pi * age / (24.0 * 60.0**2)) + 30.0
    y = 20 * math.cos(2 * np.pi * age / (24.0 * 60.0**2)) + 30.0

    return [x, y]


def rotation_example(fieldset, outfile, method=parcels.AdvectionRK4):
    npart = 2  # Test two particles on the rotating fieldset.
    pset = parcels.ParticleSet.from_line(
        fieldset,
        size=npart,
        pclass=parcels.Particle,
        start=(30.0, 30.0),
        finish=(30.0, 50.0),
    )  # One particle in centre, one on periphery of Field.

    runtime = timedelta(hours=17)
    dt = timedelta(minutes=5)
    outputdt = timedelta(hours=1)

    pset.execute(
        method,
        runtime=runtime,
        dt=dt,
        output_file=pset.ParticleFile(name=outfile, outputdt=outputdt),
    )

    return pset


def test_rotation_example(tmpdir):
    fieldset = radial_rotation_fieldset()
    outfile = tmpdir.join("RadialParticle")
    pset = rotation_example(fieldset, outfile)
    assert (
        pset[0].lon == 30.0 and pset[0].lat == 30.0
    )  # Particle at centre of Field remains stationary.
    vals = true_values(pset.time[1])
    assert np.allclose(
        pset[1].lon, vals[0], 1e-5
    )  # Check advected values against calculated values.
    assert np.allclose(pset[1].lat, vals[1], 1e-5)


def main():
    fieldset = radial_rotation_fieldset()
    outfile = "RadialParticle"
    rotation_example(fieldset, outfile)


if __name__ == "__main__":
    main()

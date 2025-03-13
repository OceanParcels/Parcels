from datetime import timedelta

import numpy as np

import parcels

# Define some constants.
u_g = 0.04  # Geostrophic current
u_0 = 0.3  # Initial speed in x dirrection. v_0 = 0
gamma = (
    1.0 / timedelta(days=2.89).total_seconds()
)  # Dissipitave effects due to viscousity.
gamma_g = 1.0 / timedelta(days=28.9).total_seconds()
f = 1.0e-4  # Coriolis parameter.
start_lon = [10000.0]  # Define the start longitude and latitude for the particle.
start_lat = [10000.0]


def decaying_moving_eddy_fieldset(
    xdim=2, ydim=2
):  # Define 2D flat, square fieldset for testing purposes.
    """Simulate an ocean that accelerates subject to Coriolis force
    and dissipative effects, upon which a geostrophic current is
    superimposed.

    The original test description can be found in: N. Fabbroni, 2009,
    Numerical Simulation of Passive tracers dispersion in the sea,
    Ph.D. dissertation, University of Bologna
    http://amsdottorato.unibo.it/1733/1/Fabbroni_Nicoletta_Tesi.pdf
    """
    depth = np.zeros(1, dtype=np.float32)
    time = np.arange(0.0, 2.0 * 86400.0 + 1e-5, 60.0 * 5.0, dtype=np.float64)
    lon = np.linspace(0, 20000, xdim, dtype=np.float32)
    lat = np.linspace(5000, 12000, ydim, dtype=np.float32)

    U = np.zeros((time.size, lat.size, lon.size), dtype=np.float32)
    V = np.zeros((time.size, lat.size, lon.size), dtype=np.float32)

    for t in range(time.size):
        U[t, :, :] = u_g * np.exp(-gamma_g * time[t]) + (u_0 - u_g) * np.exp(
            -gamma * time[t]
        ) * np.cos(f * time[t])
        V[t, :, :] = -(u_0 - u_g) * np.exp(-gamma * time[t]) * np.sin(f * time[t])

    data = {"U": U, "V": V}
    dimensions = {"lon": lon, "lat": lat, "depth": depth, "time": time}
    return parcels.FieldSet.from_data(data, dimensions, mesh="flat")


def true_values(
    t, x_0, y_0
):  # Calculate the expected values for particles at the endtime, given their start location.
    x = (
        x_0
        + (u_g / gamma_g) * (1 - np.exp(-gamma_g * t))
        + f
        * ((u_0 - u_g) / (f**2 + gamma**2))
        * (
            (gamma / f)
            + np.exp(-gamma * t) * (np.sin(f * t) - (gamma / f) * np.cos(f * t))
        )
    )
    y = y_0 - ((u_0 - u_g) / (f**2 + gamma**2)) * f * (
        1 - np.exp(-gamma * t) * (np.cos(f * t) + (gamma / f) * np.sin(f * t))
    )

    return np.array([x, y])


def decaying_moving_example(fieldset, outfile, method=parcels.AdvectionRK4):
    pset = parcels.ParticleSet(
        fieldset, pclass=parcels.Particle, lon=start_lon, lat=start_lat
    )

    dt = timedelta(minutes=5)
    runtime = timedelta(days=2)
    outputdt = timedelta(hours=1)

    pset.execute(
        method,
        runtime=runtime,
        dt=dt,
        output_file=pset.ParticleFile(name=outfile, outputdt=outputdt),
    )

    return pset


def test_rotation_example(tmpdir):
    outfile = tmpdir.join("DecayingMovingParticle.zarr")
    fieldset = decaying_moving_eddy_fieldset()
    pset = decaying_moving_example(fieldset, outfile)
    vals = true_values(
        pset[0].time, start_lon, start_lat
    )  # Calculate values for the particle.
    assert np.allclose(
        np.array([[pset[0].lon], [pset[0].lat]]), vals, 1e-2
    )  # Check advected values against calculated values.


def main():
    outfile = "DecayingMovingParticle.zarr"
    fieldset = decaying_moving_eddy_fieldset()

    decaying_moving_example(fieldset, outfile)


if __name__ == "__main__":
    main()

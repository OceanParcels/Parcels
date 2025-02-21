import random
from datetime import timedelta

import numpy as np
import pytest

import parcels


def mesh_conversion(mesh):
    return (1852.0 * 60) if mesh == "spherical" else 1.0


@pytest.mark.parametrize("mesh", ["flat", "spherical"])
def test_brownian_example(mesh, npart=3000):
    fieldset = parcels.FieldSet.from_data(
        {"U": 0, "V": 0}, {"lon": 0, "lat": 0}, mesh=mesh
    )

    # Set diffusion constants.
    kh_zonal = 100  # in m^2/s
    kh_meridional = 100  # in m^2/s

    # Create field of constant Kh_zonal and Kh_meridional
    fieldset.add_field(parcels.Field("Kh_zonal", kh_zonal, lon=0, lat=0, mesh=mesh))
    fieldset.add_field(
        parcels.Field("Kh_meridional", kh_meridional, lon=0, lat=0, mesh=mesh)
    )

    # Set random seed
    random.seed(123456)

    runtime = timedelta(days=1)

    random.seed(1234)
    pset = parcels.ParticleSet(
        fieldset=fieldset,
        pclass=parcels.Particle,
        lon=np.zeros(npart),
        lat=np.zeros(npart),
    )
    pset.execute(
        pset.Kernel(parcels.DiffusionUniformKh), runtime=runtime, dt=timedelta(hours=1)
    )

    expected_std_x = np.sqrt(2 * kh_zonal * runtime.total_seconds())
    expected_std_y = np.sqrt(2 * kh_meridional * runtime.total_seconds())

    ys = pset.lat * mesh_conversion(mesh)
    xs = pset.lon * mesh_conversion(
        mesh
    )  # since near equator, we do not need to care about curvature effect

    tol = 250  # 250m tolerance
    assert np.allclose(np.std(xs), expected_std_x, atol=tol)
    assert np.allclose(np.std(ys), expected_std_y, atol=tol)
    assert np.allclose(np.mean(xs), 0, atol=tol)
    assert np.allclose(np.mean(ys), 0, atol=tol)


if __name__ == "__main__":
    test_brownian_example("spherical", npart=2000)

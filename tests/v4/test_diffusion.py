import random

import numpy as np
import pytest

from parcels._datasets.structured.generic import simple_UV_dataset
from parcels.application_kernels import DiffusionUniformKh
from parcels.field import Field, VectorField
from parcels.fieldset import FieldSet
from parcels.particleset import ParticleSet
from parcels.xgrid import _XGRID_AXES, XGrid


def BiLinear(  # TODO move to interpolation file
    field: Field,
    ti: int,
    position: dict[_XGRID_AXES, tuple[int, float | np.ndarray]],
    tau: np.float32 | np.float64,
    t: np.float32 | np.float64,
    z: np.float32 | np.float64,
    y: np.float32 | np.float64,
    x: np.float32 | np.float64,
):
    """Bilinear interpolation on a regular grid."""
    xi, xsi = position["X"]
    yi, eta = position["Y"]
    zi, zeta = position["Z"]

    data = field.data.data[:, zi, yi : yi + 2, xi : xi + 2]
    data = (1 - tau) * data[ti, :, :] + tau * data[ti + 1, :, :]

    return (
        (1 - xsi) * (1 - eta) * data[0, 0]
        + xsi * (1 - eta) * data[0, 1]
        + xsi * eta * data[1, 1]
        + (1 - xsi) * eta * data[1, 0]
    )


@pytest.mark.parametrize("mesh_type", ["spherical", "flat"])
def test_fieldKh_Brownian(mesh_type):
    kh_zonal = 100
    kh_meridional = 50
    mesh_conversion = 1 / 1852.0 / 60 if mesh_type == "spherical" else 1

    ds = simple_UV_dataset(dims=(2, 1, 2, 2), mesh_type=mesh_type)
    ds["lon"].data = np.array([-1e6, 1e6])
    ds["lat"].data = np.array([-1e6, 1e6])
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, mesh_type=mesh_type, interp_method=BiLinear)
    V = Field("V", ds["V"], grid, mesh_type=mesh_type, interp_method=BiLinear)
    ds["Kh_zonal"] = (["time", "depth", "YG", "XG"], np.full((2, 1, 2, 2), kh_zonal))
    ds["Kh_meridional"] = (["time", "depth", "YG", "XG"], np.full((2, 1, 2, 2), kh_meridional))
    Kh_zonal = Field("Kh_zonal", ds["Kh_zonal"], grid=grid, mesh_type=mesh_type, interp_method=BiLinear)
    Kh_meridional = Field("Kh_meridional", ds["Kh_meridional"], grid=grid, mesh_type=mesh_type, interp_method=BiLinear)
    UV = VectorField("UV", U, V)
    fieldset = FieldSet([U, V, UV, Kh_zonal, Kh_meridional])

    npart = 100
    runtime = np.timedelta64(2, "h")

    random.seed(1234)
    pset = ParticleSet(fieldset=fieldset, lon=np.zeros(npart), lat=np.zeros(npart))
    pset.execute(pset.Kernel(DiffusionUniformKh), runtime=runtime, dt=np.timedelta64(1, "h"))

    expected_std_lon = np.sqrt(2 * kh_zonal * mesh_conversion**2 * (runtime / np.timedelta64(1, "s")))
    expected_std_lat = np.sqrt(2 * kh_meridional * mesh_conversion**2 * (runtime / np.timedelta64(1, "s")))

    tol = 500 * mesh_conversion  # effectively 500 m errors
    assert np.allclose(np.std(pset.lat), expected_std_lat, atol=tol)
    assert np.allclose(np.std(pset.lon), expected_std_lon, atol=tol)
    assert np.allclose(np.mean(pset.lon), 0, atol=tol)
    assert np.allclose(np.mean(pset.lat), 0, atol=tol)

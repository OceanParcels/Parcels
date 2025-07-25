import numpy as np
import pytest

from parcels._datasets.structured.generic import datasets
from parcels.application_kernels import AdvectionRK4
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
    data = (1 - tau) * data[0, :, :] + tau * data[1, :, :]

    return (
        (1 - xsi) * (1 - eta) * data[0, 0]
        + xsi * (1 - eta) * data[0, 1]
        + xsi * eta * data[1, 1]
        + (1 - xsi) * eta * data[1, 0]
    )


def TriLinear(  # TODO move to interpolation file
    field: Field,
    ti: int,
    position: dict[_XGRID_AXES, tuple[int, float | np.ndarray]],
    tau: np.float32 | np.float64,
    t: np.float32 | np.float64,
    z: np.float32 | np.float64,
    y: np.float32 | np.float64,
    x: np.float32 | np.float64,
):
    """Trilinear interpolation on a regular grid."""
    xi, xsi = position["X"]
    yi, eta = position["Y"]
    zi, zeta = position["Z"]

    data = field.data.data[:, zi : zi + 2, yi : yi + 2, xi : xi + 2]
    data = (1 - tau) * data[0, :, :, :] + tau * data[1, :, :, :]
    data = (1 - zeta) * data[0, :, :] + zeta * data[1, :, :]

    return (
        (1 - xsi) * (1 - eta) * data[0, 0]
        + xsi * (1 - eta) * data[0, 1]
        + xsi * eta * data[1, 1]
        + (1 - xsi) * eta * data[1, 0]
    )


@pytest.mark.parametrize("mesh_type", ["spherical", "flat"])
def test_advection_zonal(mesh_type, npart=10):
    """Particles at high latitude move geographically faster due to the pole correction in `GeographicPolar`."""
    ds = datasets[f"pure_zonal_flow_{mesh_type}"]
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, mesh_type=mesh_type, interp_method=BiLinear)
    V = Field("V", ds["V"], grid, mesh_type=mesh_type, interp_method=BiLinear)
    UV = VectorField("UV", U, V)
    fieldset2D = FieldSet([U, V, UV])

    pset2D = ParticleSet(fieldset2D, lon=np.zeros(npart) + 20.0, lat=np.linspace(0, 80, npart))
    pset2D.execute(AdvectionRK4, runtime=np.timedelta64(2, "h"), dt=np.timedelta64(15, "m"))

    if mesh_type == "spherical":
        assert (np.diff(pset2D.lon) > 1.0e-4).all()
    else:
        assert (np.diff(pset2D.lon) < 1.0e-4).all()


def test_advection_3D(npart=10):
    """Flat 2D zonal flow that increases linearly with depth from 0 m/s to 1 m/s."""
    ds = datasets["pure_zonal_flow_flat"]
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, interp_method=TriLinear)
    U.data[:, 0, :, :] = 0.0  # Set U to 0 at the surface
    V = Field("V", ds["V"], grid, interp_method=TriLinear)
    UV = VectorField("UV", U, V)
    fieldset = FieldSet([U, V, UV])

    pset = ParticleSet(fieldset, lon=np.zeros(npart), lat=np.zeros(npart), depth=np.linspace(0.1, 0.9, npart))
    pset.execute(AdvectionRK4, runtime=np.timedelta64(2, "h"), dt=np.timedelta64(15, "m"))

    expected_lon = pset.depth * (pset.time - fieldset.time_interval.left) / np.timedelta64(1, "s")
    assert np.allclose(expected_lon, pset.lon_nextloop, atol=1.0e-1)

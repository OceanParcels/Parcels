import numpy as np
import pytest

from parcels._datasets.structured.generic import datasets
from parcels.field import Field, VectorField
from parcels.xgrid import _XGRID_AXES, XGrid


def BiRectiLinear(  # TODO move to interpolation file
    field: Field,
    ti: int,
    position: dict[_XGRID_AXES, tuple[int, float | np.ndarray]],
    tau: np.float32 | np.float64,
    t: np.float32 | np.float64,
    z: np.float32 | np.float64,
    y: np.float32 | np.float64,
    x: np.float32 | np.float64,
):
    """Bilinear interpolation on a rectilinear grid."""
    xi, xsi = position["X"]
    yi, eta = position["Y"]

    data = field.data.data[:, :, yi : yi + 2, xi : xi + 2]
    val_t0 = (
        (1 - xsi) * (1 - eta) * data[0, 0, 0, 0]
        + xsi * (1 - eta) * data[0, 0, 0, 1]
        + xsi * eta * data[0, 0, 1, 1]
        + (1 - xsi) * eta * data[0, 0, 1, 0]
    )

    val_t1 = (
        (1 - xsi) * (1 - eta) * data[1, 0, 0, 0]
        + xsi * (1 - eta) * data[1, 0, 0, 1]
        + xsi * eta * data[1, 0, 1, 1]
        + (1 - xsi) * eta * data[1, 0, 1, 0]
    )
    return val_t0 * (1 - tau) + val_t1 * tau


@pytest.mark.parametrize("mesh_type", ["spherical", "flat"])
def test_interpolation_mesh_type(mesh_type, npart=10):
    ds = datasets[f"pure_zonal_flow_{mesh_type}"]
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, mesh_type=mesh_type, interp_method=BiRectiLinear)
    V = Field("V", ds["V"], grid, mesh_type=mesh_type, interp_method=BiRectiLinear)
    UV = VectorField("UV", U, V)

    lat = 30.0
    time = U.time_interval.left
    u_expected = 1.0 if mesh_type == "flat" else 1.0 / (1852 * 60 * np.cos(np.radians(lat)))

    assert np.isclose(U[time, 0, lat, 0], u_expected, atol=1e-7)
    assert V[time, 0, lat, 0] == 0.0

    u, v = UV[time, 0, lat, 0]
    assert np.isclose(u, u_expected, atol=1e-7)
    assert v == 0.0

    assert U.eval(time, 0, lat, 0, applyConversion=False) == 1

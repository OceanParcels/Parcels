import numpy as np
import pytest
import xarray as xr

from parcels import (
    AdvectionRK4,
    Field,
    FieldSet,
    ParticleSet,
)
from parcels.field import Field, VectorField
from parcels.fieldset import FieldSet
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
    return (val_t0 * (1 - tau) + val_t1 * tau).values


@pytest.mark.parametrize("mesh_type", ["spherical", "flat"])
def test_advection_zonal(mesh_type, npart=10):
    """Particles at high latitude move geographically faster due to the pole correction in `GeographicPolar`."""
    dims = (7, 5, 30, 4)  # time, depth, lat, lon
    max_lon = 180.0 if mesh_type == "spherical" else 1e6

    ds = xr.Dataset(
        {"U": (["time", "depth", "YG", "XG"], np.ones(dims)), "V": (["time", "depth", "YG", "XG"], np.zeros(dims))},
        coords={
            "time": (["time"], xr.date_range("2000", "2001", dims[0]), {"axis": "T"}),
            "depth": (["depth"], np.linspace(0, 10, dims[1]), {"axis": "Z"}),
            "YC": (["YC"], np.arange(dims[2]) + 0.5, {"axis": "Y"}),
            "YG": (["YG"], np.arange(dims[2]), {"axis": "Y", "c_grid_axis_shift": -0.5}),
            "XC": (["XC"], np.arange(dims[3]) + 0.5, {"axis": "X"}),
            "XG": (["XG"], np.arange(dims[3]), {"axis": "X", "c_grid_axis_shift": -0.5}),
            "lat": (["YG"], np.linspace(-90, 90, dims[2]), {"axis": "Y", "c_grid_axis_shift": 0.5}),
            "lon": (["XG"], np.linspace(-max_lon, max_lon, dims[3]), {"axis": "X", "c_grid_axis_shift": -0.5}),
        },
    )
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U"], grid, mesh_type=mesh_type, interp_method=BiRectiLinear)
    V = Field("V", ds["V"], grid, mesh_type=mesh_type, interp_method=BiRectiLinear)
    UV = VectorField("UV", U, V)
    fieldset2D = FieldSet([U, V, UV])

    pset2D = ParticleSet(fieldset2D, lon=np.zeros(npart) + 20.0, lat=np.linspace(0, 80, npart))
    pset2D.execute(AdvectionRK4, runtime=np.timedelta64(2, "h"), dt=np.timedelta64(15, "m"))

    if mesh_type == "spherical":
        assert (np.diff(pset2D.lon) > 1.0e-4).all()
    else:
        assert (np.diff(pset2D.lon) < 1.0e-4).all()

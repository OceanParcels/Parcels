import math

import numpy as np
import pandas as pd
import uxarray as ux

__all__ = ["Nx", "datasets"]

Nx = 20
vmax = 1.0
delta = 0.1


def _stommel_gyre_delaunay():
    """
    Stommel gyre on a Delaunay grid. the naming convention of the dataset and grid is consistent with what is 
    provided by UXArray when reading in FESOM2 datasets.
    This dataset is a single vertical layer of a barotropic ocean gyre on a square domain with closed boundaries.
    The velocity field provides a slow moving interior circulation and a western boundary current. All fields are placed
    on the vertices of the grid and at the element vertical faces.
    """
    lon, lat = np.meshgrid(np.linspace(0, 60.0, Nx, dtype=np.float32), np.linspace(0, 60.0, Nx, dtype=np.float32))
    lon_flat = lon.ravel()
    lat_flat = lat.ravel()

    # mask any point on one of the boundaries
    mask = (
        np.isclose(lon_flat, 0.0) | np.isclose(lon_flat, 60.0) | np.isclose(lat_flat, 0.0) | np.isclose(lat_flat, 60.0)
    )

    boundary_points = np.flatnonzero(mask)

    uxgrid = ux.Grid.from_points(
        (lon_flat, lat_flat),
        method="regional_delaunay",
        boundary_points=boundary_points,
    )

    # Define arrays U (zonal), V (meridional) and P (sea surface height)
    U = np.zeros((1, 1, lat.size), dtype=np.float64)
    V = np.zeros((1, 1, lat.size), dtype=np.float64)
    P = np.zeros((1, 1, lat.size), dtype=np.float64)

    for i, (x, y) in enumerate(zip(lon_flat, lat_flat, strict=False)):
        xi = x / 60.0
        yi = y / 60.0

        P[0, 0, i] = -vmax * delta * (1 - xi) * (math.exp(-xi / delta) - 1) * np.sin(math.pi * yi)
        U[0, 0, i] = -vmax * (1 - math.exp(-xi / delta) - xi) * np.cos(math.pi * yi)
        V[0, 0, i] = vmax * ((2.0 - xi) * math.exp(-xi / delta) - 1) * np.sin(math.pi * yi)

    u = ux.UxDataArray(
        data=U,
        name="U",
        uxgrid=uxgrid,
        dims=["time", "nz1", "n_node"],
        coords=dict(
            time=(["time"], pd.to_datetime(["2000-01-01"])),
            nz1=(["nz1"], [0]),
        ),
        attrs=dict(
            description="zonal velocity", units="m/s", location="node", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    v = ux.UxDataArray(
        data=V,
        name="V",
        uxgrid=uxgrid,
        dims=["time", "nz1", "n_node"],
        coords=dict(
            time=(["time"], pd.to_datetime(["2000-01-01"])),
            nz1=(["nz1"], [0]),
        ),
        attrs=dict(
            description="meridional velocity", units="m/s", location="node", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    p = ux.UxDataArray(
        data=P,
        name="p",
        uxgrid=uxgrid,
        dims=["time", "nz1", "n_node"],
        coords=dict(
            time=(["time"], pd.to_datetime(["2000-01-01"])),
            nz1=(["nz1"], [0]),
        ),
        attrs=dict(description="pressure", units="N/m^2", location="node", mesh="delaunay", Conventions="UGRID-1.0"),
    )

    return ux.UxDataset({"U": u, "V": v, "p": p}, uxgrid=uxgrid)


datasets = {
    "stommel_gyre_delaunay": _stommel_gyre_delaunay(),
}

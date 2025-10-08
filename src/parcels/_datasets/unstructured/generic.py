import math

import numpy as np
import uxarray as ux
import xarray as xr

__all__ = ["Nx", "datasets"]

T = 13
Nx = 20
vmax = 1.0
delta = 0.1
TIME = xr.date_range("2000", "2001", T)


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
    zf = np.linspace(0.0, 1000.0, 2, endpoint=True, dtype=np.float32)  # Vertical element faces
    zc = 0.5 * (zf[:-1] + zf[1:])  # Vertical element centers
    nz = zf.size
    nz1 = zc.size

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
    uxgrid.attrs["Conventions"] = "UGRID-1.0"

    # Define arrays U (zonal), V (meridional) and P (sea surface height)
    U = np.zeros((1, nz1, uxgrid.n_face), dtype=np.float64)
    V = np.zeros((1, nz1, uxgrid.n_face), dtype=np.float64)
    W = np.zeros((1, nz, lat.size), dtype=np.float64)
    P = np.zeros((1, nz1, uxgrid.n_face), dtype=np.float64)

    for i, (x, y) in enumerate(zip(uxgrid.face_lon, uxgrid.face_lat, strict=False)):
        xi = x / 60.0
        yi = y / 60.0

        P[0, 0, i] = -vmax * delta * (1 - xi) * (math.exp(-xi / delta) - 1) * np.sin(math.pi * yi)
        U[0, 0, i] = -vmax * (1 - math.exp(-xi / delta) - xi) * np.cos(math.pi * yi)
        V[0, 0, i] = vmax * ((2.0 - xi) * math.exp(-xi / delta) - 1) * np.sin(math.pi * yi)

    u = ux.UxDataArray(
        data=U,
        name="U",
        uxgrid=uxgrid,
        dims=["time", "nz1", "n_face"],
        coords=dict(
            time=(["time"], [TIME[0]]),
            nz1=(["nz1"], zc),
        ),
        attrs=dict(
            description="zonal velocity", units="m/s", location="node", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    v = ux.UxDataArray(
        data=V,
        name="V",
        uxgrid=uxgrid,
        dims=["time", "nz1", "n_face"],
        coords=dict(
            time=(["time"], [TIME[0]]),
            nz1=(["nz1"], zc),
        ),
        attrs=dict(
            description="meridional velocity", units="m/s", location="node", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    w = ux.UxDataArray(
        data=W,
        name="W",
        uxgrid=uxgrid,
        dims=["time", "nz", "n_node"],
        coords=dict(
            time=(["time"], [TIME[0]]),
            nz=(["nz"], zf),
        ),
        attrs=dict(
            description="meridional velocity", units="m/s", location="node", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    p = ux.UxDataArray(
        data=P,
        name="p",
        uxgrid=uxgrid,
        dims=["time", "nz1", "n_face"],
        coords=dict(
            time=(["time"], [TIME[0]]),
            nz1=(["nz1"], zc),
        ),
        attrs=dict(description="pressure", units="N/m^2", location="node", mesh="delaunay", Conventions="UGRID-1.0"),
    )

    return ux.UxDataset({"U": u, "V": v, "W": w, "p": p}, uxgrid=uxgrid)


def _fesom2_square_delaunay_uniform_z_coordinate():
    """
    Delaunay grid with uniform z-coordinate, mimicking a FESOM2 dataset.
    This dataset consists of a square domain with closed boundaries, where the grid is generated using Delaunay triangulation.
    The bottom topography is flat and uniform, and the vertical grid spacing is constant with 10 layers spanning [0,1000.0]
    The lateral velocity field components are non-zero constant, and the vertical velocity component is zero.
    The pressure field is constant.
    All fields are placed on location consistent with FESOM2 variable placement conventions
    """
    lon, lat = np.meshgrid(np.linspace(0, 60.0, Nx, dtype=np.float32), np.linspace(0, 60.0, Nx, dtype=np.float32))
    lon_flat = lon.ravel()
    lat_flat = lat.ravel()
    zf = np.linspace(0.0, 1000.0, 10, endpoint=True, dtype=np.float32)  # Vertical element faces
    zc = 0.5 * (zf[:-1] + zf[1:])  # Vertical element centers
    nz = zf.size
    nz1 = zc.size

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
    uxgrid.attrs["Conventions"] = "UGRID-1.0"

    # Define arrays U (zonal), V (meridional) and P (sea surface height)
    U = np.ones(
        (T, nz1, uxgrid.n_face), dtype=np.float64
    )  # Lateral velocity is on the element centers and face centers
    V = np.ones(
        (T, nz1, uxgrid.n_face), dtype=np.float64
    )  # Lateral velocity is on the element centers and face centers
    W = np.zeros(
        (T, nz, uxgrid.n_node), dtype=np.float64
    )  # Vertical velocity is on the element faces and face vertices
    P = np.ones((T, nz1, uxgrid.n_node), dtype=np.float64)  # Pressure is on the element centers and face vertices

    u = ux.UxDataArray(
        data=U,
        name="U",
        uxgrid=uxgrid,
        dims=["time", "nz1", "n_face"],
        coords=dict(
            time=(["time"], TIME),
            nz1=(["nz1"], zc),
        ),
        attrs=dict(
            description="zonal velocity", units="m/s", location="face", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    v = ux.UxDataArray(
        data=V,
        name="V",
        uxgrid=uxgrid,
        dims=["time", "nz1", "n_face"],
        coords=dict(
            time=(["time"], TIME),
            nz1=(["nz1"], zc),
        ),
        attrs=dict(
            description="meridional velocity", units="m/s", location="face", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    w = ux.UxDataArray(
        data=W,
        name="w",
        uxgrid=uxgrid,
        dims=["time", "nz", "n_node"],
        coords=dict(
            time=(["time"], TIME),
            nz=(["nz"], zf),
        ),
        attrs=dict(
            description="vertical velocity", units="m/s", location="node", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    p = ux.UxDataArray(
        data=P,
        name="p",
        uxgrid=uxgrid,
        dims=["time", "nz1", "n_node"],
        coords=dict(
            time=(["time"], TIME),
            nz1=(["nz1"], zc),
        ),
        attrs=dict(description="pressure", units="N/m^2", location="node", mesh="delaunay", Conventions="UGRID-1.0"),
    )

    return ux.UxDataset({"U": u, "V": v, "W": w, "p": p}, uxgrid=uxgrid)


def _fesom2_square_delaunay_antimeridian():
    """
    Delaunay grid that crosses the antimeridian with uniform z-coordinate, mimicking a FESOM2 dataset.
    This dataset consists of a square domain with closed boundaries, where the grid is generated using Delaunay triangulation.
    The bottom topography is flat and uniform, and the vertical grid spacing is constant with 10 layers spanning [0,1000.0]
    The lateral velocity field components are non-zero constant, and the vertical velocity component is zero.
    The pressure field is constant.
    All fields are placed on location consistent with FESOM2 variable placement conventions
    """
    lon, lat = np.meshgrid(
        np.linspace(-210.0, -150.0, Nx, dtype=np.float32), np.linspace(-40.0, 40.0, Nx, dtype=np.float32)
    )
    # wrap longitude from [-180,180]
    lon_flat = lon.ravel()
    lat_flat = lat.ravel()
    zf = np.linspace(0.0, 1000.0, 10, endpoint=True, dtype=np.float32)  # Vertical element faces
    zc = 0.5 * (zf[:-1] + zf[1:])  # Vertical element centers
    nz = zf.size
    nz1 = zc.size

    # mask any point on one of the boundaries
    mask = (
        np.isclose(lon_flat, -210.0)
        | np.isclose(lon_flat, -150.0)
        | np.isclose(lat_flat, -40.0)
        | np.isclose(lat_flat, 40.0)
    )

    boundary_points = np.flatnonzero(mask)

    uxgrid = ux.Grid.from_points(
        (lon_flat, lat_flat),
        method="regional_delaunay",
        boundary_points=boundary_points,
    )
    uxgrid.attrs["Conventions"] = "UGRID-1.0"

    # Define arrays U (zonal), V (meridional) and P (sea surface height)
    U = np.ones(
        (T, nz1, uxgrid.n_face), dtype=np.float64
    )  # Lateral velocity is on the element centers and face centers
    V = np.ones(
        (T, nz1, uxgrid.n_face), dtype=np.float64
    )  # Lateral velocity is on the element centers and face centers
    W = np.zeros(
        (T, nz, uxgrid.n_node), dtype=np.float64
    )  # Vertical velocity is on the element faces and face vertices
    P = np.ones((T, nz1, uxgrid.n_node), dtype=np.float64)  # Pressure is on the element centers and face vertices

    u = ux.UxDataArray(
        data=U,
        name="U",
        uxgrid=uxgrid,
        dims=["time", "nz1", "n_face"],
        coords=dict(
            time=(["time"], TIME),
            nz1=(["nz1"], zc),
        ),
        attrs=dict(
            description="zonal velocity", units="m/s", location="face", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    v = ux.UxDataArray(
        data=V,
        name="V",
        uxgrid=uxgrid,
        dims=["time", "nz1", "n_face"],
        coords=dict(
            time=(["time"], TIME),
            nz1=(["nz1"], zc),
        ),
        attrs=dict(
            description="meridional velocity", units="m/s", location="face", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    w = ux.UxDataArray(
        data=W,
        name="w",
        uxgrid=uxgrid,
        dims=["time", "nz", "n_node"],
        coords=dict(
            time=(["time"], TIME),
            nz=(["nz"], zf),
        ),
        attrs=dict(
            description="vertical velocity", units="m/s", location="node", mesh="delaunay", Conventions="UGRID-1.0"
        ),
    )
    p = ux.UxDataArray(
        data=P,
        name="p",
        uxgrid=uxgrid,
        dims=["time", "nz1", "n_node"],
        coords=dict(
            time=(["time"], TIME),
            nz1=(["nz1"], zc),
        ),
        attrs=dict(description="pressure", units="N/m^2", location="node", mesh="delaunay", Conventions="UGRID-1.0"),
    )

    return ux.UxDataset({"U": u, "V": v, "W": w, "p": p}, uxgrid=uxgrid)


datasets = {
    "stommel_gyre_delaunay": _stommel_gyre_delaunay(),
    "fesom2_square_delaunay_uniform_z_coordinate": _fesom2_square_delaunay_uniform_z_coordinate(),
    "fesom2_square_delaunay_antimeridian": _fesom2_square_delaunay_antimeridian(),
}

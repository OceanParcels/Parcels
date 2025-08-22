"""General helper functions and utilies for test suite."""

import struct
from pathlib import Path

import numpy as np
import xarray as xr

import parcels
from parcels import FieldSet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROOT = PROJECT_ROOT / "tests"
TEST_DATA = TEST_ROOT / "test_data"


def create_fieldset_unit_mesh(xdim=20, ydim=20, mesh="flat", transpose=False) -> FieldSet:
    """Standard unit mesh fieldset with U and V equivalent to longitude and latitude."""
    lon = np.linspace(0.0, 1.0, xdim, dtype=np.float32)
    lat = np.linspace(0.0, 1.0, ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    data = {"U": np.array(U, dtype=np.float32), "V": np.array(V, dtype=np.float32)}
    dimensions = {"lat": lat, "lon": lon}
    return FieldSet.from_data(data, dimensions, mesh=mesh, transpose=transpose)


def create_fieldset_zeros_3d(zdim=5, ydim=10, xdim=10):
    """3d fieldset with U, V, and W equivalent to longitude, latitude, and depth."""
    tdim = 20
    ds = xr.Dataset(
        {
            "U": (("time", "depth", "lat", "lon"), np.zeros((tdim, zdim, ydim, xdim))),
            "V": (("time", "depth", "lat", "lon"), np.zeros((tdim, zdim, ydim, xdim))),
            "W": (("time", "depth", "lat", "lon"), np.zeros((tdim, zdim, ydim, xdim))),
        },
        coords={
            "time": np.linspace(0, tdim - 1, tdim),
            "depth": np.linspace(0, 1, zdim),
            "lat": np.linspace(0, 1, ydim),
            "lon": np.linspace(0, 1, xdim),
        },
    )
    variables = {"U": "U", "V": "V", "W": "W"}
    dimensions = {"time": "time", "lon": "lon", "lat": "lat", "depth": "depth"}
    return FieldSet.from_xarray_dataset(ds, variables, dimensions, mesh="flat")


def create_fieldset_zeros_unit_mesh(xdim=100, ydim=100):
    """Standard unit mesh fieldset with flat mesh, and zero velocity."""
    data = {"U": np.zeros((ydim, xdim), dtype=np.float32), "V": np.zeros((ydim, xdim), dtype=np.float32)}
    dimensions = {"lon": np.linspace(0, 1, xdim, dtype=np.float32), "lat": np.linspace(0, 1, ydim, dtype=np.float32)}
    return FieldSet.from_data(data, dimensions, mesh="flat")


def create_fieldset_global(xdim=200, ydim=100):
    """Standard fieldset spanning the earth's coordinates with U and V equivalent to longitude and latitude in deg."""
    lon = np.linspace(-180, 180, xdim, dtype=np.float32)
    lat = np.linspace(-90, 90, ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    data = {"U": U, "V": V}
    dimensions = {"lon": lon, "lat": lat}
    return FieldSet.from_data(data, dimensions, mesh="flat", transpose=True)


def create_fieldset_zeros_conversion(mesh="spherical", xdim=200, ydim=100, mesh_conversion=1) -> FieldSet:
    """Zero velocity field with lat and lon determined by a conversion factor."""
    lon = np.linspace(-1e5 * mesh_conversion, 1e5 * mesh_conversion, xdim, dtype=np.float32)
    lat = np.linspace(-1e5 * mesh_conversion, 1e5 * mesh_conversion, ydim, dtype=np.float32)
    dimensions = {"lon": lon, "lat": lat}
    data = {"U": np.zeros((ydim, xdim), dtype=np.float32), "V": np.zeros((ydim, xdim), dtype=np.float32)}
    return FieldSet.from_data(data, dimensions, mesh=mesh)


def create_simple_pset(n=1):
    zeros = np.zeros(n)
    return parcels.ParticleSet(
        fieldset=create_fieldset_unit_mesh(),
        pclass=parcels.ScipyParticle,
        lon=zeros,
        lat=zeros,
        depth=zeros,
        time=zeros,
    )


def create_spherical_positions(n_particles, max_depth=100000):
    yrange = 2 * np.random.rand(n_particles)
    lat = 180 * (np.arccos(1 - yrange) - 0.5 * np.pi) / np.pi
    lon = 360 * np.random.rand(n_particles)
    depth = max_depth * np.random.rand(n_particles)
    return np.array((depth, lat, lon))


def create_flat_positions(n_particle):
    return np.random.rand(n_particle * 3).reshape(3, n_particle)


def create_fieldset_zeros_simple(xdim=40, ydim=100, withtime=False):
    lon = np.linspace(0, 1, xdim, dtype=np.float32)
    lat = np.linspace(-60, 60, ydim, dtype=np.float32)
    depth = np.zeros(1, dtype=np.float32)
    dimensions = {"lat": lat, "lon": lon, "depth": depth}
    if withtime:
        tdim = 10
        time = np.linspace(0, 86400, tdim, dtype=np.float64)
        dimensions["time"] = time
        datadims = (tdim, ydim, xdim)
        allow_time_extrapolation = False
    else:
        datadims = (ydim, xdim)
        allow_time_extrapolation = True
    U = np.zeros(datadims, dtype=np.float32)
    V = np.zeros(datadims, dtype=np.float32)
    data = {"U": np.array(U, dtype=np.float32), "V": np.array(V, dtype=np.float32)}
    return FieldSet.from_data(data, dimensions, allow_time_extrapolation=allow_time_extrapolation)


def assert_empty_folder(path: Path):
    assert [p.name for p in path.iterdir()] == []


def round_and_hash_float_array(arr, decimals=6):
    arr = np.round(arr, decimals=decimals)

    # Adapted from https://cs.stackexchange.com/a/37965
    h = 1
    for f in arr:
        # Mimic Float.floatToIntBits: converts float to 4-byte binary, then interprets as int
        float_as_int = struct.unpack("!i", struct.pack("!f", f))[0]
        h = 31 * h + float_as_int

    # Mimic Java's HashMap hash transformation
    h ^= (h >> 20) ^ (h >> 12)
    return h ^ (h >> 7) ^ (h >> 4)

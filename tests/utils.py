"""General helper functions and utilies for test suite."""

import numpy as np

from parcels import FieldSet


def create_fieldset_unit_mesh(xdim=20, ydim=20, mesh="spherical", transpose=False) -> FieldSet:
    """Standard unit mesh fieldset."""
    lon = np.linspace(0.0, 1.0, xdim, dtype=np.float32)
    lat = np.linspace(0.0, 1.0, ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    data = {"U": np.array(U, dtype=np.float32), "V": np.array(V, dtype=np.float32)}
    dimensions = {"lat": lat, "lon": lon}
    return FieldSet.from_data(data, dimensions, mesh=mesh, transpose=transpose)


def create_zeros_fieldset(mesh="spherical", xdim=200, ydim=100, mesh_conversion=1) -> FieldSet:
    """Generates a zero velocity field."""
    lon = np.linspace(-1e5 * mesh_conversion, 1e5 * mesh_conversion, xdim, dtype=np.float32)
    lat = np.linspace(-1e5 * mesh_conversion, 1e5 * mesh_conversion, ydim, dtype=np.float32)

    dimensions = {"lon": lon, "lat": lat}
    data = {"U": np.zeros((ydim, xdim), dtype=np.float32), "V": np.zeros((ydim, xdim), dtype=np.float32)}
    return FieldSet.from_data(data, dimensions, mesh=mesh)


def create_spherical_positions(n_particles, max_depth=100000):
    yrange = 2 * np.random.rand(n_particles)
    lat = 180 * (np.arccos(1 - yrange) - 0.5 * np.pi) / np.pi
    lon = 360 * np.random.rand(n_particles)
    depth = max_depth * np.random.rand(n_particles)
    return np.array((depth, lat, lon))


def create_flat_positions(n_particle):
    return np.random.rand(n_particle * 3).reshape(3, n_particle)

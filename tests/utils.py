"""General helper functions and utilies for test suite."""

import numpy as np

from parcels import FieldSet


def create_fieldset_unit_mesh(xdim=20, ydim=20, mesh="spherical"):
    """Standard unit mesh fieldset."""
    lon = np.linspace(0.0, 1.0, xdim, dtype=np.float32)
    lat = np.linspace(0.0, 1.0, ydim, dtype=np.float32)
    U, V = np.meshgrid(lat, lon)
    data = {"U": np.array(U, dtype=np.float32), "V": np.array(V, dtype=np.float32)}
    dimensions = {"lat": lat, "lon": lon}
    return FieldSet.from_data(data, dimensions, mesh=mesh)

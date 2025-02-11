# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.

# TODO: Write some benchmarks for parcels


import numpy as np
from parcels import FieldSet, ParticleSet, AdvectionRK4, ScipyParticle
from datetime import timedelta


class Advection3D:
    """Benchmark running the Parcels brownian motion example."""

    def time_run_whole_example(self):
        """Flat 2D zonal flow that increases linearly with depth from 0 m/s to 1 m/s."""
        xdim = ydim = zdim = 2
        npart = 11
        dimensions = {
            "lon": np.linspace(0.0, 1e4, xdim, dtype=np.float32),
            "lat": np.linspace(0.0, 1e4, ydim, dtype=np.float32),
            "depth": np.linspace(0.0, 1.0, zdim, dtype=np.float32),
        }
        data = {
            "U": np.ones((xdim, ydim, zdim), dtype=np.float32),
            "V": np.zeros((xdim, ydim, zdim), dtype=np.float32),
        }
        data["U"][:, :, 0] = 0.0
        fieldset = FieldSet.from_data(data, dimensions, mesh="flat", transpose=True)

        pset = ParticleSet(
            fieldset,
            pclass=ScipyParticle,
            lon=np.zeros(npart),
            lat=np.zeros(npart) + 1e2,
            depth=np.linspace(0, 1, npart),
        )
        pset.execute(AdvectionRK4, runtime=timedelta(hours=2), dt=timedelta(seconds=30))
        assert np.allclose(pset.depth * pset.time, pset.lon, atol=1.0e-1)


class ExampleTimeSuite:
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """

    def setup(self):
        self.d = {}
        for x in range(500):
            self.d[x] = None

    def time_keys(self):
        for _ in self.d.keys():
            pass

    def time_values(self):
        for _ in self.d.values():
            pass

    def time_range(self):
        d = self.d
        for key in range(500):
            _ = d[key]


class ExampleMemSuite:
    def mem_list(self):
        return [0] * 256

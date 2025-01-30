import numpy as np
import pytest
import xarray as xr

from parcels import AdvectionRK4_3D, FieldSet, JITParticle, ParticleSet, ScipyParticle


def create_interpolation_data():
    """Reference data used for testing interpolation.

    Most interpolation will be focussed around index
    (depth, lat, lon) = (zi, yi, xi) = (1, 1, 1) with ti=0.
    """
    z0 = np.array(  # each x is +1 from the previous, each y is +2 from the previous
        [
            [0.0, 1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0, 7.0],
            [6.0, 7.0, 8.0, 9.0],
        ]
    )
    spatial_data = [z0, z0 + 3, z0 + 6, z0 + 9]  # each z is +3 from the previous
    return xr.DataArray([spatial_data, spatial_data, spatial_data], dims=("time", "depth", "lat", "lon"))


def create_interpolation_data_with_land():
    np.random.seed(0)  # Ensure reproducibility
    tdim, zdim, ydim, xdim = 20, 5, 10, 10
    ds = xr.Dataset(
        {
            "U": (("time", "depth", "lat", "lon"), np.random.random((tdim, zdim, ydim, xdim))),
            "V": (("time", "depth", "lat", "lon"), np.random.random((tdim, zdim, ydim, xdim))),
            "W": (("time", "depth", "lat", "lon"), np.random.random((tdim, zdim, ydim, xdim))),
        },
        coords={
            "time": np.linspace(0, tdim - 1, tdim),
            "depth": np.linspace(0, 1, zdim),
            "lat": np.linspace(0, 1, ydim),
            "lon": np.linspace(0, 1, xdim),
        },
    )
    # Set a land point (for testing freeslip)
    ds["U"][:, :, 2, 5] = 0.0
    ds["V"][:, :, 2, 5] = 0.0
    ds["W"][:, :, 2, 5] = 0.0

    return ds


@pytest.mark.parametrize(
    "interp_method",
    [
        "linear",
        "freeslip",
        "nearest",
        "cgrid_velocity",
    ],
)
def test_scipy_vs_jit(interp_method):
    """Test that the scipy and JIT versions of the interpolation are the same."""
    variables = {"U": "U", "V": "V", "W": "W"}
    dimensions = {"time": "time", "lon": "lon", "lat": "lat", "depth": "depth"}
    fieldset = FieldSet.from_xarray_dataset(create_interpolation_data_with_land(), variables, dimensions, mesh="flat")

    for field in [fieldset.U, fieldset.V, fieldset.W]:  # Set a land point (for testing freeslip)
        field.interp_method = interp_method

    x, y, z = np.meshgrid(np.linspace(0, 1, 7), np.linspace(0, 1, 13), np.linspace(0, 1, 5))

    TestP = ScipyParticle.add_variable("pid", dtype=np.int32, initial=0)
    pset_scipy = ParticleSet(fieldset, pclass=TestP, lon=x, lat=y, depth=z, pid=np.arange(x.size))
    pset_jit = ParticleSet(fieldset, pclass=JITParticle, lon=x, lat=y, depth=z)

    def DeleteParticle(particle, fieldset, time):
        if particle.state >= 50:
            particle.delete()

    for pset in [pset_scipy, pset_jit]:
        pset.execute([AdvectionRK4_3D, DeleteParticle], runtime=4e-3, dt=1e-3)

    tol = 1e-6
    count = 0
    for i in range(len(pset_scipy)):
        # Check that the Scipy and JIT particles are at the same location
        assert np.isclose(pset_scipy[i].lon, pset_jit[i].lon, atol=tol)
        assert np.isclose(pset_scipy[i].lat, pset_jit[i].lat, atol=tol)
        assert np.isclose(pset_scipy[i].depth, pset_jit[i].depth, atol=tol)
        # Check that the Scipy and JIT particles have moved
        assert not np.isclose(pset_scipy[i].lon, x.flatten()[pset_scipy.pid[i]], atol=tol)
        assert not np.isclose(pset_scipy[i].lat, y.flatten()[pset_scipy.pid[i]], atol=tol)

        if np.isclose(pset_scipy[i].depth, z.flatten()[pset_scipy.pid[i]], atol=tol):
            count += 1
    print(f"{count}/{len(pset_scipy)} particles are at the same depth as the initial condition.")

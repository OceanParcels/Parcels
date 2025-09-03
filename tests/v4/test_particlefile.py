import os
import tempfile
from datetime import timedelta

import numpy as np
import pytest
import xarray as xr
from zarr.storage import MemoryStore

import parcels
from parcels import AdvectionRK4, Field, FieldSet, Particle, ParticleSet, Variable, VectorField
from parcels._core.utils.time import TimeInterval
from parcels._datasets.structured.generic import datasets
from parcels.particle import Particle, create_particle_data
from parcels.particlefile import ParticleFile
from parcels.xgrid import XGrid
from tests.common_kernels import DoNothing


@pytest.fixture
def fieldset() -> FieldSet:  # TODO v4: Move into a `conftest.py` file and remove duplicates
    """Fixture to create a FieldSet object for testing."""
    ds = datasets["ds_2d_left"]
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U (A grid)"], grid)
    V = Field("V", ds["V (A grid)"], grid)
    UV = VectorField("UV", U, V)

    return FieldSet(
        [U, V, UV],
    )


@pytest.mark.skip
def test_metadata(fieldset, tmp_zarrfile):
    pset = ParticleSet(fieldset, pclass=Particle, lon=0, lat=0)

    pset.execute(DoNothing, runtime=1, output_file=pset.ParticleFile(tmp_zarrfile, outputdt=np.timedelta64(1, "s")))

    ds = xr.open_zarr(tmp_zarrfile)
    assert ds.attrs["parcels_kernels"].lower() == "ParticleDoNothing".lower()


def test_pfile_array_write_zarr_memorystore(fieldset):
    """Check that writing to a Zarr MemoryStore works."""
    npart = 10
    zarr_store = MemoryStore()
    pset = ParticleSet(
        fieldset,
        pclass=Particle,
        lon=np.linspace(0, 1, npart),
        lat=0.5 * np.ones(npart),
        time=fieldset.time_interval.left,
    )
    pfile = pset.ParticleFile(zarr_store, outputdt=np.timedelta64(1, "s"))
    pfile.write(pset, time=fieldset.time_interval.left)

    ds = xr.open_zarr(zarr_store)
    assert ds.sizes["trajectory"] == npart


def test_pfile_array_remove_particles(fieldset, tmp_zarrfile):
    npart = 10
    pset = ParticleSet(
        fieldset,
        pclass=Particle,
        lon=np.linspace(0, 1, npart),
        lat=0.5 * np.ones(npart),
        time=fieldset.time_interval.left,
    )
    pfile = pset.ParticleFile(tmp_zarrfile, outputdt=np.timedelta64(1, "s"))
    pfile.write(pset, time=fieldset.time_interval.left)
    pset.remove_indices(3)
    for p in pset:
        p.time = 1
    pfile.write(pset, 1)

    ds = xr.open_zarr(tmp_zarrfile)
    timearr = ds["time"][:]
    assert (np.isnat(timearr[3, 1])) and (np.isfinite(timearr[3, 0]))


@pytest.mark.parametrize("chunks_obs", [1, None])
def test_pfile_array_remove_all_particles(fieldset, chunks_obs, tmp_zarrfile):
    npart = 10
    pset = ParticleSet(
        fieldset,
        pclass=Particle,
        lon=np.linspace(0, 1, npart),
        lat=0.5 * np.ones(npart),
        time=fieldset.time_interval.left,
    )
    chunks = (npart, chunks_obs) if chunks_obs else None
    pfile = pset.ParticleFile(tmp_zarrfile, chunks=chunks, outputdt=np.timedelta64(1, "s"))
    pfile.write(pset, time=fieldset.time_interval.left)
    for _ in range(npart):
        pset.remove_indices(-1)
    pfile.write(pset, 1)
    pfile.write(pset, 2)

    ds = xr.open_zarr(tmp_zarrfile).load()
    assert np.allclose(ds["time"][:, 0], np.timedelta64(0, "s"), atol=np.timedelta64(1, "ms"))
    if chunks_obs is not None:
        assert ds["time"][:].shape == chunks
    else:
        assert ds["time"][:].shape[0] == npart
        assert np.all(np.isnan(ds["time"][:, 1:]))


@pytest.mark.xfail(reason="lonlatdepth_dtype removed. Update implementation to use a different particle")
def test_variable_write_double(fieldset, tmp_zarrfile):
    def Update_lon(particle, fieldset, time):  # pragma: no cover
        particle.dlon += 0.1

    pset = ParticleSet(fieldset, pclass=Particle, lon=[0], lat=[0], lonlatdepth_dtype=np.float64)
    ofile = pset.ParticleFile(tmp_zarrfile, outputdt=np.timedelta64(10, "us"))
    pset.execute(pset.Kernel(Update_lon), endtime=0.001, dt=0.00001, output_file=ofile)

    ds = xr.open_zarr(tmp_zarrfile)
    lons = ds["lon"][:]
    assert isinstance(lons.values[0, 0], np.float64)


def test_write_dtypes_pfile(fieldset, tmp_zarrfile):
    dtypes = [
        np.float32,
        np.float64,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.bool_,
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
    ]

    extra_vars = [Variable(f"v_{d.__name__}", dtype=d, initial=0.0) for d in dtypes]
    MyParticle = Particle.add_variable(extra_vars)

    pset = ParticleSet(fieldset, pclass=MyParticle, lon=0, lat=0, time=fieldset.time_interval.left)
    pfile = pset.ParticleFile(tmp_zarrfile, outputdt=np.timedelta64(1, "s"))
    pfile.write(pset, time=fieldset.time_interval.left)

    ds = xr.open_zarr(
        tmp_zarrfile, mask_and_scale=False
    )  # Note masking issue at https://stackoverflow.com/questions/68460507/xarray-loading-int-data-as-float
    for d in dtypes:
        assert ds[f"v_{d.__name__}"].dtype == d


def test_variable_written_once():
    # Test that a vaiable is only written once. This should also work with gradual particle release (so the written once time is actually after the release of the particle)
    ...


@pytest.mark.parametrize("dt", [-1, 1])
@pytest.mark.parametrize("maxvar", [2, 4, 10])
def test_pset_repeated_release_delayed_adding_deleting(fieldset, tmp_zarrfile, dt, maxvar):
    runtime = 10
    fieldset.maxvar = maxvar
    pset = None

    MyParticle = Particle.add_variable(
        [Variable("sample_var", initial=0.0), Variable("v_once", dtype=np.float64, initial=0.0, to_write="once")]
    )

    pset = ParticleSet(
        fieldset, lon=np.zeros(runtime), lat=np.zeros(runtime), pclass=MyParticle, time=list(range(runtime))
    )
    pfile = pset.ParticleFile(tmp_zarrfile, outputdt=abs(dt), chunks=(1, 1))

    def IncrLon(particle, fieldset, time):  # pragma: no cover
        particle.sample_var += 1.0
        if particle.sample_var > fieldset.maxvar:
            particle.delete()

    for _ in range(runtime):
        pset.execute(IncrLon, dt=dt, runtime=1.0, output_file=pfile)

    ds = xr.open_zarr(tmp_zarrfile)
    samplevar = ds["sample_var"][:]
    assert samplevar.shape == (runtime, min(maxvar + 1, runtime))
    # test whether samplevar[:, k] = k
    for k in range(samplevar.shape[1]):
        assert np.allclose([p for p in samplevar[:, k] if np.isfinite(p)], k + 1)
    filesize = os.path.getsize(str(tmp_zarrfile))
    assert filesize < 1024 * 65  # test that chunking leads to filesize less than 65KB


def test_write_timebackward(fieldset, tmp_zarrfile):
    def Update_lon(particle, fieldset, time):  # pragma: no cover
        dt = particle.dt / np.timedelta64(1, "s")
        particle.dlon -= 0.1 * dt

    pset = ParticleSet(
        fieldset,
        pclass=Particle,
        lat=np.linspace(0, 1, 3),
        lon=[0, 0, 0],
        time=np.array([np.datetime64("2000-01-01") for _ in range(3)]),
    )
    pfile = pset.ParticleFile(tmp_zarrfile, outputdt=np.timedelta64(1, "s"))
    pset.execute(pset.Kernel(Update_lon), runtime=np.timedelta64(1, "s"), dt=-np.timedelta64(1, "s"), output_file=pfile)
    ds = xr.open_zarr(tmp_zarrfile)
    trajs = ds["trajectory"][:]
    assert trajs.values.dtype == "int64"
    assert np.all(np.diff(trajs.values) < 0)  # all particles written in order of release


def test_write_xiyi(fieldset, tmp_zarrfile):
    fieldset.U.data[:] = 1  # set a non-zero zonal velocity
    fieldset.add_field(Field(name="P", data=np.zeros((3, 20)), lon=np.linspace(0, 1, 20), lat=[-2, 0, 2]))
    dt = 3600

    XiYiParticle = Particle.add_variable(
        [
            Variable("pxi0", dtype=np.int32, initial=0.0),
            Variable("pxi1", dtype=np.int32, initial=0.0),
            Variable("pyi", dtype=np.int32, initial=0.0),
        ]
    )

    def Get_XiYi(particle, fieldset, time):  # pragma: no cover
        """Kernel to sample the grid indices of the particle.
        Note that this sampling should be done _before_ the advection kernel
        and that the first outputted value is zero.
        Be careful when using multiple grids, as the index may be different for the grids.
        """
        particle.pxi0 = fieldset.U.unravel_index(particle.ei)[2]
        particle.pxi1 = fieldset.P.unravel_index(particle.ei)[2]
        particle.pyi = fieldset.U.unravel_index(particle.ei)[1]

    def SampleP(particle, fieldset, time):  # pragma: no cover
        if time > 5 * 3600:
            _ = fieldset.P[particle]  # To trigger sampling of the P field

    pset = ParticleSet(fieldset, pclass=XiYiParticle, lon=[0, 0.2], lat=[0.2, 1], lonlatdepth_dtype=np.float64)
    pfile = pset.ParticleFile(tmp_zarrfile, outputdt=dt)
    pset.execute([SampleP, Get_XiYi, AdvectionRK4], endtime=10 * dt, dt=dt, output_file=pfile)

    ds = xr.open_zarr(tmp_zarrfile)
    pxi0 = ds["pxi0"][:].values.astype(np.int32)
    pxi1 = ds["pxi1"][:].values.astype(np.int32)
    lons = ds["lon"][:].values
    pyi = ds["pyi"][:].values.astype(np.int32)
    lats = ds["lat"][:].values

    for p in range(pyi.shape[0]):
        assert (pxi0[p, 0] == 0) and (pxi0[p, -1] == pset[p].pxi0)  # check that particle has moved
        assert np.all(pxi1[p, :6] == 0)  # check that particle has not been sampled on grid 1 until time 6
        assert np.all(pxi1[p, 6:] > 0)  # check that particle has not been sampled on grid 1 after time 6
        for xi, lon in zip(pxi0[p, 1:], lons[p, 1:], strict=True):
            assert fieldset.U.grid.lon[xi] <= lon < fieldset.U.grid.lon[xi + 1]
        for xi, lon in zip(pxi1[p, 6:], lons[p, 6:], strict=True):
            assert fieldset.P.grid.lon[xi] <= lon < fieldset.P.grid.lon[xi + 1]
        for yi, lat in zip(pyi[p, 1:], lats[p, 1:], strict=True):
            assert fieldset.U.grid.lat[yi] <= lat < fieldset.U.grid.lat[yi + 1]


def test_reset_dt(fieldset, tmp_zarrfile):
    # Assert that p.dt gets reset when a write_time is not a multiple of dt
    # for p.dt=0.02 to reach outputdt=0.05 and endtime=0.1, the steps should be [0.2, 0.2, 0.1, 0.2, 0.2, 0.1], resulting in 6 kernel executions

    def Update_lon(particle, fieldset, time):  # pragma: no cover
        particle.dlon += 0.1

    pset = ParticleSet(fieldset, pclass=Particle, lon=[0], lat=[0], lonlatdepth_dtype=np.float64)
    ofile = pset.ParticleFile(tmp_zarrfile, outputdt=np.timedelta64(50, "ms"))
    pset.execute(pset.Kernel(Update_lon), endtime=0.12, dt=0.02, output_file=ofile)

    assert np.allclose(pset.lon, 0.6)


def test_correct_misaligned_outputdt_dt(fieldset, tmp_zarrfile):
    """Testing that outputdt does not need to be a multiple of dt."""

    def Update_lon(particle, fieldset, time):  # pragma: no cover
        particle.dlon += particle.dt

    pset = ParticleSet(fieldset, pclass=Particle, lon=[0], lat=[0], lonlatdepth_dtype=np.float64)
    ofile = pset.ParticleFile(tmp_zarrfile, outputdt=np.timedelta64(3, "s"))
    pset.execute(pset.Kernel(Update_lon), endtime=11, dt=2, output_file=ofile)

    ds = xr.open_zarr(tmp_zarrfile)
    assert np.allclose(ds.lon.values, [0, 3, 6, 9])
    assert np.allclose(
        ds.time.values[0, :], [np.timedelta64(t, "s") for t in [0, 3, 6, 9]], atol=np.timedelta64(1, "ns")
    )


def setup_pset_execute(*, fieldset: FieldSet, outputdt: timedelta, execute_kwargs, particle_class=Particle):
    npart = 10

    pset = ParticleSet(
        fieldset,
        pclass=particle_class,
        lon=np.full(npart, fieldset.U.lon.mean()),
        lat=np.full(npart, fieldset.U.lat.mean()),
    )

    with tempfile.TemporaryDirectory() as dir:
        name = f"{dir}/test.zarr"
        output_file = pset.ParticleFile(name, outputdt=outputdt)

        pset.execute(DoNothing, output_file=output_file, **execute_kwargs)
        ds = xr.open_zarr(name).load()

    return ds


def test_pset_execute_outputdt_forwards(fieldset):
    """Testing output data dt matches outputdt in forward time."""
    outputdt = timedelta(hours=1)
    runtime = timedelta(hours=5)
    dt = timedelta(minutes=5)

    ds = setup_pset_execute(fieldset=fieldset, outputdt=outputdt, execute_kwargs=dict(runtime=runtime, dt=dt))

    assert np.all(ds.isel(trajectory=0).time.diff(dim="obs").values == np.timedelta64(outputdt))


def test_pset_execute_outputdt_backwards(fieldset):
    """Testing output data dt matches outputdt in backwards time."""
    outputdt = timedelta(hours=1)
    runtime = timedelta(days=2)
    dt = -timedelta(minutes=5)

    ds = setup_pset_execute(fieldset=fieldset, outputdt=outputdt, execute_kwargs=dict(runtime=runtime, dt=dt))
    file_outputdt = ds.isel(trajectory=0).time.diff(dim="obs").values
    assert np.all(file_outputdt == np.timedelta64(-outputdt))


@pytest.mark.xfail(reason="TODO v4: Update dataset loading")
def test_pset_execute_outputdt_backwards_fieldset_timevarying():
    """test_pset_execute_outputdt_backwards() still passed despite #1722 as it doesn't account for time-varying fields,
    which for some reason #1722
    """
    outputdt = timedelta(hours=1)
    runtime = timedelta(days=2)
    dt = -timedelta(minutes=5)

    # TODO: Not ideal using the `download_example_dataset` here, but I'm struggling to recreate this error using the test suite fieldsets we have
    example_dataset_folder = parcels.download_example_dataset("MovingEddies_data")
    filenames = {
        "U": str(example_dataset_folder / "moving_eddiesU.nc"),
        "V": str(example_dataset_folder / "moving_eddiesV.nc"),
    }
    variables = {"U": "vozocrtx", "V": "vomecrty"}
    dimensions = {"lon": "nav_lon", "lat": "nav_lat", "time": "time_counter"}
    fieldset = parcels.FieldSet.from_netcdf(filenames, variables, dimensions)

    ds = setup_pset_execute(outputdt=outputdt, execute_kwargs=dict(runtime=runtime, dt=dt), fieldset=fieldset)
    file_outputdt = ds.isel(trajectory=0).time.diff(dim="obs").values
    assert np.all(file_outputdt == np.timedelta64(-outputdt)), (file_outputdt, np.timedelta64(-outputdt))


@pytest.fixture
def store():
    return MemoryStore()


@pytest.mark.new
def test_particlefile_init(store):
    ParticleFile(store, outputdt=np.timedelta64(1, "s"), chunks=(1, 3))


@pytest.mark.new
def test_particlefile_init_invalid(store):  # TODO: Add test for read only store
    with pytest.raises(ValueError, match="chunks must be a tuple"):
        ParticleFile(store, outputdt=np.timedelta64(1, "s"), chunks=1)


@pytest.mark.new
def test_particlefile_write_particle_data(store):
    nparticles = 100

    pfile = ParticleFile(store, outputdt=np.timedelta64(1, "s"), chunks=(nparticles, 40))
    pclass = Particle

    left, right = np.datetime64("2019-05-30T12:00:00.000000000", "ns"), np.datetime64("2020-01-02", "ns")
    time_interval = TimeInterval(left=left, right=right)

    initial_lon = np.linspace(0, 1, nparticles)
    data = create_particle_data(
        pclass=pclass,
        nparticles=nparticles,
        ngrids=4,
        time_interval=time_interval,
        initial={
            "time": np.full(nparticles, fill_value=left),
            "lon": initial_lon,
            "dt": np.full(nparticles, fill_value=1.0),
            "trajectory": np.arange(nparticles),
        },
    )
    np.testing.assert_array_equal(data["time"], left)
    pfile._write_particle_data(
        particle_data=data,
        pclass=pclass,
        time_interval=time_interval,
        time=left,
    )
    ds = xr.open_zarr(store, decode_cf=False)  # TODO: Fix metadata and re-enable decode_cf
    # assert ds.time.dtype == "datetime64[ns]"
    # np.testing.assert_equal(ds["time"].isel(obs=0).values, left)
    assert ds.sizes["trajectory"] == nparticles
    np.testing.assert_allclose(ds["lon"].isel(obs=0).values, initial_lon)

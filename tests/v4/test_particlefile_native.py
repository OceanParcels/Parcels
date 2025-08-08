import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr

from parcels.particle import Variable
from parcels.particlefile_native import (
    bump_array_size_by_chunksize,
    initialize_zarr_dataset,
    write_particle_data,
)


@pytest.fixture
def store():
    return zarr.MemoryStore()


@pytest.mark.parametrize(
    "variables",
    [
        pytest.param(
            [
                Variable(name="a", dtype=np.float32),
                Variable(name="b", dtype=np.float64),
            ],
            id="simple",
        ),
        pytest.param(
            [
                Variable(name="temperature", dtype=np.float32, attrs={"units": "degrees_Celsius"}),
                Variable(name="time", dtype=np.float64, attrs={"units": "seconds"}),
            ],
            id="with attrs",
        ),
    ],
)
def test_intialize_zarr_dataset(store, variables):
    nparticles = 10
    obs_chunksize = 5

    root = initialize_zarr_dataset(
        store, n_particles=nparticles, chunks=(nparticles, obs_chunksize), variables=variables
    )
    assert len(root.keys()) == len(variables)

    for variable in variables:
        arr = root[variable.name]
        assert arr.dtype == variable.dtype
        assert arr.shape == (nparticles, obs_chunksize)


@pytest.mark.parametrize("chunksize", [(10, 10), (5, 20), (1, 100)])
def test_bump_array_size_by_chunksize(store, chunksize):
    root = zarr.group(store=store)
    arr = root.create_dataset(
        "test",
        shape=(10, 10),
        chunks=chunksize,
        dtype=np.float32,
        fill_value=np.nan,
    )
    assert arr.shape == (10, 10)

    bump_array_size_by_chunksize(arr, axis=1)
    assert arr.shape == (10, 10 + chunksize[1])

    bump_array_size_by_chunksize(arr, axis=0)
    assert arr.shape == (10 + chunksize[0], 10 + chunksize[1])

    bump_array_size_by_chunksize(arr, axis=0)
    assert arr.shape == (10 + 2 * chunksize[0], 10 + chunksize[1])


@pytest.mark.parametrize("consolidate", [True, False])
def test_bump_array_size_by_chunksize_consolidate(store, consolidate):
    root = zarr.group(store=store)
    arr = root.create_dataset(
        "test",
        shape=(10, 10),
        chunks=(5, 5),
        dtype=np.float32,
        fill_value=np.nan,
    )
    bump_array_size_by_chunksize(arr, axis=1, consolidate=consolidate)


@pytest.mark.parametrize(
    "variables",
    [
        pytest.param(
            [
                Variable(name="temperature", dtype=np.float32, attrs={"units": "degrees_Celsius"}),
                Variable(name="time", dtype=np.float64, attrs={"units": "seconds"}),
            ],
            id="with attrs",
        )
    ],
)
def test_write_particle_data(store, variables):
    nparticles = 10
    obs_chunksize = 5

    root = initialize_zarr_dataset(
        store, n_particles=nparticles, chunks=(nparticles, obs_chunksize), variables=variables
    )
    particle_data = pd.DataFrame(
        {
            "obs_written": np.zeros(nparticles, dtype=np.int32),
            "trajectory": np.arange(nparticles, dtype=np.int32),
            "time": np.zeros(nparticles, dtype=np.float64),
            "temperature": np.random.uniform(15, 35, nparticles),
        }
    )

    for expected_obs_written in range(1, 2 * obs_chunksize + 1):
        write_particle_data(root, particle_data, np.full(nparticles, True, dtype=bool))

        obs_written_temperature = (~np.isnan(root.temperature)).all(axis=0)
        obs_written_time = (~np.isnan(root.time)).all(axis=0)
        np.testing.assert_equal(obs_written_temperature, obs_written_time)
        np.testing.assert_equal(obs_written_time[:expected_obs_written], True)
        np.testing.assert_equal(obs_written_time[expected_obs_written:], False)


def get_particle_data(nparticles: int | None = None, prior: pd.DataFrame | None = None) -> pd.DataFrame:
    if prior is None:
        assert nparticles is not None, "nparticles must be specified when no prior is provided"
        df = pd.DataFrame(
            {
                "trajectory": np.arange(nparticles).astype(np.int32),
                "obs_written": np.random.randn(nparticles).astype(np.int32),
                "time": np.zeros(nparticles, dtype=np.float64),
                "temperature": np.random.uniform(15, 35, nparticles),
            }
        )
    else:
        assert nparticles is None, "Cannot specify nparticles when prior is provided"
        df = prior
    df["temperature"] = df["temperature"] + np.sin(df["time"] / 5)
    df["time"] = df["time"] + 1
    return df


if __name__ == "__main__":
    nparticles = 100

    df = get_particle_data(nparticles)

    variables = [
        Variable("temperature", np.float32, attrs={"units": "degrees_Celsius"}),
        Variable("time", np.float64, attrs={"units": "seconds"}),
    ]

    obs_chunksize = 200

    zarr_store = zarr.MemoryStore()
    root = initialize_zarr_dataset(
        zarr_store, n_particles=nparticles, chunks=(nparticles, obs_chunksize), variables=variables
    )

    for _ in range(100):
        write_particle_data(root, df, np.full_like(df["trajectory"], True, dtype=bool))
        get_particle_data(prior=df)

    ds = xr.open_zarr(zarr_store, decode_cf=False)

    values = np.array(root.temperature)

    for array in root:
        print(root[array])

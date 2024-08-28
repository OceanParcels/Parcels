import os
import unittest.mock

import numpy as np
import pytest
import requests

from parcels import (
    AdvectionRK4_3D,
    AdvectionRK45,
    FieldSet,
    FieldSetWarning,
    KernelWarning,
    ParticleSet,
    ScipyParticle,
    download_example_dataset,
    list_example_datasets,
)


@pytest.mark.skip(reason="too time intensive")
def test_download_example_dataset(tmp_path):
    # test valid datasets
    for dataset in list_example_datasets():
        dataset_folder_path = download_example_dataset(dataset, data_home=tmp_path)

        assert dataset_folder_path.exists()
        assert dataset_folder_path.name == dataset
        assert str(dataset_folder_path.parent) == str(tmp_path)

    # test non-existing dataset
    with pytest.raises(ValueError):
        download_example_dataset("non_existing_dataset", data_home=tmp_path)


def test_download_example_dataset_lite(tmp_path):
    # test valid datasets
    # avoids downloading the dataset (only verifying that the URL is responsive, and folders are created)
    with unittest.mock.patch("urllib.request.urlretrieve", new=mock_urlretrieve) as mock_function:  # noqa: F841
        for dataset in list_example_datasets()[0:1]:
            dataset_folder_path = download_example_dataset(dataset, data_home=tmp_path)

            assert dataset_folder_path.exists()
            assert dataset_folder_path.name == dataset
            assert str(dataset_folder_path.parent) == str(tmp_path)

    # test non-existing dataset
    with pytest.raises(ValueError):
        download_example_dataset("non_existing_dataset", data_home=tmp_path)


def test_download_example_dataset_no_data_home():
    # This test depends on your default data_home location and whether
    # it's okay to download files there. Be careful with this test in a CI environment.
    dataset = list_example_datasets()[0]
    dataset_folder_path = download_example_dataset(dataset)
    assert dataset_folder_path.exists()
    assert dataset_folder_path.name == dataset


def mock_urlretrieve(url, filename):
    # send a HEAD request to the URL
    response = requests.head(url)

    # check the status code of the response
    if 400 <= response.status_code < 600:
        raise Exception(f"Failed to access URL: {url}. Status code: {response.status_code}")

    print(f"Pinged URL successfully: {url}")


def test_fieldset_warnings():
    with pytest.warns(FieldSetWarning):
        # halo with inconsistent boundaries
        lat = [0, 1, 5, 10]
        lon = [0, 1, 5, 10]
        u = [[1, 1, 1, 1] for _ in range(4)]
        v = [[1, 1, 1, 1] for _ in range(4)]
        fieldset = FieldSet.from_data(data={"U": u, "V": v}, dimensions={"lon": lon, "lat": lat}, transpose=True)
        fieldset.add_periodic_halo(meridional=True, zonal=True)

    with pytest.warns(FieldSetWarning):
        # flipping warning
        lat = [0, 1, 5, -5]
        lon = [0, 1, 5, 10]
        u = [[1, 1, 1, 1] for _ in range(4)]
        v = [[1, 1, 1, 1] for _ in range(4)]
        fieldset = FieldSet.from_data(data={"U": u, "V": v}, dimensions={"lon": lon, "lat": lat}, transpose=True)

    with pytest.warns(FieldSetWarning):
        # allow_time_extrapolation with time_periodic warning
        fieldset = FieldSet.from_data(
            data={"U": u, "V": v},
            dimensions={"lon": lon, "lat": lat},
            transpose=True,
            allow_time_extrapolation=True,
            time_periodic=1,
        )

    with pytest.warns(FieldSetWarning):
        # b-grid with s-levels and POP output in meters
        mesh = os.path.join(os.path.join(os.path.dirname(__file__), "test_data"), "POPtestdata_time.nc")
        filenames = mesh
        variables = {"U": "U", "V": "V", "W": "W", "T": "T"}
        dimensions = {"lon": "lon", "lat": "lat", "depth": "w_deps", "time": "time"}
        fieldset = FieldSet.from_pop(filenames, variables, dimensions, mesh="flat")

    with pytest.warns(FieldSetWarning):
        # timestamps with time in file warning
        fieldset = FieldSet.from_pop(filenames, variables, dimensions, mesh="flat", timestamps=[0, 1, 2, 3])


def test_kernel_warnings():
    with pytest.warns(KernelWarning):
        # positive scaling factor for W
        mesh = os.path.join(os.path.join(os.path.dirname(__file__), "test_data"), "POPtestdata_time.nc")
        filenames = mesh
        variables = {"U": "U", "V": "V", "W": "W", "T": "T"}
        dimensions = {"lon": "lon", "lat": "lat", "depth": "w_deps", "time": "time"}
        fieldset = FieldSet.from_pop(filenames, variables, dimensions, mesh="flat")
        fieldset.W._scaling_factor = 0.01
        pset = ParticleSet(fieldset=fieldset, pclass=ScipyParticle, lon=[0], lat=[0], depth=[0], time=[0])
        pset.execute(AdvectionRK4_3D, runtime=1, dt=1)

    with pytest.warns(KernelWarning):
        # RK45 warnings
        lat = [0, 1, 5, 10]
        lon = [0, 1, 5, 10]
        u = [[1, 1, 1, 1] for _ in range(4)]
        v = [[1, 1, 1, 1] for _ in range(4)]
        fieldset = FieldSet.from_data(data={"U": u, "V": v}, dimensions={"lon": lon, "lat": lat}, transpose=True)
        pset = ParticleSet(
            fieldset=fieldset,
            pclass=ScipyParticle.add_variable("next_dt", dtype=np.float32, initial=1),
            lon=[0],
            lat=[0],
            depth=[0],
            time=[0],
            next_dt=1,
        )
        pset.execute(AdvectionRK45, runtime=1, dt=1)

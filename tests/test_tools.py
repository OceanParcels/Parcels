import unittest.mock

import pytest
import requests

from parcels import download_example_dataset, list_example_datasets


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
    with unittest.mock.patch('urllib.request.urlretrieve', new=mock_urlretrieve) as mock_function:  # noqa: F841
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

import pytest
import requests

from parcels._tutorial import (
    _get_pooch,
    download_example_dataset,
    list_example_datasets,
)


@pytest.mark.parametrize("url", [_get_pooch().get_url(filename) for filename in _get_pooch().registry.keys()])
def test_pooch_registry_url_reponse(url):
    response = requests.head(url)
    assert not (400 <= response.status_code < 600)


@pytest.mark.parametrize("dataset", list_example_datasets()[:1])
def test_download_example_dataset_folder_creation(tmp_path, dataset):
    dataset_folder_path = download_example_dataset(dataset, data_home=tmp_path)

    assert dataset_folder_path.exists()
    assert dataset_folder_path.name == dataset
    assert dataset_folder_path.parent == tmp_path


def test_download_non_existing_example_dataset(tmp_path):
    with pytest.raises(ValueError):
        download_example_dataset("non_existing_dataset", data_home=tmp_path)


def test_download_example_dataset_no_data_home():
    # This test depends on your default data_home location and whether
    # it's okay to download files there. Be careful with this test in a CI environment.
    dataset = list_example_datasets()[0]
    dataset_folder_path = download_example_dataset(dataset)
    assert dataset_folder_path.exists()
    assert dataset_folder_path.name == dataset

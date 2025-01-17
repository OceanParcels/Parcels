from pathlib import Path

import pytest
import requests

from parcels.tools.exampledata_utils import (
    download_example_dataset,
    list_example_datasets,
)


@pytest.fixture
def mock_download(monkeypatch):
    """Avoid the download, only check the status code and create empty file."""

    def mock_urlretrieve(url, filename):
        response = requests.head(url)

        if 400 <= response.status_code < 600:
            raise Exception(f"Failed to access URL: {url}. Status code: {response.status_code}")

        Path(filename).touch()

    monkeypatch.setattr("parcels.tools.exampledata_utils.urlretrieve", mock_urlretrieve)


@pytest.mark.usefixtures("mock_download")
@pytest.mark.parametrize("dataset", list_example_datasets())
def test_download_example_dataset(tmp_path, dataset):
    if dataset == "GlobCurrent_example_data":
        pytest.skip(f"{dataset} too time consuming.")

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

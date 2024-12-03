import pytest


@pytest.fixture()
def tmp_zarrfile(tmp_path, request):
    test_name = request.node.name
    yield tmp_path / f"{test_name}-output.zarr"

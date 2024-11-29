import pytest


@pytest.fixture()
def tmp_zarr(tmp_path, request):
    test_name = request.node.name
    yield tmp_path / f"{test_name}-output.zarr"

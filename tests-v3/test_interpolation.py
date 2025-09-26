import pytest

import parcels._interpolation as interpolation
from tests.utils import create_fieldset_zeros_3d


@pytest.fixture
def tmp_interpolator_registry():
    """Resets the interpolator registry after the test. Vital when testing manipulating the registry."""
    old_2d = interpolation._interpolator_registry_2d.copy()
    old_3d = interpolation._interpolator_registry_3d.copy()
    yield
    interpolation._interpolator_registry_2d = old_2d
    interpolation._interpolator_registry_3d = old_3d


@pytest.mark.usefixtures("tmp_interpolator_registry")
def test_interpolation_registry():
    @interpolation.register_3d_interpolator("test")
    @interpolation.register_2d_interpolator("test")
    def some_function():
        return "test"

    assert "test" in interpolation.get_2d_interpolator_registry()
    assert "test" in interpolation.get_3d_interpolator_registry()

    f = interpolation.get_2d_interpolator_registry()["test"]
    g = interpolation.get_3d_interpolator_registry()["test"]
    assert f() == g() == "test"


@pytest.mark.v4remove
@pytest.mark.xfail(reason="GH1946")
@pytest.mark.usefixtures("tmp_interpolator_registry")
def test_interpolator_override():
    fieldset = create_fieldset_zeros_3d()

    @interpolation.register_3d_interpolator("linear")
    def test_interpolator(ctx: interpolation.InterpolationContext3D):
        raise NotImplementedError

    with pytest.raises(NotImplementedError):
        fieldset.U[0, 0.5, 0.5, 0.5]


@pytest.mark.v4remove
@pytest.mark.xfail(reason="GH1946")
@pytest.mark.usefixtures("tmp_interpolator_registry")
def test_full_depth_provided_to_interpolators():
    """The full depth needs to be provided to the interpolation schemes as some interpolators
    need to know whether they are at the surface or bottom of the water column.

    https://github.com/OceanParcels/Parcels/pull/1816#discussion_r1908840408
    """
    xdim, ydim, zdim = 10, 11, 12
    fieldset = create_fieldset_zeros_3d(xdim=xdim, ydim=ydim, zdim=zdim)

    @interpolation.register_3d_interpolator("linear")
    def test_interpolator2(ctx: interpolation.InterpolationContext3D):
        assert ctx.data.shape[1] == zdim
        # The array z dimension is the same as the fieldset z dimension
        return 0

    fieldset.U[0.5, 0.5, 0.5, 0.5]

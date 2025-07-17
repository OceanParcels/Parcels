import numpy as np
import pytest

from parcels import (
    AdvectionRK4,
    Field,
    FieldSet,
    ParticleSet,
)
from parcels._datasets.structured.generic import datasets as datasets_structured
from parcels.xgrid import XGrid
from tests.common_kernels import MoveEast, MoveNorth


@pytest.fixture
def fieldset() -> FieldSet:
    ds = datasets_structured["ds_2d_left"]
    grid = XGrid.from_dataset(ds)
    U = Field("U", ds["U (A grid)"], grid, mesh_type="flat")
    V = Field("V", ds["V (A grid)"], grid, mesh_type="flat")
    return FieldSet([U, V])


def test_multi_kernel_reuse_varnames(fieldset):
    pset = ParticleSet(fieldset, lon=[0.5], lat=[0.5])

    # Testing for merging of two Kernels with the same variable declared
    def MoveEast1(particle, fieldset, time):  # pragma: no cover
        add_lon = 0.2
        particle_dlon += add_lon  # noqa

    def MoveEast2(particle, fieldset, time):  # pragma: no cover
        particle_dlon += add_lon  # noqa

    pset.execute([MoveEast1, MoveEast2], runtime=np.timedelta64(2, "s"))
    assert np.allclose(pset.lon, [0.9], atol=1e-5)  # should be 0.5 + 0.2 + 0.2 = 0.9


def test_unknown_var_in_kernel(fieldset):
    pset = ParticleSet(fieldset, lon=[0.5], lat=[0.5])

    def ErrorKernel(particle, fieldset, time):  # pragma: no cover
        particle.unknown_varname += 0.2

    with pytest.raises(KeyError, match="No variable named 'unknown_varname'"):
        pset.execute(ErrorKernel, runtime=np.timedelta64(2, "s"))


def test_combined_kernel_from_list(fieldset):
    """
    Test pset.Kernel(List[function])

    Tests that a Kernel can be created from a list functions, or a list of
    mixed functions and kernel objects.
    """
    pset = ParticleSet(fieldset, lon=[0.5], lat=[0.5])
    kernels_single = pset.Kernel([AdvectionRK4])
    kernels_functions = pset.Kernel([AdvectionRK4, MoveEast, MoveNorth])

    # Check if the kernels were combined correctly
    assert kernels_single.funcname == "AdvectionRK4"
    assert kernels_functions.funcname == "AdvectionRK4MoveEastMoveNorth"


def test_combined_kernel_from_list_error_checking(fieldset):
    """
    Test pset.Kernel(List[function])

    Tests that various error cases raise appropriate messages.
    """
    pset = ParticleSet(fieldset, lon=[0.5], lat=[0.5])

    # Test that list has to be non-empty
    with pytest.raises(ValueError):
        pset.Kernel([])

    # Test that list has to be all functions
    with pytest.raises(ValueError):
        pset.Kernel([AdvectionRK4, "something else"])

    # Can't mix kernel objects and functions in list
    with pytest.raises(ValueError):
        kernels_mixed = pset.Kernel([pset.Kernel(AdvectionRK4), MoveEast, MoveNorth])
        assert kernels_mixed.funcname == "AdvectionRK4MoveEastMoveNorth"

import numpy as np
import pytest

from parcels import (
    AdvectionRK4,
    Field,
    FieldSet,
    ParticleSet,
)
from parcels._datasets.structured.generic import datasets as datasets_structured
from parcels.kernel import Kernel
from parcels.particle import Particle
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
        particle.dlon += add_lon

    def MoveEast2(particle, fieldset, time):  # pragma: no cover
        particle.dlon += add_lon  # noqa

    pset.execute([MoveEast1, MoveEast2], runtime=np.timedelta64(2, "s"))
    assert np.allclose(pset.lon, [0.9], atol=1e-5)  # should be 0.5 + 0.2 + 0.2 = 0.9


def test_unknown_var_in_kernel(fieldset):
    pset = ParticleSet(fieldset, lon=[0.5], lat=[0.5])

    def ErrorKernel(particle, fieldset, time):  # pragma: no cover
        particle.unknown_varname += 0.2

    with pytest.raises(KeyError, match="'unknown_varname'"):
        pset.execute(ErrorKernel, runtime=np.timedelta64(2, "s"))


def test_kernel_init(fieldset):
    Kernel(fieldset, ptype=Particle, pyfuncs=[AdvectionRK4])


def test_kernel_merging(fieldset):
    k1 = Kernel(fieldset, ptype=Particle, pyfuncs=[AdvectionRK4])
    k2 = Kernel(fieldset, ptype=Particle, pyfuncs=[MoveEast, MoveNorth])

    merged_kernel = k1 + k2
    assert merged_kernel.funcname == "AdvectionRK4MoveEastMoveNorth"
    assert len(merged_kernel._pyfuncs) == 3
    assert merged_kernel._pyfuncs == [AdvectionRK4, MoveEast, MoveNorth]

    merged_kernel = k2 + k1
    assert merged_kernel.funcname == "MoveEastMoveNorthAdvectionRK4"
    assert len(merged_kernel._pyfuncs) == 3
    assert merged_kernel._pyfuncs == [MoveEast, MoveNorth, AdvectionRK4]


def test_kernel_from_list(fieldset):
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


def test_kernel_from_list_error_checking(fieldset):
    """
    Test pset.Kernel(List[function])

    Tests that various error cases raise appropriate messages.
    """
    pset = ParticleSet(fieldset, lon=[0.5], lat=[0.5])

    with pytest.raises(ValueError, match="List of `pyfuncs` should have at least one function."):
        pset.Kernel([])

    with pytest.raises(ValueError, match="Argument `pyfunc_list` should be a list of functions."):
        pset.Kernel([AdvectionRK4, "something else"])

    with pytest.raises(ValueError, match="Argument `pyfunc_list` should be a list of functions."):
        kernels_mixed = pset.Kernel([pset.Kernel(AdvectionRK4), MoveEast, MoveNorth])
        assert kernels_mixed.funcname == "AdvectionRK4MoveEastMoveNorth"

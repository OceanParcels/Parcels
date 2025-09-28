import numpy as np
import pytest

from parcels import (
    Field,
    FieldSet,
    ParticleSet,
)
from parcels._datasets.structured.generic import datasets as datasets_structured
from parcels import Kernel
from parcels.kernels import AdvectionRK4
from parcels import Particle
from parcels import XGrid
from tests.common_kernels import MoveEast, MoveNorth


@pytest.fixture
def fieldset() -> FieldSet:
    ds = datasets_structured["ds_2d_left"]
    grid = XGrid.from_dataset(ds, mesh="flat")
    U = Field("U", ds["U (A grid)"], grid)
    V = Field("V", ds["V (A grid)"], grid)
    return FieldSet([U, V])


def test_unknown_var_in_kernel(fieldset):
    pset = ParticleSet(fieldset, lon=[0.5], lat=[0.5])

    def ErrorKernel(particles, fieldset):  # pragma: no cover
        particles.unknown_varname += 0.2

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


def test_kernel_signature(fieldset):
    pset = ParticleSet(fieldset, lon=[0.5], lat=[0.5])

    def good_kernel(particles, fieldset):
        pass

    def version_3_kernel(particle, fieldset, time):
        pass

    def version_3_kernel_without_time(particle, fieldset):
        pass

    def kernel_switched_args(fieldset, particle):
        pass

    def kernel_with_forced_kwarg(particles, *, fieldset=0):
        pass

    pset.Kernel(good_kernel)

    with pytest.raises(ValueError, match="Kernel function must have 2 parameters, got 3"):
        pset.Kernel(version_3_kernel)

    with pytest.raises(
        ValueError, match="Parameter 'particle' has incorrect name. Expected 'particles', got 'particle'"
    ):
        pset.Kernel(version_3_kernel_without_time)

    with pytest.raises(
        ValueError, match="Parameter 'fieldset' has incorrect name. Expected 'particles', got 'fieldset'"
    ):
        pset.Kernel(kernel_switched_args)

    with pytest.raises(
        ValueError,
        match="Parameter 'fieldset' has incorrect parameter kind. Expected POSITIONAL_OR_KEYWORD, got KEYWORD_ONLY",
    ):
        pset.Kernel(kernel_with_forced_kwarg)

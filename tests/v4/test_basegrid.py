from __future__ import annotations

import itertools

import numpy as np
import pytest

from parcels.basegrid import BaseGrid


class MockGrid(BaseGrid):
    def __init__(self, axis_dim: dict[str, int]):
        self.axis_dim = axis_dim

    def search(self, z: float, y: float, x: float, ei=None) -> dict[str, tuple[int, float | np.ndarray]]:
        pass

    @property
    def axes(self) -> list[str]:
        return list(self.axis_dim.keys())

    def get_axis_dim(self, axis: str) -> int:
        return self.axis_dim[axis]


@pytest.mark.parametrize(
    "grid",
    [
        MockGrid({"Z": 10, "Y": 20, "X": 30}),
        MockGrid({"Z": 5, "Y": 15}),
        MockGrid({"Z": 8}),
        MockGrid({"Z": 12, "FACE": 25}),
    ],
)
def test_basegrid_ravel_unravel_index(grid):
    axes = grid.axes
    dimensionalities = (grid.get_axis_dim(axis) for axis in axes)
    all_possible_axis_indices = itertools.product(*[np.arange(dim)[:, np.newaxis] for dim in dimensionalities])

    encountered_eis = []

    for axis_indices_numeric in all_possible_axis_indices:
        axis_indices = dict(zip(axes, axis_indices_numeric, strict=True))

        ei = grid.ravel_index(axis_indices)
        axis_indices_test = grid.unravel_index(ei)
        assert axis_indices_test == axis_indices
        encountered_eis.append(ei[0])

    encountered_eis = sorted(encountered_eis)
    assert len(set(encountered_eis)) == len(encountered_eis), "Raveled indices are not unique."
    assert np.allclose(np.diff(np.array(encountered_eis)), 1), "Raveled indices are not consecutive integers."
    assert encountered_eis[0] == 0, "Raveled indices do not start at 0."

from numba.core.typing.asnumbatype import as_numba_type
from numba.experimental import jitclass

from parcels.numba.field.field import _base_field_spec, NumbaField
from parcels.numba.grid.curvilinear import CurvilinearZGrid, CurvilinearSGrid
from parcels.numba.grid.rectilinear import RectilinearZGrid, RectilinearSGrid
from parcels.numba.grid.base import GridCode

# Compile specific numba field classes, depending on the grid.
NumbaFieldCZG = jitclass(NumbaField, spec=_base_field_spec() + [
    ("grid", as_numba_type(CurvilinearZGrid))])
NumbaFieldCSG = jitclass(NumbaField, spec=_base_field_spec() + [
    ("grid", as_numba_type(CurvilinearSGrid))])
NumbaFieldRZG = jitclass(NumbaField, spec=_base_field_spec() + [
    ("grid", as_numba_type(RectilinearZGrid))])
NumbaFieldRSG = jitclass(NumbaField, spec=_base_field_spec() + [
    ("grid", as_numba_type(RectilinearSGrid))])


def _get_numba_field_class(grid):
    "Get compiled numba class from grid (which has a type)"
    if grid.gtype == GridCode.CurvilinearZGrid:
        return NumbaFieldCZG
    if grid.gtype == GridCode.CurvilinearSGrid:
        return NumbaFieldCSG
    if grid.gtype == GridCode.RectilinearZGrid:
        return NumbaFieldRZG
    if grid.gtype == GridCode.RectilinearSGrid:
        return NumbaFieldRSG

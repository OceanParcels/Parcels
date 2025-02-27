import importlib.util
import os
import sys
import warnings
from copy import deepcopy
from glob import glob

import dask.array as da
import numpy as np
import uxarray as ux

from parcels._compat import MPI
from parcels._typing import GridIndexingType, InterpMethodOption, Mesh
from parcels.field import DeferredArray, Field, NestedField, VectorField
from parcels.grid import Grid
from parcels.gridset import GridSet
from parcels.particlefile import ParticleFile
from parcels.tools._helpers import fieldset_repr
from parcels.tools.converters import TimeConverter, convert_xarray_time_units
from parcels.tools.loggers import logger
from parcels.tools.statuscodes import TimeExtrapolationError
from parcels.tools.warnings import FieldSetWarning

__all__ = ["UXFieldSet"]

class UXFieldSet:
    """A FieldSet class that holds hydrodynamic data needed to execute particles
    in a UXArray.Dataset"""

    def __init__(self, uxds: ux.UxDataset):

        # Ensure that dataset provides a grid, and the u and v velocity 
        # components at a minimum
        if not hasattr(uxds, "uxgrid"):
            raise ValueError("The UXArray dataset does not provide a grid")
        if not hasattr(uxds, "u"):
            raise ValueError("The UXArray dataset does not provide u velocity data")
        if not hasattr(uxds, "v"):
            raise ValueError("The UXArray dataset does not provide v velocity data")
        
        self.uxds = uxds

    def _check_complete(self):
        assert self.uxds is not None, "UXFieldSet has not been loaded"
        assert self.uxds.u is not None, "UXFieldSet does not provide u velocity data"
        assert self.uxds.v is not None, "UXFieldSet does not provide v velocity data"
        assert self.uxds.uxgrid is not None, "UXFieldSet does not provide a grid"

import importlib.util
import os
import sys
import warnings
from glob import glob

import numpy as np

from parcels._typing import GridIndexingType, InterpMethodOption, Mesh
from parcels.xfield import XField, XVectorField
from parcels.particlefile import ParticleFile
from parcels.tools._helpers import fieldset_repr, default_repr
from parcels.tools.converters import TimeConverter
from parcels.tools.warnings import FieldSetWarning

import xarray as xr
import uxarray as ux

__all__ = ["FieldSet"]


class XFieldSet:
    """XFieldSet class that holds hydrodynamic data needed to execute particles.
    
    Parameters
    ----------
    ds : xarray.Dataset | uxarray.UxDataset)
        xarray.Dataset and/or uxarray.UxDataset objects containing the field data.

    Notes
    -----
    The `ds` object is a xarray.Dataset or uxarray.UxDataset object. 
    In XArray terminology, the (Ux)Dataset holds multiple (Ux)DataArray objects. 
    Each (Ux)DataArray object is a single "field" that is associated with their own
    dimensions and coordinates within the (Ux)Dataset.

    A (Ux)Dataset object is associated with a single mesh, which can have multiple
    types of "points" (multiple "grids") (e.g. for UxDataSets, these are "face_lon", 
    "face_lat", "node_lon", "node_lat", "edge_lon", "edge_lat"). Each (Ux)DataArray is 
    registered to a specific set of points on the mesh.

    For UxDataset objects, each `UXDataArray.attributes` field dictionary contains 
    the necessary metadata to help determine which set of points a field is registered
    to and what parent model the field is associated with. Parcels uses this metadata
    during execution for interpolation.  Each `UXDataArray.attributes` field dictionary
    must have: 
      * "location" key set to "face", "node", or "edge" to define which pairing of points a field is associated with.
      * "mesh" key to define which parent model the fields are associated with (e.g. "fesom_mesh", "icon_mesh")

    """

    def __init__(self, ds: xr.Dataset | ux.UxDataset):
        self.ds = ds

        # Create pointers to each (Ux)DataArray
        for field in self.ds.data_vars:
            setattr(self, field, XField(field,self.ds[field]))

        self._add_UVfield()

    def add_vector_field(self, vfield):
        """Add a :class:`parcels.field.VectorField` object to the FieldSet.

        Parameters
        ----------
        vfield : parcels.XVectorField
            class:`parcels.xfieldset.XVectorField` object to be added
        """
        setattr(self, vfield.name, vfield)
        for v in vfield.__dict__.values():
            if isinstance(v, XField) and (v not in self.get_fields()):
                self.add_field(v)

    def get_fields(self) -> list[XField | XVectorField]:
        """Returns a list of all the :class:`parcels.field.Field` and :class:`parcels.field.VectorField`
        objects associated with this FieldSet.
        """
        fields = []
        for v in self.__dict__.values():
            if type(v) in [XField, XVectorField]:
                if v not in fields:
                    fields.append(v)
        return fields
    
    def _add_UVfield(self):
        if not hasattr(self, "UV") and hasattr(self, "u") and hasattr(self, "v"):
            self.add_xvector_field(XVectorField("UV", self.u, self.v))
        if not hasattr(self, "UVW") and hasattr(self, "w"):
            self.add_xvector_field(XVectorField("UVW", self.u, self.v, self.w))



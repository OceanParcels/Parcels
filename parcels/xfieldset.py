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

__all__ = ["XFieldSet"]


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

        self._completed: bool = False
        # Create pointers to each (Ux)DataArray
        for field in self.ds.data_vars:
            setattr(self, field, XField(field,self.ds[field]))

        self._add_UVfield()

    def __repr__(self):
        return fieldset_repr(self)
    
    # @property
    # def particlefile(self):
    #     return self._particlefile

    # @staticmethod
    # def checkvaliddimensionsdict(dims):
    #     for d in dims:
    #         if d not in ["lon", "lat", "depth", "time"]:
    #             raise NameError(f"{d} is not a valid key in the dimensions dictionary")

    def add_field(self, field: XField, name: str | None = None):
        """Add a :class:`parcels.field.Field` object to the FieldSet.

        Parameters
        ----------
        field : parcels.field.Field
            Field object to be added
        name : str
            Name of the :class:`parcels.field.Field` object to be added. Defaults
            to name in Field object.


        Examples
        --------
        For usage examples see the following tutorials:

        * `Unit converters <../examples/tutorial_unitconverters.ipynb>`__ (Default value = None)

        """
        if self._completed:
            raise RuntimeError(
                "FieldSet has already been completed. Are you trying to add a Field after you've created the ParticleSet?"
            )
        name = field.name if name is None else name

        if hasattr(self, name):  # check if Field with same name already exists when adding new Field
            raise RuntimeError(f"FieldSet already has a Field with name '{name}'")
        else:
            setattr(self, name, field)

    def add_constant_field(self, name: str, value: float, mesh: Mesh = "flat"):
        """Wrapper function to add a Field that is constant in space,
           useful e.g. when using constant horizontal diffusivity

        Parameters
        ----------
        name : str
            Name of the :class:`parcels.field.Field` object to be added
        value : float
            Value of the constant field (stored as 32-bit float)
        mesh : str
            String indicating the type of mesh coordinates and
            units used during velocity interpolation, see also `this tutorial <../examples/tutorial_unitconverters.ipynb>`__:

            1. spherical (default): Lat and lon in degree, with a
               correction for zonal velocity U near the poles.
            2. flat: No conversion, lat/lon are assumed to be in m.
        """
        import pandas as pd

        time = pd.to_datetime(['2000-01-01'])
        values = np.zeros((1,1,1,1), dtype=np.float32) + value
        data = xr.DataArray(
            data=values,
            name=name,
            dims='null',
            coords = [time,[0],[0],[0]],
            attrs=dict(
                description="null",
                units="null",
                location="node",
                mesh=f"constant",
                mesh_type=mesh
        ))
        self.add_field(
            XField(
                name,
                data,
                interp_method=None, # To do : Need to define an interpolation method for constants
                allow_time_extrapolation=True
            )
        )

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

    def _check_complete(self):
        assert self.u, 'FieldSet does not have a Field named "u"'
        assert self.v, 'FieldSet does not have a Field named "v"'
        for attr, value in vars(self).items():
            if type(value) is XField:
                assert value.name == attr, f"Field {value.name}.name ({attr}) is not consistent"

        self._add_UVfield()

        self._completed = True

    @classmethod
    def _parse_wildcards(cls, paths, filenames, var):
        if not isinstance(paths, list):
            paths = sorted(glob(str(paths)))
        if len(paths) == 0:
            notfound_paths = filenames[var] if isinstance(filenames, dict) and var in filenames else filenames
            raise OSError(f"FieldSet files not found for variable {var}: {notfound_paths}")
        for fp in paths:
            if not os.path.exists(fp):
                raise OSError(f"FieldSet file not found: {fp}")
        return paths
    
    # @classmethod
    # def from_netcdf(
    #     cls,
    #     filenames,
    #     variables,
    #     dimensions,
    #     fieldtype=None,
    #     mesh: Mesh = "spherical",
    #     allow_time_extrapolation: bool | None = None,
    #     **kwargs,
    # ):
        
    # @classmethod
    # def from_nemo(
    #     cls,
    #     filenames,
    #     variables,
    #     dimensions,
    #     mesh: Mesh = "spherical",
    #     allow_time_extrapolation: bool | None = None,
    #     tracer_interp_method: InterpMethodOption = "cgrid_tracer",
    #     **kwargs,
    # ):
           
    # @classmethod
    # def from_mitgcm(
    #     cls,
    #     filenames,
    #     variables,
    #     dimensions,
    #     mesh: Mesh = "spherical",
    #     allow_time_extrapolation: bool | None = None,
    #     tracer_interp_method: InterpMethodOption = "cgrid_tracer",
    #     **kwargs,
    # ):

    # @classmethod
    # def from_croco(
    #     cls,
    #     filenames,
    #     variables,
    #     dimensions,
    #     hc: float | None = None,
    #     mesh="spherical",
    #     allow_time_extrapolation=None,
    #     tracer_interp_method="cgrid_tracer",
    #     **kwargs,
    # ):

    # @classmethod
    # def from_c_grid_dataset(
    #     cls,
    #     filenames,
    #     variables,
    #     dimensions,
    #     mesh: Mesh = "spherical",
    #     allow_time_extrapolation: bool | None = None,
    #     tracer_interp_method: InterpMethodOption = "cgrid_tracer",
    #     gridindexingtype: GridIndexingType = "nemo",
    #     **kwargs,
    # ):


    # @classmethod
    # def from_mom5(
    #     cls,
    #     filenames,
    #     variables,
    #     dimensions,
    #     mesh: Mesh = "spherical",
    #     allow_time_extrapolation: bool | None = None,
    #     tracer_interp_method: InterpMethodOption = "bgrid_tracer",
    #     **kwargs,
    # ):

    # @classmethod
    # def from_a_grid_dataset(cls, filenames, variables, dimensions, **kwargs):

    # @classmethod
    # def from_b_grid_dataset(
    #     cls,
    #     filenames,
    #     variables,
    #     dimensions,
    #     mesh: Mesh = "spherical",
    #     allow_time_extrapolation: bool | None = None,
    #     tracer_interp_method: InterpMethodOption = "bgrid_tracer",
    #     **kwargs,
    # ):

    def add_constant(self, name, value):
        """Add a constant to the FieldSet. Note that all constants are
        stored as 32-bit floats.

        Parameters
        ----------
        name : str
            Name of the constant
        value :
            Value of the constant (stored as 32-bit float)


        Examples
        --------
        Tutorials using fieldset.add_constant:
        `Analytical advection <../examples/tutorial_analyticaladvection.ipynb>`__
        `Diffusion <../examples/tutorial_diffusion.ipynb>`__
        `Periodic boundaries <../examples/tutorial_periodic_boundaries.ipynb>`__
        """
        setattr(self, name, value)

    # def computeTimeChunk(self, time=0.0, dt=1):
    #     """Load a chunk of three data time steps into the FieldSet.
    #     This is used when FieldSet uses data imported from netcdf,
    #     with default option deferred_load. The loaded time steps are at or immediatly before time
    #     and the two time steps immediately following time if dt is positive (and inversely for negative dt)

    #     Parameters
    #     ----------
    #     time :
    #         Time around which the FieldSet data are to be loaded.
    #         Time is provided as a double, relatively to Fieldset.time_origin.
    #         Default is 0.
    #     dt :
    #         time step of the integration scheme, needed to set the direction of time chunk loading.
    #         Default is 1.
    #     """
    #     nextTime = np.inf if dt > 0 else -np.inf

    #     if abs(nextTime) == np.inf or np.isnan(nextTime):  # Second happens when dt=0
    #         return nextTime
    #     else:
    #         nSteps = int((nextTime - time) / dt)
    #         if nSteps == 0:
    #             return nextTime
    #         else:
    #             return time + nSteps * dt

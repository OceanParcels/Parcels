from __future__ import annotations

import functools
from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from parcels._core.utils.time import get_datetime_type_calendar
from parcels._core.utils.time import is_compatible as datetime_is_compatible
from parcels._reprs import fieldset_repr
from parcels._typing import Mesh
from parcels.field import Field, VectorField
from parcels.v4.grid import Grid

if TYPE_CHECKING:
    from parcels._typing import DatetimeLike
__all__ = ["FieldSet"]


class FieldSet:
    """FieldSet class that holds hydrodynamic data needed to execute particles.

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

    def __init__(self, fields: list[Field | VectorField]):
        for field in fields:
            if not isinstance(field, (Field, VectorField)):
                raise ValueError(f"Expected `field` to be a Field or VectorField object. Got {field}")
        assert_compatible_calendars(fields)

        self.fields = {f.name: f for f in fields}
        self.constants = {}

    def __getattr__(self, name):
        """Get the field by name. If the field is not found, check if it's a constant."""
        if name in self.fields:
            return self.fields[name]
        elif name in self.constants:
            return self.constants[name]
        else:
            raise AttributeError(f"FieldSet has no attribute '{name}'")

    @property
    def time_interval(self):
        """Returns the valid executable time interval of the FieldSet,
        which is the intersection of the time intervals of all fields
        in the FieldSet.
        """
        time_intervals = (f.time_interval for f in self.fields.values())

        # Filter out Nones from constant Fields
        time_intervals = (t for t in time_intervals if t is not None)
        return functools.reduce(lambda x, y: x.intersection(y), time_intervals)

    def __repr__(self):
        return fieldset_repr(self)

    def dimrange(self, dim):
        """Returns maximum value of a dimension (lon, lat, depth or time)
        on 'left' side and minimum value on 'right' side for all grids
        in a gridset. Useful for finding e.g. longitude range that
        overlaps on all grids in a gridset.
        """
        maxleft, minright = (-np.inf, np.inf)
        dim2ds = {
            "depth": ["nz1", "nz"],
            "lat": ["node_lat", "face_lat", "edge_lat"],
            "lon": ["node_lon", "face_lon", "edge_lon"],
            "time": ["time"],
        }
        for ds in self.datasets:
            for field in ds.data_vars:
                for d in dim2ds[dim]:  # check all possible dimensions
                    if d in ds[field].dims:
                        if dim == "depth":
                            maxleft = max(maxleft, ds[field][d].min().data)
                            minright = min(minright, ds[field][d].max().data)
                        else:
                            maxleft = max(maxleft, ds[field][d].data[0])
                            minright = min(minright, ds[field][d].data[-1])
        maxleft = 0 if maxleft == -np.inf else maxleft  # if all len(dim) == 1
        minright = 0 if minright == np.inf else minright  # if all len(dim) == 1

        return maxleft, minright

    @property
    def gridset_size(self):
        return len(self.fields)

    def add_field(self, field: Field, name: str | None = None):
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
        if not isinstance(field, (Field, VectorField)):
            raise ValueError(f"Expected `field` to be a Field or VectorField object. Got {type(field)}")
        assert_compatible_calendars((*self.fields.values(), field))

        name = field.name if name is None else name

        if name in self.fields:
            raise ValueError(f"FieldSet already has a Field with name '{name}'")

        self.fields[name] = field

    def add_constant_field(self, name: str, value, mesh: Mesh = "flat"):
        """Wrapper function to add a Field that is constant in space,
           useful e.g. when using constant horizontal diffusivity

        Parameters
        ----------
        name : str
            Name of the :class:`parcels.field.Field` object to be added
        value :
            Value of the constant field
        mesh : str
            String indicating the type of mesh coordinates and
            units used during velocity interpolation, see also `this tutorial <../examples/tutorial_unitconverters.ipynb>`__:

            1. spherical (default): Lat and lon in degree, with a
               correction for zonal velocity U near the poles.
            2. flat: No conversion, lat/lon are assumed to be in m.
        """
        da = xr.DataArray(
            data=np.full((1, 1, 1, 1), value),
            dims=["T", "ZG", "YG", "XG"],
            coords={
                "ZG": (["ZG"], np.arange(1), {"axis": "Z"}),
                "YG": (["YG"], np.arange(1), {"axis": "Y"}),
                "XG": (["XG"], np.arange(1), {"axis": "X"}),
                "lon": (["XG"], np.arange(1), {"axis": "X"}),
                "lat": (["YG"], np.arange(1), {"axis": "Y"}),
                "depth": (["ZG"], np.arange(1), {"axis": "Z"}),
            },
        )
        grid = Grid(da)
        self.add_field(
            Field(
                name,
                da,
                grid,
                interp_method=None,  # TODO : Need to define an interpolation method for constants
            )
        )

    def get_fields(self) -> list[Field | VectorField]:
        """Returns a list of all the :class:`parcels.field.Field` and :class:`parcels.field.VectorField`
        objects associated with this FieldSet.
        """
        fields = []
        for v in self.__dict__.values():
            if type(v) in [Field, VectorField]:
                if v not in fields:
                    fields.append(v)
        return fields

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
        if name in self.constants:
            raise ValueError(f"FieldSet already has a constant with name '{name}'")

        self.constants[name] = np.float32(value)

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


class CalendarError(Exception):  # TODO: Move to a parcels errors module
    """Exception raised when the calendar of a field is not compatible with the rest of the Fields. The user should ensure that they only add fields to a FieldSet that have compatible CFtime calendars."""


def assert_compatible_calendars(fields: Iterable[Field]):
    time_intervals = [f.time_interval for f in fields if f.time_interval is not None]
    reference_datetime_object = time_intervals[0].left

    for field in fields:
        if field.time_interval is None:
            continue

        if not datetime_is_compatible(reference_datetime_object, field.time_interval.left):
            msg = format_calendar_error_message(field, reference_datetime_object)
            raise CalendarError(msg)


def format_calendar_error_message(field: Field, reference_datetime: DatetimeLike) -> str:
    def datetime_to_msg(example_datetime: DatetimeLike) -> str:
        datetime_type, calendar = get_datetime_type_calendar(example_datetime)
        msg = str(datetime_type)
        if calendar is not None:
            msg += f" with cftime calendar {calendar}'"
        return msg

    return f"Expected field {field.name!r} to have calendar compatible with datetime object {datetime_to_msg(reference_datetime)}. Got field with calendar {datetime_to_msg(field.time_interval.left)}. Have you considered using xarray to update the time dimension of the dataset to have a compatible calendar?"

from __future__ import annotations

import functools
from collections.abc import Iterable
from typing import TYPE_CHECKING

import cf_xarray  # noqa: F401
import numpy as np
import xarray as xr
import xgcm

from parcels._core.converters import Geographic, GeographicPolar
from parcels._core.field import Field, VectorField
from parcels._core.utils.time import get_datetime_type_calendar
from parcels._core.utils.time import is_compatible as datetime_is_compatible
from parcels._core.xgrid import _DEFAULT_XGCM_KWARGS, XGrid
from parcels._logger import logger
from parcels._typing import Mesh

if TYPE_CHECKING:
    from parcels._core.basegrid import BaseGrid
    from parcels._typing import TimeLike
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
        self.constants: dict[str, float] = {}

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
        time_intervals = [t for t in time_intervals if t is not None]
        if len(time_intervals) == 0:  # All fields are constant fields
            return None
        return functools.reduce(lambda x, y: x.intersection(y), time_intervals)

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
        ds = xr.Dataset({name: (["time", "lat", "lon", "depth"], np.full((1, 1, 1, 1), value))})
        grid = XGrid(xgcm.Grid(ds, **_DEFAULT_XGCM_KWARGS))
        self.add_field(
            Field(
                name,
                ds[name],
                grid,
                interp_method=None,  # TODO : Need to define an interpolation method for constants
            )
        )

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
        if not isinstance(value, (float, np.floating, int, np.integer)):
            raise ValueError(f"FieldSet constants have to be of type float or int, got a {type(value)}")
        self.constants[name] = np.float32(value)

    @property
    def gridset(self) -> list[BaseGrid]:
        grids = []
        for field in self.fields.values():
            if field.grid not in grids:
                grids.append(field.grid)
        return grids

    def from_copernicusmarine(ds: xr.Dataset):
        """Create a FieldSet from a Copernicus Marine Service xarray.Dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            xarray.Dataset as obtained from the copernicusmarine toolbox.

        Returns
        -------
        FieldSet
            FieldSet object containing the fields from the dataset that can be used for a Parcels simulation.

        Notes
        -----
        See https://help.marine.copernicus.eu/en/collections/9080063-copernicus-marine-toolbox for more information on the copernicusmarine toolbox.
        The toolbox to ingest data from most of the products on the Copernicus Marine Service (https://data.marine.copernicus.eu/products) into an xarray.Dataset.
        You can use indexing and slicing to select a subset of the data before passing it to this function.
        Note that most Parcels uses will require both U and V fields to be present in the dataset. This function will try to find out which variables in the dataset correspond to U and V.
        To override the automatic detection, rename the appropriate variables in your dataset to 'U' and 'V' before passing it to this function.

        """
        ds = ds.copy()
        ds = _discover_copernicusmarine_U_and_V(ds)
        expected_axes = set("XYZT")  # TODO: Update after we have support for 2D spatial fields
        if missing_axes := (expected_axes - set(ds.cf.axes)):
            raise ValueError(
                f"Dataset missing axes {missing_axes} to have coordinates for all {expected_axes} axes according to CF conventions."
            )

        ds = _rename_coords_copernicusmarine(ds)
        grid = XGrid(
            xgcm.Grid(
                ds,
                coords={
                    "X": {
                        "left": "lon",
                    },
                    "Y": {
                        "left": "lat",
                    },
                    "Z": {
                        "left": "depth",
                    },
                    "T": {
                        "center": "time",
                    },
                },
                autoparse_metadata=False,
                **_DEFAULT_XGCM_KWARGS,
            )
        )

        fields = {}
        if "U" in ds.data_vars and "V" in ds.data_vars:
            fields["U"] = Field("U", ds["U"], grid)
            fields["V"] = Field("V", ds["V"], grid)
            fields["U"].units = GeographicPolar()
            fields["V"].units = Geographic()

            if "W" in ds.data_vars:
                ds["W"] -= ds[
                    "W"
                ]  # Negate W to convert from up positive to down positive (as that's the direction of positive depth)
                fields["W"] = Field("W", ds["W"], grid)
                fields["UVW"] = VectorField("UVW", fields["U"], fields["V"], fields["W"])
            else:
                fields["UV"] = VectorField("UV", fields["U"], fields["V"])

        for varname in set(ds.data_vars) - set(fields.keys()):
            fields[varname] = Field(varname, ds[varname], grid)

        return FieldSet(list(fields.values()))


class CalendarError(Exception):  # TODO: Move to a parcels errors module
    """Exception raised when the calendar of a field is not compatible with the rest of the Fields. The user should ensure that they only add fields to a FieldSet that have compatible CFtime calendars."""


def assert_compatible_calendars(fields: Iterable[Field | VectorField]):
    time_intervals = [f.time_interval for f in fields if f.time_interval is not None]

    if len(time_intervals) == 0:  # All time intervals are none
        return

    reference_datetime_object = time_intervals[0].left

    for field in fields:
        if field.time_interval is None:
            continue

        if not datetime_is_compatible(reference_datetime_object, field.time_interval.left):
            msg = _format_calendar_error_message(field, reference_datetime_object)
            raise CalendarError(msg)


def _datetime_to_msg(example_datetime: TimeLike) -> str:
    datetime_type, calendar = get_datetime_type_calendar(example_datetime)
    msg = str(datetime_type)
    if calendar is not None:
        msg += f" with cftime calendar {calendar}'"
    return msg


def _format_calendar_error_message(field: Field, reference_datetime: TimeLike) -> str:
    return f"Expected field {field.name!r} to have calendar compatible with datetime object {_datetime_to_msg(reference_datetime)}. Got field with calendar {_datetime_to_msg(field.time_interval.left)}. Have you considered using xarray to update the time dimension of the dataset to have a compatible calendar?"


_COPERNICUS_MARINE_AXIS_VARNAMES = {
    "X": "lon",
    "Y": "lat",
    "Z": "depth",
    "T": "time",
}


def _rename_coords_copernicusmarine(ds):
    try:
        for axis, [coord] in ds.cf.axes.items():
            ds = ds.rename({coord: _COPERNICUS_MARINE_AXIS_VARNAMES[axis]})
    except ValueError as e:
        raise ValueError(f"Multiple coordinates found for Copernicus dataset on axis '{axis}'. Check your data.") from e
    return ds


def _discover_copernicusmarine_U_and_V(ds: xr.Dataset) -> xr.Dataset:
    # Assumes that the dataset has U and V data

    cf_UV_standard_name_fallbacks = [
        (
            "eastward_sea_water_velocity",
            "northward_sea_water_velocity",
        ),  # GLOBAL_ANALYSISFORECAST_PHY_001_024, MEDSEA_ANALYSISFORECAST_PHY_006_013, BALTICSEA_ANALYSISFORECAST_PHY_003_006, BLKSEA_ANALYSISFORECAST_PHY_007_001, IBI_ANALYSISFORECAST_PHY_005_001, NWSHELF_ANALYSISFORECAST_PHY_004_013, MULTIOBS_GLO_PHY_MYNRT_015_003, MULTIOBS_GLO_PHY_W_3D_REP_015_007
        (
            "surface_geostrophic_eastward_sea_water_velocity",
            "surface_geostrophic_northward_sea_water_velocity",
        ),  # SEALEVEL_GLO_PHY_L4_MY_008_047, SEALEVEL_EUR_PHY_L4_NRT_008_060
        (
            "geostrophic_eastward_sea_water_velocity",
            "geostrophic_northward_sea_water_velocity",
        ),  # MULTIOBS_GLO_PHY_TSUV_3D_MYNRT_015_012
        (
            "sea_surface_wave_stokes_drift_x_velocity",
            "sea_surface_wave_stokes_drift_y_velocity",
        ),  # GLOBAL_ANALYSISFORECAST_WAV_001_027, MEDSEA_MULTIYEAR_WAV_006_012, ARCTIC_ANALYSIS_FORECAST_WAV_002_014, BLKSEA_ANALYSISFORECAST_WAV_007_003, IBI_ANALYSISFORECAST_WAV_005_005, NWSHELF_ANALYSISFORECAST_WAV_004_014
        ("sea_water_x_velocity", "sea_water_y_velocity"),  # ARCTIC_ANALYSISFORECAST_PHY_002_001
        (
            "eastward_sea_water_velocity_vertical_mean_over_pelagic_layer",
            "northward_sea_water_velocity_vertical_mean_over_pelagic_layer",
        ),  # GLOBAL_MULTIYEAR_BGC_001_033
    ]
    cf_W_standard_name_fallbacks = ["upward_sea_water_velocity", "vertical_sea_water_velocity"]

    if "W" not in ds:
        for cf_standard_name_W in cf_W_standard_name_fallbacks:
            if cf_standard_name_W in ds.cf.standard_names:
                ds = _ds_rename_using_standard_names(ds, {cf_standard_name_W: "W"})
                break

    if "U" in ds and "V" in ds:
        return ds  # U and V already present
    elif "U" in ds or "V" in ds:
        raise ValueError(
            "Dataset has only one of the two variables 'U' and 'V'. Please rename the appropriate variable in your dataset to have both 'U' and 'V' for Parcels simulation."
        )

    for cf_standard_name_U, cf_standard_name_V in cf_UV_standard_name_fallbacks:
        if cf_standard_name_U in ds.cf.standard_names:
            if cf_standard_name_V not in ds.cf.standard_names:
                raise ValueError(
                    f"Dataset has variable with CF standard name {cf_standard_name_U!r}, "
                    f"but not the matching variable with CF standard name {cf_standard_name_V!r}. "
                    "Please rename the appropriate variables in your dataset to have both 'U' and 'V' for Parcels simulation."
                )
        else:
            continue

        ds = _ds_rename_using_standard_names(ds, {cf_standard_name_U: "U", cf_standard_name_V: "V"})
        break
    return ds


def _ds_rename_using_standard_names(ds: xr.Dataset, name_dict: dict[str, str]) -> xr.Dataset:
    for standard_name, rename_to in name_dict.items():
        name = ds.cf[standard_name].name
        ds = ds.rename({name: rename_to})
        logger.info(
            f"cf_xarray found variable {name!r} with CF standard name {standard_name!r} in dataset, renamed it to {rename_to!r} for Parcels simulation."
        )
    return ds

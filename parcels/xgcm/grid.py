"""This Grid object is adapted from xgcm.Grid, removing a lot of the code that is not needed for Parcels."""

import warnings
from collections import OrderedDict
from collections.abc import Iterable
from typing import (
    Any,
)

from parcels.xgcm import comodo

_VALID_BOUNDARY = [None, "fill", "extend", "periodic"]


def _maybe_promote_str_to_list(a):
    # TODO: improve this
    if isinstance(a, str):
        return [a]
    else:
        return a


class Axis:
    """
    An object that represents a group of coordinates that all lie along the same
    physical dimension but at different positions with respect to a grid cell.
    There are four possible positions:

         Center
         |------o-------|------o-------|------o-------|------o-------|
               [0]            [1]            [2]            [3]

         Left
         |------o-------|------o-------|------o-------|------o-------|
        [0]            [1]            [2]            [3]

         Right
         |------o-------|------o-------|------o-------|------o-------|
                       [0]            [1]            [2]            [3]

         Inner
         |------o-------|------o-------|------o-------|------o-------|
                       [0]            [1]            [2]

         Outer
         |------o-------|------o-------|------o-------|------o-------|
        [0]            [1]            [2]            [3]            [4]

    The `center` position is the only one without the `c_grid_axis_shift`
    attribute, which must be present for the other four. However, the actual
    value of `c_grid_axis_shift` is ignored for `inner` and `outer`, which are
    differentiated by their length.
    """

    def __init__(
        self,
        ds,
        axis_name,
        periodic=True,
        default_shifts=None,
        coords=None,
        boundary=None,
        fill_value=None,
    ):
        """
        Create a new Axis object from an input dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Contains the relevant grid information. Coordinate attributes
            should conform to Comodo conventions [1]_.
        axis_name : str
            The name of the axis (should match axis attribute)
        periodic : bool, optional
            Whether the domain is periodic along this axis
        default_shifts : dict, optional
            Default mapping from and to grid positions
            (e.g. `{'center': 'left'}`). Will be inferred if not specified.
        coords : dict, optional
            Mapping of axis positions to coordinate names
            (e.g. `{'center': 'XC', 'left: 'XG'}`)
        boundary : str or dict, optional,
            boundary can either be one of {None, 'fill', 'extend', 'extrapolate', 'periodic'}

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition where
              the difference at the boundary will be zero.)
            * 'extrapolate': Set values by extrapolating linearly from the two
              points nearest to the edge
            * 'periodic' : Wrap arrays around. Equivalent to setting `periodic=True`
            This sets the default value. It can be overriden by specifying the
            boundary kwarg when calling specific methods.
        fill_value : float, optional
            The value to use in the boundary condition when `boundary='fill'`.

        References
        ----------
        .. [1] Comodo Conventions https://web.archive.org/web/20160417032300/http://pycomodo.forge.imag.fr/norm.html
        """
        if default_shifts is None:
            default_shifts = {}
        self._ds = ds
        self.name = axis_name
        self._periodic = periodic
        if boundary not in _VALID_BOUNDARY:
            raise ValueError(f"Expected 'boundary' to be one of {_VALID_BOUNDARY}. Received {boundary!r} instead.")
        self.boundary = boundary
        if fill_value is not None and not isinstance(fill_value, (int, float)):
            raise ValueError("Expected 'fill_value' to be a number.")
        self.fill_value = fill_value if fill_value is not None else 0.0

        if coords:
            # use specified coords
            self.coords = {pos: name for pos, name in coords.items()}
        else:
            # fall back on comodo conventions
            self.coords = comodo.get_axis_positions_and_coords(ds, axis_name)

        # self.coords is a dictionary with the following structure
        #   key: position_name {'center' ,'left' ,'right', 'outer', 'inner'}
        #   value: name of the dimension

        # set default position shifts
        fallback_shifts = {
            "center": ("left", "right", "outer", "inner"),
            "left": ("center",),
            "right": ("center",),
            "outer": ("center",),
            "inner": ("center",),
        }
        self._default_shifts = {}
        for pos in self.coords:
            # use user-specified value if present
            if pos in default_shifts:
                self._default_shifts[pos] = default_shifts[pos]
            else:
                for possible_shift in fallback_shifts[pos]:
                    if possible_shift in self.coords:
                        self._default_shifts[pos] = possible_shift
                        break

        ########################################################################
        # DEVELOPER DOCUMENTATION
        #
        # The attributes below are my best attempt to represent grid topology
        # in a general way. The data structures are complicated, but I can't
        # think of any way to simplify them.
        #
        # self._facedim (str) is the name of a dimension (e.g. 'face') or None.
        # If it is None, that means that the grid topology is _simple_, i.e.
        # that this is not a cubed-sphere grid or similar. For example:
        #
        #     ds.dims == ('time', 'lat', 'lon')
        #
        # If _facedim is set to a dimension name, that means that shifting
        # grid positions requires exchanging data among multiple "faces"
        # (a.k.a. "tiles", "facets", etc.). For this to work, there must be a
        # dimension corresponding to the different faces. This is `_facedim`.
        # For example:
        #
        #     ds.dims == ('time', 'face', 'lat', 'lon')
        #
        # In this case, `self._facedim == 'face'`
        #
        # We initialize all of this to None and let the `Grid` class handle
        # setting these attributes for complex geometries.
        self._facedim = None
        #
        # `self._connections` is a dictionary. It contains information about the
        # connectivity among this axis and other axes.
        # It should have the structure
        #
        #     {facedim_index: ((left_facedim_index, left_axis, left_reverse),
        #                      (right_facedim_index, right_axis, right_reverse)}
        #
        # `facedim_index` : a value used to index the `self._facedim` dimension
        #   (If `self._facedim` is `None`, then there should be only one key in
        #   `facedim_index` and that key should be `None`.)
        # `left_facedim_index` : the facedim index of the neighbor to the left.
        #   (If `self._facedim` is `None`, this must also be `None`.)
        # `left_axis` : an `Axis` object for the values to the left of this axis
        # `left_reverse` : bool, whether the connection should be reversed. By
        #   default, the left side of this axis will be connected to the right
        #   side of the neighboring axis. `left_reverse` overrides this and
        #   instead connects to the left side of the neighboring axis
        self._connections = {None: (None, None)}

        # now we implement periodic coordinates by setting appropriate
        # connections
        if periodic:
            self._connections = {None: ((None, self, False), (None, self, False))}

    def __repr__(self):
        is_periodic = "periodic" if self._periodic else "not periodic"
        summary = [f"<parcels.Axis '{self.name}' ({is_periodic}, boundary={self.boundary!r})>"]
        summary.append("Axis Coordinates:")
        summary += self._coord_desc()
        return "\n".join(summary)

    def _coord_desc(self):
        summary = []
        for name, cname in self.coords.items():
            coord_info = f"  * {name:<8} {cname}"
            if name in self._default_shifts:
                coord_info += f" --> {self._default_shifts[name]}"
            summary.append(coord_info)
        return summary

    def _get_position_name(self, da):
        """Return the position and name of the axis coordinate in a DataArray."""
        for position, coord_name in self.coords.items():
            # TODO: should we have more careful checking of alignment here?
            if coord_name in da.dims:
                return position, coord_name

        raise KeyError(f"None of the DataArray's dims {da.dims!r} were found in axis coords.")

    def _get_axis_dim_num(self, da):
        """Return the dimension number of the axis coordinate in a DataArray."""
        _, coord_name = self._get_position_name(da)
        return da.get_axis_num(coord_name)


class Grid:
    """
    An object with multiple :class:`parcels.Axis` objects representing different
    independent axes.
    """

    def __init__(
        self,
        ds,
        check_dims=True,
        periodic=True,
        default_shifts=None,
        face_connections=None,
        coords=None,
        metrics=None,
        boundary=None,
        fill_value=None,
    ):
        """
        Create a new Grid object from an input dataset.

        Parameters
        ----------
        ds : xarray.Dataset
            Contains the relevant grid information. Coordinate attributes
            should conform to Comodo conventions [1]_.
        check_dims : bool, optional
            Whether to check the compatibility of input data dimensions before
            performing grid operations.
        periodic : {True, False, list}
            Whether the grid is periodic (i.e. "wrap-around"). If a list is
            specified (e.g. ``['X', 'Y']``), the axis names in the list will be
            be periodic and any other axes founds will be assumed non-periodic.
        default_shifts : dict
            A dictionary of dictionaries specifying default grid position
            shifts (e.g. ``{'X': {'center': 'left', 'left': 'center'}}``)
        face_connections : dict
            Grid topology
        coords : dict, optional
            Specifies positions of dimension names along axes X, Y, Z, e.g
            ``{'X': {'center': 'XC', 'left: 'XG'}}``.
            Each key should be an axis name (e.g., `X`, `Y`, or `Z`) and map
            to a dictionary which maps positions (`center`, `left`, `right`,
            `outer`, `inner`) to dimension names in the dataset
            (in the example above, `XC` is at the `center` position and `XG`
            at the `left` position along the `X` axis).
            If the values are not present in ``ds`` or are not dimensions,
            an error will be raised.
        metrics : dict, optional
            Specification of grid metrics mapping axis names (X, Y, Z) to corresponding
            metric variable names in the dataset
            (e.g. {('X',):['dx_t'], ('X', 'Y'):['area_tracer', 'area_u']}
            for the cell distance in the x-direction ``dx_t`` and the
            horizontal cell areas ``area_tracer`` and ``area_u``, located at
            different grid positions).
        boundary : {None, 'fill', 'extend', 'extrapolate', dict}, optional
            A flag indicating how to handle boundaries:

            * None:  Do not apply any boundary conditions. Raise an error if
              boundary conditions are required for the operation.
            * 'fill':  Set values outside the array boundary to fill_value
              (i.e. a Dirichlet boundary condition.)
            * 'extend': Set values outside the array to the nearest array
              value. (i.e. a limited form of Neumann boundary condition.)
            * 'extrapolate': Set values by extrapolating linearly from the two
              points nearest to the edge
            Optionally a dict mapping axis name to seperate values for each axis
            can be passed.
        fill_value : {float, dict}, optional
            The value to use in boundary conditions with `boundary='fill'`.
            Optionally a dict mapping axis name to seperate values for each axis
            can be passed.

        References
        ----------
        .. [1] Comodo Conventions https://web.archive.org/web/20160417032300/http://pycomodo.forge.imag.fr/norm.html
        """
        if default_shifts is None:
            default_shifts = {}
        self._ds = ds
        self._check_dims = check_dims

        if boundary:
            warnings.warn(
                "The `boundary` argument will be renamed "
                "to `padding` to better reflect the process "
                "of array padding and avoid confusion with "
                "physical boundary conditions (e.g. ocean land boundary).",
                category=DeprecationWarning,
                stacklevel=2,
            )

        # Deprecation Warnigns
        if periodic:
            warnings.warn(
                "The `periodic` argument will be deprecated. "
                "To preserve previous behavior supply `boundary = 'periodic'.",
                category=DeprecationWarning,
                stacklevel=2,
            )

        if fill_value:
            warnings.warn(
                "The default fill_value will be changed to nan (from 0.0 previously) "
                "in future versions. Provide `fill_value=0.0` to preserve previous behavior.",
                category=DeprecationWarning,
                stacklevel=2,
            )

        extrapolate_warning = False
        if boundary == "extrapolate":
            extrapolate_warning = True
        if isinstance(boundary, dict):
            if any([k == "extrapolate" for k in boundary.keys()]):
                extrapolate_warning = True
        if extrapolate_warning:
            warnings.warn(
                "The `boundary='extrapolate'` option will no longer be supported in future releases.",
                category=DeprecationWarning,
                stacklevel=2,
            )

        if coords:
            all_axes = coords.keys()
        else:
            all_axes = comodo.get_all_axes(ds)
            coords = {}

        # check coords input validity
        for axis, positions in coords.items():
            for pos, dim in positions.items():
                if not (dim in ds.variables or dim in ds.dims):
                    raise ValueError(
                        f"Could not find dimension `{dim}` (for the `{pos}` position on axis `{axis}`) in input dataset."
                    )
                if dim not in ds.dims:
                    raise ValueError(
                        f"Input `{dim}` (for the `{pos}` position on axis `{axis}`) is not a dimension in the input datasets `ds`."
                    )

        # Convert all inputs to axes-kwarg mappings
        # TODO We need a way here to check valid input. Maybe also in _as_axis_kwargs?
        # Parse axis properties
        boundary = self._as_axis_kwarg_mapping(boundary, axes=all_axes)
        fill_value = self._as_axis_kwarg_mapping(fill_value, axes=all_axes)
        # TODO: In the future we want this the only place where we store these.
        # TODO: This info needs to then be accessible to e.g. pad()

        # Parse list input. This case does only apply to periodic.
        # Since we plan on deprecating it soon handle it here, so we can easily
        # remove it later
        if isinstance(periodic, list):
            periodic = {axname: True for axname in periodic}
        periodic = self._as_axis_kwarg_mapping(periodic, axes=all_axes)

        # Set properties on grid object.
        self._facedim = list(face_connections.keys())[0] if face_connections else None
        self._connections = face_connections if face_connections else None
        # TODO: I think of the face connection data as grid not axes properties, since they almost by defintion
        # TODO: involve multiple axes. In a future PR we should remove this info from the axes
        # TODO: but make sure to properly port the checking functionality!

        # Populate axes. Much of this is just for backward compatibility.
        self.axes = OrderedDict()
        for axis_name in all_axes:
            # periodic
            is_periodic = periodic.get(axis_name, False)

            # default_shifts
            if axis_name in default_shifts:
                axis_default_shifts = default_shifts[axis_name]
            else:
                axis_default_shifts = {}

            # boundary
            if isinstance(boundary, dict):
                axis_boundary = boundary.get(axis_name, None)
            elif isinstance(boundary, str) or boundary is None:
                axis_boundary = boundary
            else:
                raise ValueError(
                    f"boundary={boundary} is invalid. Please specify a dictionary "
                    "mapping axis name to a boundary option; a string or None."
                )

            if isinstance(fill_value, dict):
                axis_fillvalue = fill_value.get(axis_name, None)  # TODO: This again sets defaults. Dont do that here.
            elif isinstance(fill_value, (int, float)) or fill_value is None:
                axis_fillvalue = fill_value
            else:
                raise ValueError(
                    f"fill_value={fill_value} is invalid. Please specify a dictionary "
                    "mapping axis name to a boundary option; a number or None."
                )

            self.axes[axis_name] = Axis(
                ds,
                axis_name,
                is_periodic,
                default_shifts=axis_default_shifts,
                coords=coords.get(axis_name),
                boundary=axis_boundary,
                fill_value=axis_fillvalue,
            )

        if face_connections is not None:
            self._assign_face_connections(face_connections)

        self._metrics = {}

        if metrics is not None:
            for key, value in metrics.items():
                self.set_metrics(key, value)

    def _as_axis_kwarg_mapping(
        self,
        kwargs: Any | dict[str, Any],
        axes: Iterable[str] | None = None,
        ax_property_name=None,
        default_value: Any | None = None,
    ) -> dict[str, Any]:
        """Convert kwarg input into dict for each available axis
        E.g. for a grid with 2 axes for the keyword argument `periodic`
        periodic = True --> periodic = {'X': True, 'Y':True}
        or if not all axes are provided, the other axes will be parsed as defaults (None)
        periodic = {'X':True} --> periodic={'X': True, 'Y':None}
        """
        if axes is None:
            axes = self.axes

        parsed_kwargs: dict[str, Any] = dict()

        if isinstance(kwargs, dict):
            parsed_kwargs = kwargs
        else:
            for axname in axes:
                parsed_kwargs[axname] = kwargs

        # Check axis properties for values that were not provided (before using the default)
        if ax_property_name is not None:
            for axname in axes:
                if axname not in parsed_kwargs.keys() or parsed_kwargs[axname] is None:
                    ax_property = getattr(self.axes[axname], ax_property_name)
                    parsed_kwargs[axname] = ax_property

        # if None set to default value.
        parsed_kwargs_w_defaults = {k: default_value if v is None else v for k, v in parsed_kwargs.items()}
        # At this point the output should be guaranteed to have an entry per existing axis.
        # If neither a default value was given, nor an axis property was found, the value will be mapped to None.

        # temporary hack to get periodic conditions from axis
        if ax_property_name == "boundary":
            for axname in axes:
                if self.axes[axname]._periodic:
                    if axname not in parsed_kwargs_w_defaults.keys():
                        parsed_kwargs_w_defaults[axname] = "periodic"

        return parsed_kwargs_w_defaults

    def _assign_face_connections(self, fc):
        """Check a dictionary of face connections to make sure all the links are
        consistent.
        """
        if len(fc) > 1:
            raise ValueError(f"Only one face dimension is supported for now. Instead found {repr(fc.keys())!r}")

        # we will populate this with the axes we find in face_connections
        axis_connections = {}

        facedim = list(fc.keys())[0]
        assert facedim in self._ds

        face_links = fc[facedim]
        for fidx, face_axis_links in face_links.items():
            for axis, axis_links in face_axis_links.items():
                # initialize the axis dict if necssary
                if axis not in axis_connections:
                    axis_connections[axis] = {}
                link_left, link_right = axis_links

                def check_neighbor(link, position):
                    if link is None:
                        return
                    idx, ax, rev = link
                    # need to swap position if the link is reversed
                    correct_position = int(not position) if rev else position
                    try:
                        neighbor_link = face_links[idx][ax][correct_position]
                    except (KeyError, IndexError):
                        raise KeyError(  # noqa: B904
                            f"Couldn't find a face link for face {idx!r}in axis {ax!r} at position {correct_position!r}"
                        )
                    idx_n, ax_n, rev_n = neighbor_link
                    if ax not in self.axes:
                        raise KeyError(f"axis {ax!r} is not a valid axis")
                    if ax_n not in self.axes:
                        raise KeyError(f"axis {ax_n!r} is not a valid axis")
                    if idx not in self._ds[facedim].values:
                        raise IndexError(f"{idx!r} is not a valid index for face dimension {facedim!r}")
                    if idx_n not in self._ds[facedim].values:
                        raise IndexError(f"{idx!r} is not a valid index for face dimension {facedim!r}")
                    # check for consistent links from / to neighbor
                    if (idx_n != fidx) or (ax_n != axis) or (rev_n != rev):  # noqa: B023 # TODO: fix?
                        raise ValueError(
                            "Face link mismatch: neighbor doesn't"
                            " correctly link back to this face. "
                            f"face: {fidx!r}, axis: {axis!r}, position: {position!r}, "  # noqa: B023 # TODO: fix?
                            f"rev: {rev!r}, link: {link!r}, neighbor_link: {neighbor_link!r}"
                        )
                    # convert the axis name to an acutal axis object
                    actual_axis = self.axes[ax]
                    return idx, actual_axis, rev

                left = check_neighbor(link_left, 1)
                right = check_neighbor(link_right, 0)
                axis_connections[axis][fidx] = (left, right)

        for axis, axis_links in axis_connections.items():
            self.axes[axis]._facedim = facedim
            self.axes[axis]._connections = axis_links

    def set_metrics(self, key, value, overwrite=False):
        metric_axes = frozenset(_maybe_promote_str_to_list(key))
        axes_not_found = [ma for ma in metric_axes if ma not in self.axes]
        if len(axes_not_found) > 0:
            raise KeyError(f"Metric axes {axes_not_found!r} not compatible with grid axes {tuple(self.axes)!r}")

        metric_value = _maybe_promote_str_to_list(value)
        for metric_varname in metric_value:
            if metric_varname not in self._ds.variables:
                raise KeyError(f"Metric variable {metric_varname} not found in dataset.")

        existing_metric_axes = set(self._metrics.keys())
        if metric_axes in existing_metric_axes:
            value_exist = self._metrics.get(metric_axes)
            # resetting coords avoids potential broadcasting / alignment issues
            value_new = self._ds[metric_varname].reset_coords(drop=True)
            did_overwrite = False
            # go through each existing value until data array with matching dimensions is selected
            for idx, ve in enumerate(value_exist):
                # double check if dimensions match
                if set(value_new.dims) == set(ve.dims):
                    if overwrite:
                        # replace existing data array with new data array input
                        self._metrics[metric_axes][idx] = value_new
                        did_overwrite = True
                    else:
                        raise ValueError(
                            f"Metric variable {ve.name} with dimensions {ve.dims} already assigned in metrics."
                            f" Overwrite {ve.name} with {metric_varname} by setting overwrite=True."
                        )
            # if no existing value matches new value dimension-wise, just append new value
            if not did_overwrite:
                self._metrics[metric_axes].append(value_new)
        else:
            # no existing metrics for metric_axes yet; initialize empty list
            self._metrics[metric_axes] = []
            for metric_varname in metric_value:
                metric_var = self._ds[metric_varname].reset_coords(drop=True)
                self._metrics[metric_axes].append(metric_var)

    def __repr__(self):
        summary = ["<parcels.Grid>"]
        for name, axis in self.axes.items():
            is_periodic = "periodic" if axis._periodic else "not periodic"
            summary.append(f"{name} Axis ({is_periodic}, boundary={axis.boundary!r}):")
            summary += axis._coord_desc()
        return "\n".join(summary)

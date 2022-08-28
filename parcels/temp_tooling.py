"""
A temporary file to house unmerged changes to the parcels codebase.
"""
import numpy as np
import xarray as xr
from parcels import FieldSet
from pathlib import Path

from typing import Tuple

def delft_to_fieldset(path: Path, return_ds: bool = False, **kwargs) -> FieldSet:
    """
    Generates a FieldSet from a Delft NetCDF dataset.


    Delft Netcdf format:
    Dimension indices:
    - MC: Index of cell edges (east-west)
    - NC: Index of cell edges (north-south)
    - M: Index of cell centres (east-west)
    - N: Index of cell centres (north-south)
    
    Dimension values:
    - XCOR[MC, NC]: East-west position of cell edges for entire grid
    - YCOR[MC, NC]: North-south position of cell edges for entire grid
    - XZ[M, N]: East-west position of cell centres for entire grid
    - YZ[M, N]: North-south position of cell centres for entire grid

    Variables:
    - U1[XCOR, YZ]: East-west velocity
    - V1[XZ, YCOR]: North-south velocity
    
            V
        +--------+
        |        |
       U|        |U
        |        |
        +--------+
            V
    """

    # Reading the netcdf file
    ds = xr.open_dataset(path)[["U1", "V1", "S1"]]

    # Generating missing points in meshgrid by extrapolating & interpolating
    ds = generate_missing_gridpoints(ds)

    # Coercing into mitgcm format (where everything is from the reference point of the bottom left gridpoint for a cell)
    ds = ds.assign(
        U1 = ds.U1.swap_dims({"N":"NC"}),
        V1 = ds.V1.swap_dims({"M":"MC"}),
        S1 = ds.S1.swap_dims({"N":"NC", "M": "MC"}),
        XZ = ds.XZ.swap_dims({"N":"NC", "M":"MC"}),
        YZ = ds.YZ.swap_dims({"N":"NC", "M":"MC"}),
    )[["U1", "V1", "S1"]].drop(["XZ", "YZ"]).isel(KMAXOUT_RESTR=0) # Dropping extraneous coordinates
    
    # Adding in coordinates for the axes
    ds = ds.assign(
        x = ds.XCOR.isel(NC=0),
        y = ds.YCOR.isel(MC=0),
    ).set_coords(["x", "y"])

    ds = ds.transpose("time", "NC", "MC", ...) # Transposing xarray dataset as per https://github.com/OceanParcels/parcels/issues/1180
    ds = ds.rename({"U1": "U", "V1": "V", "S1": "S", "XCOR": "x_mesh", "YCOR": "y_mesh", "MC": "x_index", "NC": "y_index"}) # Renaming variables


    variables = {
        "U": "U",
        "V": "V",
        }
    dimensions = {
            "lon": "x_mesh", # XCOR dimension corresponding to the `MC` index in the netcdf file
            "lat": "y_mesh", # YZ dimension corresponding to the `N` index in the netcdf file
            # "depth": "KMAXOUT_RESTR", # Needs to be gridded
            "time": "time",
    }
    
    # TODO: Add support for depth (both sigma grids and Z-grids). Currently the depth `KMAXOUT_RESTR` is a 1D vector, however it must be a grid according to https://github.com/OceanParcels/parcels/blob/4110c393d19d0001530bc42de16e465e0cfa0d34/parcels/grid.py#L461
    # Only works for NetCDF files
    # field_set = FieldSet.from_mitgcm(path, variables, dimensions, mesh="flat")
    
    if return_ds:
        return ds
    field_set = FieldSet.from_xarray_dataset(
        ds,
        variables,
        dimensions,
        mesh="flat",
        interp_method="cgrid_velocity",
        gridindexingtype="mitgcm",
        **kwargs
        )
    return field_set

    
def generate_missing_gridpoints(ds: xr.Dataset, missing_value: float = 0) -> xr.Dataset:
    """
    Generates missing gridpoints for the given Delft NetCDF dataset.
    Works under the fundamental assumption that the mesh is a regular, catesian grid.

    Parameters
    ----------
    ds : xr.Dataset
        Delft NetCDF dataset
    missing_value : float
        Value representing missing gridpoints in the dataset
    
    Returns
    -------
    xr.Dataset
        Dataset with missing gridpoints
    """
    # Get the x and y coordinates of the cell edges as arrays
    x_edges = ds.XCOR
    y_edges = ds.YCOR
    if missing_value is not np.nan: # Converting missing_value to nan if it is not nan
        x_edges = x_edges.where(x_edges != missing_value)
        y_edges = y_edges.where(y_edges != missing_value)

    X = x_edges.transpose("NC", "MC").values
    Y = y_edges.transpose("NC", "MC").values

    # Fill over the nans in the cell edges arrays
    X, Y = patch_mesh_arrays(X, Y)

    ds = ds.assign({
        "XCOR": (("NC", "MC"), X),
        "YCOR": (("NC", "MC"), Y),
        })
    return ds


def patch_mesh_arrays(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the filled meshgrid of the X and Y coordinates. Filling occurs by linearly interpolating/extrapolating over nan values.

    Parameters
    ----------
    X : np.ndarray
        Meshgrid array of x coordinates of the cell edges (mesh contains nan values)
    Y : np.ndarray
        Meshgrid array of y coordinates of the cell edges (mesh contains nan values)
    
    Returns
    -------
    X : np.ndarray
        Meshgrid array of x coordinates of the cell edges
    Y : np.ndarray
        Meshgrid array of y coordinates of the cell edges
    """
    # Finding a good refrence in meshgrid for x and y axis
    for row in range(X.shape[0]):
        x = X[row, :].flatten()
        x_valid = ~np.isnan(x)
        if x_valid.sum() >= 2: # ie. more than 2 datapoints to extrapolate between
            break
    else:
        raise ValueError("No valid X coordinates found")
    
    for col in range(Y.shape[0]):
        y = Y[:, col].flatten()
        y_valid = ~np.isnan(y)
        if y_valid.sum() >= 2:
            break
    else:
        raise ValueError("No valid Y coordinates found")
    # return x, y
    x = interp(x)
    y = interp(y)
    X, Y = np.meshgrid(x, y)
    return X, Y

def interp(x: np.ndarray) -> np.ndarray:
    """
    Given a 1D array with nans, takes the first two valid points and uses them to generate a new axis that is linearly interpolated between them.

    Parameters
    ----------
    x : np.ndarray
        1D array with nans
    """
    # Finding two valid points in the array to use for interpolation
    x_points = x[~np.isnan(x)]
    try:
        p1, p2 = x_points[0], x_points[1]
    except IndexError:
        raise IndexError("Not enough valid points to interpolate (need 2)")
    
    # Finding indices for these points
    p1_index = np.where(x == p1)[0]
    p2_index = np.where(x == p2)[0]

    # Determining start of axis and interpolation step
    dx = (p2 - p1) / (p2_index - p1_index)
    x0 = p1 - p1_index * dx
    
    return np.arange(x0, x0 + dx * x.shape[0], dx) # Generating linearly extraplated axis

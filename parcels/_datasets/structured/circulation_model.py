import numpy as np
import pandas as pd
import xarray as xr

from . import T, X, Y, Z

__all__ = ["T", "X", "Y", "Z", "datasets"]


def _nemo_data() -> xr.Dataset:
    """Dataset matching level 0 NEMO model output.

    Example dataset is based off of data from the MOi GLO12 run.

    https://www.mercator-ocean.eu/en/solutions-expertise/accessing-digital-data/product-details/?offer=4217979b-2662-329a-907c-602fdc69c3a3&system=d35404e4-40d3-59d6-3608-581c9495d86a
    """
    # Using data from lorenz.
    # Mesh file: /storage/shared/oceanparcels/input_data/MOi/domain_ORCA0083-N006/SY4V3R1_mesh_hgr.nc
    # Data files: /storage/shared/oceanparcels/input_data/MOi/GLO12/psy4v3r1-daily_{U,V}_*.nc
    # used modulefile for reference: "/storage/shared/oceanparcels/input_data/MOi/psy4v3r1/create_fieldset2D.py"

    # scp "lorenz:/storage/shared/oceanparcels/input_data/MOi/GLO12/psy4v3r1-daily_{U,V,W,T}_2007-01-0{1,2}.nc" data-v4/nemo/field

    time_counter_data = pd.date_range(start="2007-01-01T12:00:00", periods=T, freq="D")
    y_data = np.arange(1, Y + 1)
    x_data = np.arange(1, X + 1)
    deptht_data = np.linspace(0.494, 5.728e03, Z)

    # Create the dataset
    return xr.Dataset(
        data_vars={
            "sotkeavmu1": (
                ("time_counter", "y", "x"),
                np.random.rand(T, Y, X).astype(np.float64),
                {
                    "units": "m2 s-1",
                    "valid_min": np.float64(0.0),
                    "valid_max": np.float64(100.0),
                    "long_name": "Vertical Eddy Viscosity U 1m",
                    "standard_name": "ocean_vertical_eddy_viscosity_u_1m",
                    "short_name": "sotkeavmu1",
                    "online_operation": "N/A",
                    "interval_operation": np.int64(86400),
                    "interval_write": np.int64(86400),
                    "associate": "time_counter nav_lat nav_lon",
                },
            ),
            "sotkeavmu15": (
                ("time_counter", "y", "x"),
                np.random.rand(T, Y, X).astype(np.float64),
                {
                    "units": "m2 s-1",
                    "valid_min": np.float64(0.0),
                    "valid_max": np.float64(100.0),
                    "long_name": "Vertical Eddy Viscosity U 15m",
                    "standard_name": "ocean_vertical_eddy_viscosity_u_15m",
                    "short_name": "sotkeavmu15",
                    "online_operation": "N/A",
                    "interval_operation": np.int64(86400),
                    "interval_write": np.int64(86400),
                    "associate": "time_counter nav_lat nav_lon",
                },
            ),
            "sotkeavmu30": (
                ("time_counter", "y", "x"),
                np.random.rand(T, Y, X).astype(np.float64),
                {
                    "units": "m2 s-1",
                    "valid_min": np.float64(0.0),
                    "valid_max": np.float64(100.0),
                    "long_name": "Vertical Eddy Viscosity U 30m",
                    "standard_name": "ocean_vertical_eddy_viscosity_u_30m",
                    "short_name": "sotkeavmu30",
                    "online_operation": "N/A",
                    "interval_operation": np.int64(86400),
                    "interval_write": np.int64(86400),
                    "associate": "time_counter nav_lat nav_lon",
                },
            ),
            "sotkeavmu50": (
                ("time_counter", "y", "x"),
                np.random.rand(T, Y, X).astype(np.float64),
                {
                    "units": "m2 s-1",
                    "valid_min": np.float64(0.0),
                    "valid_max": np.float64(100.0),
                    "long_name": "Vertical Eddy Viscosity U 50m",
                    "standard_name": "ocean_vertical_eddy_viscosity_u_50m",
                    "short_name": "sotkeavmu50",
                    "online_operation": "N/A",
                    "interval_operation": np.int64(86400),
                    "interval_write": np.int64(86400),
                    "associate": "time_counter nav_lat nav_lon",
                },
            ),
            "vozocrtx": (
                ("time_counter", "deptht", "y", "x"),
                np.random.rand(T, Z, Y, X).astype(np.float64),
                {
                    "units": "m s-1",
                    "valid_min": np.float64(-10.0),
                    "valid_max": np.float64(10.0),
                    "long_name": "Zonal velocity",
                    "standard_name": "sea_water_x_velocity",
                    "short_name": "vozocrtx",
                    "online_operation": "N/A",
                    "interval_operation": np.int64(86400),
                    "interval_write": np.int64(86400),
                    "associate": "time_counter deptht nav_lat nav_lon",
                },
            ),
        },
        coords={
            "nav_lon": (
                ("y", "x"),
                np.random.rand(Y, X).astype(np.float32),
                {
                    "units": "degrees_east",
                    "valid_min": np.float32(-179.99984754002182),
                    "valid_max": np.float32(179.999842386314),
                    "long_name": "Longitude",
                    "nav_model": "Default grid",
                    "standard_name": "longitude",
                },
            ),
            "nav_lat": (
                ("y", "x"),
                np.random.rand(Y, X).astype(np.float32),
                {
                    "units": "degrees_north",
                    "valid_min": np.float32(-77.0104751586914),
                    "valid_max": np.float32(89.9591064453125),
                    "long_name": "Latitude",
                    "nav_model": "Default grid",
                    "standard_name": "latitude",
                },
            ),
            "x": (("x",), x_data, {"standard_name": "projection_x_coordinate", "axis": "X", "units": "1"}),
            "y": (("y",), y_data, {"standard_name": "projection_y_coordinate", "axis": "Y", "units": "1"}),
            "time_counter": (
                ("time_counter",),
                time_counter_data,
                {"standard_name": "time", "long_name": "Time axis", "axis": "T", "time_origin": "1950-JAN-01 00:00:00"},
            ),
            "deptht": (
                ("deptht",),
                deptht_data,
                {
                    "units": "m",
                    "positive": "down",
                    "valid_min": np.float64(0.4940253794193268),
                    "valid_max": np.float64(5727.91650390625),
                    "long_name": "Vertical T levels",
                    "standard_name": "depth",
                    "axis": "Z",
                },
            ),
        },
        attrs={
            "Conventions": "CF-1.0",
            "file_name": "ORCA12_LIM-T00_y2021m09d27_gridU.nc",
            "institution": "MERCATOR OCEAN",
            "source": "NEMO",
            "TimeStamp": "2021-OCT-03 18:27:01 GMT-0000",
            "references": "http://www.mercator-ocean.eu",
        },
    )


def _hycom_data() -> xr.Dataset:
    """Dataset matching level 0 HYCOM model output.

    Example dataset is based off of data from the GOFS 3.1: 41-layer HYCOM + NCODA Global 1/12° Analysis.

    https://www.hycom.org/dataserver/gofs-3pt1/analysis
    """
    ...


def _mitgcm_data() -> xr.Dataset:
    """Dataset matching level 0 MITgcm model output.

    Example dataset is based on the Pre-SWOT Level-4 Hourly MITgcm LLC4320 simulation,
    which provides high-resolution (1/48°) global ocean state estimates with hourly outputs.

    https://podaac.jpl.nasa.gov/dataset/MITgcm_LLC4320_Pre-SWOT_JPL_L4_ACC_SMST_v1.0
    """
    ...


def _pop_data() -> xr.Dataset:
    """Dataset matching level 0 POP model output.

    TODO: Identify a suitable public dataset to mimick.
    """
    ...


def _ecco_data() -> xr.Dataset:
    """Dataset matching level 0 ECCO model output.

    TODO: Identify a suitable public dataset to mimick.

    """
    ...


def _croco_data() -> xr.Dataset:
    """Dataset matching level 0 CROCO model output.

    TODO: Identify a suitable public dataset to mimick.
    """
    ...


datasets = {}

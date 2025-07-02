"""Datasets mimicking the layout of real-world hydrodynamic models"""

import numpy as np
import xarray as xr

from . import T, X, Y, Z

__all__ = ["T", "X", "Y", "Z", "datasets"]

TIME = xr.date_range("2000", "2001", T)


datasets = {
    "ds_copernicusmarine": xr.Dataset(
        # Copernicus Marine Service dataset as retrieved by the `copernicusmarine` toolkit
        {
            "uo": (
                ["time", "depth", "latitude", "longitude"],
                np.random.rand(T, Z, Y, X),
                {
                    "valid_max": 5.0,
                    "unit_long": "Meters per second",
                    "units": "m s-1",
                    "long_name": "Eastward velocity",
                    "standard_name": "eastward_sea_water_velocity",
                    "valid_min": -5.0,
                },
            ),
            "vo": (
                ["time", "depth", "latitude", "longitude"],
                np.random.rand(T, Z, Y, X),
                {
                    "valid_max": 5.0,
                    "unit_long": "Meters per second",
                    "units": "m s-1",
                    "long_name": "Northward velocity",
                    "standard_name": "northward_sea_water_velocity",
                    "valid_min": -5.0,
                },
            ),
        },
        coords={
            "depth": (
                ["depth"],
                np.linspace(0.49, 5727.92, Z),
                {
                    "unit_long": "Meters",
                    "units": "m",
                    "axis": "Z",
                    "long_name": "depth",
                    "standard_name": "depth",
                    "positive": "down",
                },
            ),
            "latitude": (
                ["latitude"],
                np.linspace(-90, 90, Y),
                {
                    "unit_long": "Degrees North",
                    "units": "degrees_north",
                    "axis": "Y",
                    "long_name": "Latitude",
                    "standard_name": "latitude",
                },
            ),
            "longitude": (
                ["longitude"],
                np.linspace(-180, 180, X),
                {
                    "unit_long": "Degrees East",
                    "units": "degrees_east",
                    "axis": "X",
                    "long_name": "Longitude",
                    "standard_name": "longitude",
                },
            ),
            "time": (
                ["time"],
                TIME,
                {
                    "unit_long": "Hours Since 1950-01-01",
                    "axis": "T",
                    "long_name": "Time",
                    "standard_name": "time",
                },
            ),
        },
    ),
    "ds_NEMO_MOI_U": xr.Dataset(
        # NEMO model dataset (U component) as serviced by Mercator Ocean International
        {
            "vozocrtx": (
                ["deptht", "y", "x"],
                np.random.rand(Z, Y, X, dtype="float64"),
                {
                    "units": "m s-1",
                    "valid_min": -10.0,
                    "valid_max": 10.0,
                    "long_name": "Zonal velocity",
                    "unit_long": "Meters per second",
                    "standard_name": "sea_water_x_velocity",
                    "short_name": "vozocrtx",
                    "online_operation": "N/A",
                    "interval_operation": 86400,
                    "interval_write": 86400,
                    "associate": "time_counter deptht nav_lat nav_lon",
                },
            ),
            "sotkeavmu1": (
                ["y", "x"],
                np.random.rand(Y, X, dtype="float64"),
                {
                    "units": "m2 s-1",
                    "valid_min": 0.0,
                    "valid_max": 100.0,
                    "long_name": "Vertical Eddy Viscosity U 1m",
                    "standard_name": "ocean_vertical_eddy_viscosity_u_1m",
                    "short_name": "sotkeavmu1",
                    "online_operation": "N/A",
                    "interval_operation": 86400,
                    "interval_write": 86400,
                    "associate": "time_counter nav_lat nav_lon",
                },
            ),
        },
        coords={
            "nav_lon": (
                ["y, x"],
                np.tile(np.linspace(-179, 179, X, endpoint=False), (Y, 1)),  # note that this is not curvilinear
                {
                    "units": "degrees_east",
                    "valid_min": -179.99984754002182,
                    "valid_max": 179.999842386314,
                    "long_name": "Longitude",
                    "nav_model": "Default grid",
                    "standard_name": "longitude",
                },
            ),
            "nav_lat": (
                ["y, x"],
                np.tile(np.linspace(-75, 85, Y).reshape(-1, 1), (1, X)),  # note that this is not curvilinear
                {
                    "units": "degrees_north",
                    "valid_min": -77.0104751586914,
                    "valid_max": 89.9591064453125,
                    "long_name": "Latitude",
                    "nav_model": "Default grid",
                    "standard_name": "latitude",
                },
            ),
            "x": (
                ["x"],
                np.arange(X, dtype="int32"),
                {
                    "standard_name": "projection_x_coordinate",
                    "axis": "X",
                    "units": "1",
                },
            ),
            "y": (
                ["y"],
                np.arange(Y, dtype="int32"),
                {
                    "standard_name": "projection_y_coordinate",
                    "axis": "Y",
                    "units": "1",
                },
            ),
            "time_counter": (
                [],
                np.empty(0, dtype="datetime64[ns]"),
                {
                    "standard_name": "time",
                    "long_name": "Time axis",
                    "axis": "T",
                    "time_origin": "1950-JAN-01 00:00:00",
                },
            ),
            "deptht": (
                ["deptht"],
                np.linspace(1, 5500, Z, dtype="float64"),
                {
                    "units": "m",
                    "positive": "down",
                    "valid_min": 0.4940253794193268,
                    "valid_max": 5727.91650390625,
                    "long_name": "Vertical T levels",
                    "standard_name": "depth",
                    "axis": "Z",
                },
            ),
        },
    ),
    "ds_NEMO_MOI_V": xr.Dataset(
        # NEMO model dataset (V component) as serviced by Mercator Ocean International
        {
            "vomecrty": (
                ["deptht", "y", "x"],
                np.random.rand(Z, Y, X, dtype="float64"),
                {
                    "units": "m s-1",
                    "valid_min": -10.0,
                    "valid_max": 10.0,
                    "long_name": "Meridional velocity",
                    "unit_long": "Meters per second",
                    "standard_name": "sea_water_y_velocity",
                    "short_name": "vomecrty",
                    "online_operation": "N/A",
                    "interval_operation": 86400,
                    "interval_write": 86400,
                    "associate": "time_counter deptht nav_lat nav_lon",
                },
            ),
        },
        coords={
            "nav_lon": (
                ["y, x"],
                np.tile(np.linspace(-179, 179, X, endpoint=False), (Y, 1)),  # note that this is not curvilinear
                {
                    "units": "degrees_east",
                    "valid_min": -179.99984754002182,
                    "valid_max": 179.999842386314,
                    "long_name": "Longitude",
                    "nav_model": "Default grid",
                    "standard_name": "longitude",
                },
            ),
            "nav_lat": (
                ["y, x"],
                np.tile(np.linspace(-75, 85, Y).reshape(-1, 1), (1, X)),  # note that this is not curvilinear
                {
                    "units": "degrees_north",
                    "valid_min": -77.0104751586914,
                    "valid_max": 89.9591064453125,
                    "long_name": "Latitude",
                    "nav_model": "Default grid",
                    "standard_name": "latitude",
                },
            ),
            "x": (
                ["x"],
                np.arange(X, dtype="int32"),
                {
                    "standard_name": "projection_x_coordinate",
                    "axis": "X",
                    "units": "1",
                },
            ),
            "y": (
                ["y"],
                np.arange(Y, dtype="int32"),
                {
                    "standard_name": "projection_y_coordinate",
                    "axis": "Y",
                    "units": "1",
                },
            ),
            "time_counter": (
                [],
                np.empty(0, dtype="datetime64[ns]"),
                {
                    "standard_name": "time",
                    "long_name": "Time axis",
                    "axis": "T",
                    "time_origin": "1950-JAN-01 00:00:00",
                },
            ),
            "deptht": (
                ["deptht"],
                np.linspace(1, 5500, Z, dtype="float64"),
                {
                    "units": "m",
                    "positive": "down",
                    "valid_min": 0.4940253794193268,
                    "valid_max": 5727.91650390625,
                    "long_name": "Vertical T levels",
                    "standard_name": "depth",
                    "axis": "Z",
                },
            ),
        },
    ),
    "ds_CESM": xr.Dataset(
        # CESM model dataset
        {
            "UVEL": (
                ["time", "z_t", "nlat", "nlon"],
                np.random.rand(T, Z, Y, X, dtype="float32"),
                {
                    "long_name": "Velocity in grid-x direction",
                    "units": "centimeter/s",
                    "grid_loc": 3221,
                    "cell_methods": "time:mean",
                },
            ),
            "VVEL": (
                ["time", "z_t", "nlat", "nlon"],
                np.random.rand(T, Z, Y, X, dtype="float32"),
                {
                    "long_name": "Velocity in grid-y direction",
                    "units": "centimeter/s",
                    "grid_loc": 3221,
                    "cell_methods": "time:mean",
                },
            ),
            "WVEL": (
                ["time", "z_w_top", "nlat", "nlon"],
                np.random.rand(T, Z, Y, X, dtype="float32"),
                {
                    "long_name": "Vertical Velocity",
                    "units": "centimeter/s",
                    "grid_loc": 3112,
                    "cell_methods": "time:mean",
                },
            ),
        },
        coords={
            "time": (
                ["time"],
                TIME,
                {
                    "long_name": "time",
                    "bounds": "time_bounds",
                },
            ),
            "z_t": (
                ["z_t"],
                np.linspace(0, 5000, Z, dtype="float32"),
                {
                    "long_name": "depth from surface to midpoint of layer",
                    "units": "centimeters",
                    "positive": "down",
                    "valid_min": 500.0,
                    "valid_max": 537500.0,
                },
            ),
            "z_w_top": (
                ["z_w_top"],
                np.linspace(0, 5000, Z, dtype="float32"),
                {
                    "long_name": "depth from surface to top of layer",
                    "units": "centimeters",
                    "positive": "down",
                    "valid_min": 0.0,
                    "valid_max": 525000.94,
                },
            ),
            "ULONG": (
                ["nlat", "nlon"],
                np.tile(np.linspace(-179, 179, X, endpoint=False), (Y, 1)),  # note that this is not curvilinear
                {
                    "long_name": "array of u-grid longitudes",
                    "units": "degrees_east",
                },
            ),
            "ULAT": (
                ["nlat", "nlon"],
                np.tile(np.linspace(-75, 85, Y).reshape(-1, 1), (1, X)),  # note that this is not curvilinear
                {
                    "long_name": "array of u-grid latitudes",
                    "units": "degrees_north",
                },
            ),
        },
    ),
    "ds_ERA5_wind": xr.Dataset(
        # ERA5 10m wind model dataset
        {
            "u10": (
                ["time", "latitude", "longitude"],
                np.random.rand(T, Y, X, dtype="float32"),
                {
                    "long_name": "10 metre U wind component",
                    "units": "m s**-1",
                },
            ),
            "v10": (
                ["time", "latitude", "longitude"],
                np.random.rand(T, Y, X, dtype="float32"),
                {
                    "long_name": "10 metre V wind component",
                    "units": "m s**-1",
                },
            ),
        },
        coords={
            "time": (
                ["time"],
                TIME,
                {
                    "long_name": "time",
                },
            ),
            "latitude": (
                ["latitude"],
                np.linspace(90, -90, Y),  # Note: ERA5 uses latitudes from 90 to -90
                {
                    "long_name": "latitude",
                    "units": "degrees_north",
                },
            ),
            "longitude": (
                ["longitude"],
                np.linspace(0, 360, X, endpoint=False),
                {
                    "long_name": "longitude",
                    "units": "degrees_east",
                },
            ),
        },
    ),
    "ds_FES_tides": xr.Dataset(
        # FES tidal model dataset
        {
            "Ug": (
                ["lat", "lon"],
                np.random.rand(Y, X, dtype="float32"),
                {
                    "long_name": "Eastward sea water velocity phaselag due to non equilibrium ocean tide at m2 frequency",
                    "units": "degrees",
                    "grid_mapping": "crs",
                },
            ),
            "Ua": (
                ["lat", "lon"],
                np.random.rand(Y, X, dtype="float32"),
                {
                    "long_name": "Eastward sea water velocity amplitude due to non equilibrium ocean tide at m2 frequency",
                    "units": "cm/s",
                    "grid_mapping": "crs",
                },
            ),
        },
        coords={
            "lat": (
                ["lat"],
                np.linspace(-90, 90, Y),
                {
                    "long_name": "latitude",
                    "units": "degrees_north",
                    "bounds": "lat_bnds",
                    "axis": "Y",
                    "valid_min": -90.0,
                    "valid_max": 90.0,
                },
            ),
            "lon": (
                ["lon"],
                np.linspace(0, 360, X, endpoint=False),
                {
                    "long_name": "longitude",
                    "units": "degrees_east",
                    "bounds": "lon_bnds",
                    "axis": "X",
                    "valid_min": 0.0,
                    "valid_max": 360.0,
                },
            ),
        },
    ),
    "ds_hycom_espc": xr.Dataset(
        # HYCOM ESPC model dataset from https://data.hycom.org/datasets/ESPC-D-V02/data/daily_netcdf/2025/
        {
            "water_u": (
                ["time", "depth", "lat", "lon"],
                np.random.rand(T, Z, Y, X, dtype="float32"),
                {
                    "long_name": "Eastward Water Velocity",
                    "standard_name": "eastward_sea_water_velocity",
                    "units": "m/s",
                    "NAVO_code": 17,
                    "actual_range": [-3.3700001, 3.6840003],
                    "cell_methods": "time: mean",
                },
            ),
            "tau": (
                ["time"],
                np.arange(0, 24, T, dtype="float64"),
                {
                    "long_name": "Tau",
                    "units": "hours since analysis",
                    "time_origin": "2024-12-31 12:00:00",
                    "NAVO_code": 56,
                    "cell_methods": "time: mean",
                },
            ),
        },
        coords={
            "time": (
                ["time"],
                np.arange(0, T, dtype="float64"),
                {
                    "long_name": "Valid Time",
                    "units": "hours since 2000-01-01 00:00:00",
                    "time_origin": "2000-01-01 00:00:00",
                    "calendar": "standard",
                    "axis": "T",
                    "NAVO_code": 13,
                    "cell_methods": "time: mean",
                },
            ),
            "depth": (
                ["depth"],
                np.linspace(0, 5000, Z, dtype="float32"),
                {
                    "long_name": "Depth",
                    "standard_name": "depth",
                    "units": "m",
                    "positive": "down",
                    "axis": "Z",
                    "NAVO_code": 5,
                },
            ),
            "lat": (
                ["lat"],
                np.linspace(-80, 90, Y),
                {
                    "long_name": "Latitude",
                    "standard_name": "latitude",
                    "units": "degrees_north",
                    "point_spacing": "even",
                    "axis": "Y",
                    "NAVO_code": 1,
                },
            ),
            "lon": (
                ["lon"],
                np.linspace(0, 360, X, endpoint=False),
                {
                    "long_name": "Longitude",
                    "standard_name": "longitude",
                    "units": "degrees_east",
                    "modulo": "360 degrees",
                    "axis": "X",
                    "NAVO_code": 2,
                },
            ),
        },
    ),
    "ds_CROCO_idealized": xr.Dataset(
        # CROCO idealized model dataset
        {
            "u": (
                ["time", "s_rho", "eta_rho", "xi_u"],
                np.random.rand(T, Z, Y, X - 1, dtype="float32"),
                {
                    "long_name": "u-momentum component",
                    "units": "meter second-1",
                    "field": "u-velocity, scalar, series",
                    "standard_name": "sea_water_x_velocity_at_u_location",
                },
            ),
            "v": (
                ["time", "s_rho", "eta_v", "xi_rho"],
                np.random.rand(T, Z, Y - 1, X, dtype="float32"),
                {
                    "long_name": "v-momentum component",
                    "units": "meter second-1",
                    "field": "v-velocity, scalar, series",
                    "standard_name": "sea_water_y_velocity_at_v_location",
                },
            ),
            "w": (
                ["time", "s_rho", "eta_rho", "xi_rho"],
                np.random.rand(T, Z, Y, X, dtype="float32"),
                {
                    "long_name": "vertical momentum component",
                    "units": "meter second-1",
                    "field": "w-velocity, scalar, series",
                    "standard_name": "upward_sea_water_velocity",
                    "coordinates": "lat_rho lon_rho",
                },
            ),
            "h": (
                ["eta_rho", "xi_rho"],
                np.random.rand(Y, X, dtype="float32"),
                {
                    "long_name": "bathymetry at RHO-points",
                    "units": "meter",
                    "field": "bath, scalar",
                    "standard_name": "model_sea_floor_depth_below_geoid",
                },
            ),
            "zeta": (
                ["time", "eta_rho", "xi_rho"],
                np.random.rand(T, Y, X, dtype="float32"),
                {
                    "long_name": "free-surface",
                    "units": "meter",
                    "field": "free_surface, scalar, series",
                    "standard_name": "sea_surface_height",
                },
            ),
            "Cs_w": (
                ["s_w"],
                np.random.rand(Z + 1, dtype="float32"),
                {
                    "long_name": "S-coordinate stretching curves at W-points",
                },
            ),
            "hc": (
                [],
                np.array(0.0, dtype="float32"),
                {
                    "long_name": "S-coordinate parameter, critical depth",
                    "units": "meter",
                },
            ),
        },
        coords={
            "time": (
                ["time"],
                np.arange(0, T, dtype="float64"),
                {
                    "long_name": "time since initialization",
                    "units": "second",
                    "field": "time, scalar, series",
                    "standard_name": "time",
                    "axis": "T",
                },
            ),
            "s_rho": (
                ["s_rho"],
                np.linspace(-0.95, 0.05, Z, dtype="float32"),
                {
                    "long_name": "S-coordinate at RHO-points",
                    "standard_name": "ocean_s_coordinate_g1",
                    "positive": "up",
                    "axis": "Z",
                    "formula_terms": "s: sc_r C: Cs_r eta: zeta depth: h depth_c: hc",
                },
            ),
            "s_w": (
                ["s_w"],
                np.linspace(-1, 0, Z + 1, dtype="float32"),
                {
                    "long_name": "S-coordinate at W-points",
                    "standard_name": "ocean_s_coordinate_g1_at_w_location",
                    "positive": "up",
                    "axis": "Z",
                    "c_grid_axis_shift": -0.5,
                    "formula_terms": "s: sc_w C: Cs_w eta: zeta depth: h depth_c: hc",
                },
            ),
            "eta_rho": (
                ["eta_rho"],
                np.arange(Y, dtype="float32"),
                {
                    "long name": "y-dimension of the grid",
                    "standard_name": "y_grid_index",
                    "axis": "Y",
                    "c_grid_dynamic_range": f"2:{Y}",
                },
            ),
            "eta_v": (
                ["eta_v"],
                np.arange(Y - 1, dtype="float32"),
                {
                    "long name": "y-dimension of the grid at v location",
                    "standard_name": "y_grid_index_at_v_location",
                    "axis": "Y",
                    "c_grid_axis_shift": 0.5,
                    "c_grid_dynamic_range": f"2:{Y-1}",
                },
            ),
            "xi_rho": (
                ["xi_rho"],
                np.arange(X, dtype="float32"),
                {
                    "long name": "x-dimension of the grid",
                    "standard_name": "x_grid_index",
                    "axis": "X",
                    "c_grid_dynamic_range": f"2:{X}",
                },
            ),
            "xi_u": (
                ["xi_u"],
                np.arange(X - 1, dtype="float32"),
                {
                    "long name": "x-dimension of the grid at u location",
                    "standard_name": "x_grid_index_at_u_location",
                    "axis": "X",
                    "c_grid_axis_shift": 0.5,
                    "c_grid_dynamic_range": f"2:{X-1}",
                },
            ),
            "x_rho": (
                ["eta_rho", "xi_rho"],
                np.tile(np.linspace(-179, 179, X, endpoint=False), (Y, 1)),  # note that this is not curvilinear
                {
                    "long_name": "x-locations of RHO-points",
                    "units": "meter",
                    "standard_name": "plane_x_coordinate",
                    "field": "x_rho, scalar",
                },
            ),
            "y_rho": (
                ["eta_rho", "xi_rho"],
                np.tile(np.linspace(-89, 89, Y), (X, 1)).T,  # note that this is not curvilinear
                {
                    "long_name": "y-locations of RHO-points",
                    "units": "meter",
                    "standard_name": "plane_y_coordinate",
                    "field": "y_rho, scalar",
                },
            ),
        },
    ),
}

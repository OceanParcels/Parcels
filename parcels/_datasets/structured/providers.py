import numpy as np
import pandas as pd
import xarray as xr


def dummy_ofam_dataset():
    """Based off of the metadata in "OFAM_example_data" dataset."""
    # Create dimension coordinates
    time = pd.date_range(start="1993-01-01 12:00:00", periods=4, freq="D")
    nv = np.array([1.0, 2.0])
    st_edges_ocean = np.concatenate(
        [
            np.array([0.0, 5.0]),  # First two values shown
            np.linspace(5.0, 5000.0, 50),  # Remaining values to reach 52 total
        ]
    )
    depth = np.array([2.5])
    lon = np.arange(100.0, 300.1, 0.1)  # 2001 points from 100.0 to 300.0
    lat = np.arange(-30.0, 30.1, 0.1)  # 601 points from -30.0 to 30.0

    tdim, xdim, ydim, zdim = len(time), len(lon), len(lat), len(depth)

    U = np.random.randn(tdim, zdim, ydim, xdim)
    U = U.astype(np.float32)
    V = np.random.randn(tdim, zdim, ydim, xdim)
    V = V.astype(np.float32)

    return xr.Dataset(
        {
            "Time_bounds": (["Time", "nv"], np.zeros((tdim, len(nv)), dtype="timedelta64[ns]")),
            "average_DT": (
                ["Time"],
                np.zeros(tdim, dtype="timedelta64[ns]"),
                {"long_name": "Length of average period"},
            ),
            "average_T1": (["Time"], time.copy(), {"long_name": "Start time for average period"}),
            "average_T2": (["Time"], time.copy(), {"long_name": "End time for average period"}),
            "u": (["Time", "st_ocean", "yu_ocean", "xu_ocean"], U),
            "v": (["Time", "st_ocean", "yu_ocean", "xu_ocean"], V),
        },
        coords={
            "Time": (
                "Time",
                time,
                {"long_name": "Time", "cartesian_axis": "T", "calendar_type": "GREGORIAN", "bounds": "Time_bounds"},
            ),
            "nv": ("nv", nv, {"long_name": "vertex number", "units": "none", "cartesian_axis": "N"}),
            "st_edges_ocean": (
                "st_edges_ocean",
                st_edges_ocean,
                {"long_name": "tcell zstar depth edges", "units": "meters", "cartesian_axis": "Z", "positive": "down"},
            ),
            "st_ocean": (
                "st_ocean",
                depth,
                {
                    "long_name": "tcell zstar depth",
                    "units": "meters",
                    "cartesian_axis": "Z",
                    "positive": "down",
                    "edges": "st_edges_ocean",
                },
            ),
            "xu_ocean": (
                "xu_ocean",
                lon,
                {"long_name": "ucell longitude", "units": "degrees_E", "cartesian_axis": "X"},
            ),
            "yu_ocean": ("yu_ocean", lat, {"long_name": "ucell latitude", "units": "degrees_N", "cartesian_axis": "Y"}),
        },
        attrs={
            "grid_type": "regular",
            "grid_tile": "N/A",
            "nco_openmp_thread_number": 1,
            "NCO": '"4.5.4"',
        },
    )

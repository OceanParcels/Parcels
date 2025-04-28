import os

import xarray as xr

import parcels


def create_fieldset():
    example_dataset_folder = parcels.download_example_dataset("CROCOidealized_data")
    file = os.path.join(example_dataset_folder, "CROCO_idealized.nc")

    variables = {"U": "u", "V": "v", "W": "w", "H": "h", "Zeta": "zeta", "Cs_w": "Cs_w"}
    dimensions = {
        "U": {"lon": "x_rho", "lat": "y_rho", "depth": "s_w", "time": "time"},
        "V": {"lon": "x_rho", "lat": "y_rho", "depth": "s_w", "time": "time"},
        "W": {"lon": "x_rho", "lat": "y_rho", "depth": "s_w", "time": "time"},
        "H": {"lon": "x_rho", "lat": "y_rho"},
        "Zeta": {"lon": "x_rho", "lat": "y_rho", "time": "time"},
        "Cs_w": {"depth": "s_w"},
    }
    fieldset = parcels.FieldSet.from_croco(
        file,
        variables,
        dimensions,
        mesh="flat",
        hc=xr.open_dataset(file).hc.values,
    )

    return fieldset

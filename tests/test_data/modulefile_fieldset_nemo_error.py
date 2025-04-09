import os

import parcels


def random_function_name():
    data_path = os.path.join(os.path.dirname(__file__))

    filenames = {
        "U": {
            "lon": os.path.join(data_path, "mask_nemo_cross_180lon.nc"),
            "lat": os.path.join(data_path, "mask_nemo_cross_180lon.nc"),
            "data": os.path.join(data_path, "Uu_eastward_nemo_cross_180lon.nc"),
        },
        "V": {
            "lon": os.path.join(data_path, "mask_nemo_cross_180lon.nc"),
            "lat": os.path.join(data_path, "mask_nemo_cross_180lon.nc"),
            "data": os.path.join(data_path, "Vv_eastward_nemo_cross_180lon.nc"),
        },
    }
    variables = {"U": "U", "V": "V"}
    dimensions = {"lon": "glamf", "lat": "gphif", "time": "time_counter"}
    return parcels.FieldSet.from_nemo(filenames, variables, dimensions)


def none_returning_function():
    return None

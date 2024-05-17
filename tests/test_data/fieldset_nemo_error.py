import parcels
from os import path


def random_function_name():
    data_path = path.join(path.dirname(__file__))

    filenames = {'U': {'lon': path.join(data_path, 'mask_nemo_cross_180lon.nc'),
                       'lat': path.join(data_path, 'mask_nemo_cross_180lon.nc'),
                       'data': path.join(data_path, 'Uu_eastward_nemo_cross_180lon.nc')},
                 'V': {'lon': path.join(data_path, 'mask_nemo_cross_180lon.nc'),
                       'lat': path.join(data_path, 'mask_nemo_cross_180lon.nc'),
                       'data': path.join(data_path, 'Vv_eastward_nemo_cross_180lon.nc')}}
    variables = {'U': 'U', 'V': 'V'}
    dimensions = {'lon': 'glamf', 'lat': 'gphif', 'time': 'time_counter'}
    return parcels.FieldSet.from_nemo(filenames, variables, dimensions)

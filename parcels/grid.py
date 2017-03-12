from parcels.fieldset import FieldSet
from parcels.loggers import logger


__all__ = ['Grid']


class Grid(object):

    def __init__(self, U, V, allow_time_extrapolation=False, fields={}):
        FieldSet(U, V, allow_time_extrapolation, fields)

    @classmethod
    def from_data(cls, data_u, lon_u, lat_u, data_v, lon_v, lat_v,
                  depth=None, time=None, field_data={}, transpose=True,
                  mesh='spherical', allow_time_extrapolation=True, **kwargs):
        logger.warning("`Grid` has been renamed to `FieldSet`, please update your code")
        return FieldSet.from_data(data_u, lon_u, lat_u, data_v, lon_v, lat_v,
                                  depth, time, field_data, transpose,
                                  mesh, allow_time_extrapolation, **kwargs)

    @classmethod
    def from_netcdf(cls, filenames, variables, dimensions, indices={},
                    mesh='spherical', allow_time_extrapolation=False, **kwargs):
        logger.warning("`Grid` has been renamed to `FieldSet`, please update your code")
        return FieldSet.from_netcdf(filenames, variables, dimensions, indices,
                                    mesh, allow_time_extrapolation, **kwargs)

    @classmethod
    def from_nemo(cls, basename, uvar='vozocrtx', vvar='vomecrty',
                  indices={}, extra_vars={}, allow_time_extrapolation=False, **kwargs):
        logger.warning("`Grid` has been renamed to `FieldSet`, please update your code")
        return FieldSet.from_nemo(basename, uvar, vvar,
                                  indices, extra_vars, allow_time_extrapolation, **kwargs)

    def fields(self):
        logger.warning("`Grid` has been renamed to `FieldSet`, please update your code")
        return FieldSet.fields()

    def add_field(self, field):
        logger.warning("`Grid` has been renamed to `FieldSet`, please update your code")
        return FieldSet.add_field(field)

    def add_constant(self, name, value):
        logger.warning("`Grid` has been renamed to `FieldSet`, please update your code")
        return FieldSet.add_constant(name, value)

    def add_periodic_halo(self, zonal=False, meridional=False, halosize=5):
        logger.warning("`Grid` has been renamed to `FieldSet`, please update your code")
        return FieldSet.add_periodic_halo(zonal, meridional, halosize)

    def eval(self, x, y):
        logger.warning("`Grid` has been renamed to `FieldSet`, please update your code")
        return FieldSet.eval(x, y)

    def write(self, filename):
        logger.warning("`Grid` has been renamed to `FieldSet`, please update your code")
        return FieldSet.write(filename)

    def advancetime(self, fieldset_new):
        logger.warning("`Grid` has been renamed to `FieldSet`, please update your code")
        return FieldSet.advancetime(fieldset_new)

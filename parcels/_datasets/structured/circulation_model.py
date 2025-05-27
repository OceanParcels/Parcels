import xarray as xr

from . import T, X, Y, Z

__all__ = ["T", "X", "Y", "Z", "datasets"]


def _nemo_data() -> xr.Dataset:
    """Dataset matching level 0 NEMO model output.

    Example dataset is based off of data from the MOi GLO12 run.

    https://www.mercator-ocean.eu/en/solutions-expertise/accessing-digital-data/product-details/?offer=4217979b-2662-329a-907c-602fdc69c3a3&system=d35404e4-40d3-59d6-3608-581c9495d86a
    """
    ...


def _hycom_data() -> xr.Dataset:
    """Dataset matching level 0 HYCOM model output.

    Example dataset is based off of data from the GOFS 3.1: 41-layer HYCOM + NCODA Global 1/12° Analysis.

    https://www.hycom.org/dataserver/gofs-3pt1/analysis
    """
    ...


def _mitgcm_data() -> xr.Dataset: ...


def _pop_data() -> xr.Dataset: ...


def _echo_data() -> xr.Dataset: ...


def _croco_data() -> xr.Dataset: ...


datasets = {}

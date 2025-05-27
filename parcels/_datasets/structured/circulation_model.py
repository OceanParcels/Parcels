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


def _echo_data() -> xr.Dataset:
    """Dataset matching level 0 ECHO model output.

    TODO: Identify a suitable public dataset to mimick.

    """
    ...


def _croco_data() -> xr.Dataset:
    """Dataset matching level 0 CROCO model output.

    TODO: Identify a suitable public dataset to mimick.
    """
    ...


datasets = {}

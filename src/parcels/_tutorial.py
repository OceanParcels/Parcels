import os
from datetime import datetime, timedelta
from pathlib import Path

import pooch
import xarray as xr

from parcels._v3to4 import patch_dataset_v4_compat

__all__ = ["download_example_dataset", "list_example_datasets"]

# When modifying existing datasets in a backwards incompatible way,
# make a new release in the repo and update the DATA_REPO_TAG to the new tag
DATA_REPO_TAG = "main"

DATA_URL = f"https://github.com/OceanParcels/parcels-data/raw/{DATA_REPO_TAG}/data"

# Keys are the dataset names. Values are the filenames in the dataset folder. Note that
# you can specify subfolders in the dataset folder putting slashes in the filename list.
# e.g.,
# "my_dataset": ["file0.nc", "folder1/file1.nc", "folder2/file2.nc"]
# my_dataset/
# ├── file0.nc
# ├── folder1/
# │   └── file1.nc
# └── folder2/
#     └── file2.nc
#
# See instructions at https://github.com/OceanParcels/parcels-data for adding new datasets
EXAMPLE_DATA_FILES: dict[str, list[str]] = {
    "MovingEddies_data": [
        "moving_eddiesP.nc",
        "moving_eddiesU.nc",
        "moving_eddiesV.nc",
    ],
    "MITgcm_example_data": ["mitgcm_UV_surface_zonally_reentrant.nc"],
    "OFAM_example_data": ["OFAM_simple_U.nc", "OFAM_simple_V.nc"],
    "Peninsula_data": [
        "peninsulaU.nc",
        "peninsulaV.nc",
        "peninsulaP.nc",
        "peninsulaT.nc",
    ],
    "GlobCurrent_example_data": [
        f"{date.strftime('%Y%m%d')}000000-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc"
        for date in ([datetime(2002, 1, 1) + timedelta(days=x) for x in range(0, 365)] + [datetime(2003, 1, 1)])
    ],
    "CopernicusMarine_data_for_Argo_tutorial": [
        "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_uo-vo_31.00E-33.00E_33.00S-30.00S_0.49-2225.08m_2024-01-01-2024-02-01.nc",
        "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m_thetao_31.00E-33.00E_33.00S-30.00S_0.49-2225.08m_2024-01-01-2024-02-01.nc",
    ],
    "DecayingMovingEddy_data": [
        "decaying_moving_eddyU.nc",
        "decaying_moving_eddyV.nc",
    ],
    "FESOM_periodic_channel": [
        "fesom_channel.nc",
        "u.fesom_channel.nc",
        "v.fesom_channel.nc",
        "w.fesom_channel.nc",
    ],
    "NemoCurvilinear_data": [
        "U_purely_zonal-ORCA025_grid_U.nc4",
        "V_purely_zonal-ORCA025_grid_V.nc4",
        "mesh_mask.nc4",
    ],
    "NemoNorthSeaORCA025-N006_data": [
        "ORCA025-N06_20000104d05U.nc",
        "ORCA025-N06_20000109d05U.nc",
        "ORCA025-N06_20000114d05U.nc",
        "ORCA025-N06_20000119d05U.nc",
        "ORCA025-N06_20000124d05U.nc",
        "ORCA025-N06_20000129d05U.nc",
        "ORCA025-N06_20000104d05V.nc",
        "ORCA025-N06_20000109d05V.nc",
        "ORCA025-N06_20000114d05V.nc",
        "ORCA025-N06_20000119d05V.nc",
        "ORCA025-N06_20000124d05V.nc",
        "ORCA025-N06_20000129d05V.nc",
        "ORCA025-N06_20000104d05W.nc",
        "ORCA025-N06_20000109d05W.nc",
        "ORCA025-N06_20000114d05W.nc",
        "ORCA025-N06_20000119d05W.nc",
        "ORCA025-N06_20000124d05W.nc",
        "ORCA025-N06_20000129d05W.nc",
        "coordinates.nc",
    ],
    "POPSouthernOcean_data": [
        "t.x1_SAMOC_flux.169000.nc",
        "t.x1_SAMOC_flux.169001.nc",
        "t.x1_SAMOC_flux.169002.nc",
        "t.x1_SAMOC_flux.169003.nc",
        "t.x1_SAMOC_flux.169004.nc",
        "t.x1_SAMOC_flux.169005.nc",
    ],
    "SWASH_data": [
        "field_0065532.nc",
        "field_0065537.nc",
        "field_0065542.nc",
        "field_0065548.nc",
        "field_0065552.nc",
        "field_0065557.nc",
    ],
    "WOA_data": [f"woa18_decav_t{m:02d}_04.nc" for m in range(1, 13)],
    "CROCOidealized_data": ["CROCO_idealized.nc"],
}


def _create_pooch_registry() -> dict[str, None]:
    """Collapses the mapping of dataset names to filenames into a pooch registry.

    Hashes are set to None for all files.
    """
    registry: dict[str, None] = {}
    for dataset, filenames in EXAMPLE_DATA_FILES.items():
        for filename in filenames:
            registry[f"{dataset}/{filename}"] = None
    return registry


POOCH_REGISTRY = _create_pooch_registry()


def _get_pooch(data_home=None):
    if data_home is None:
        data_home = os.environ.get("PARCELS_EXAMPLE_DATA")
    if data_home is None:
        data_home = pooch.os_cache("parcels")

    return pooch.create(
        path=data_home,
        base_url=DATA_URL,
        registry=POOCH_REGISTRY,
    )


def list_example_datasets() -> list[str]:
    """List the available example datasets.

    Use :func:`download_example_dataset` to download one of the datasets.

    Returns
    -------
    datasets : list of str
        The names of the available example datasets.
    """
    return list(EXAMPLE_DATA_FILES.keys())


def download_example_dataset(dataset: str, data_home=None):
    """Load an example dataset from the parcels website.

    This function provides quick access to a small number of example datasets
    that are useful in documentation and testing in parcels.

    Parameters
    ----------
    dataset : str
        Name of the dataset to load.
    data_home : pathlike, optional
        The directory in which to cache data. If not specified, the value
        of the ``PARCELS_EXAMPLE_DATA`` environment variable, if any, is used.
        Otherwise the default location is assigned by :func:`get_data_home`.

    Returns
    -------
    dataset_folder : Path
        Path to the folder containing the downloaded dataset files.
    """
    # Dev note: `dataset` is assumed to be a folder name with netcdf files
    if dataset not in EXAMPLE_DATA_FILES:
        raise ValueError(
            f"Dataset {dataset!r} not found. Available datasets are: " + ", ".join(EXAMPLE_DATA_FILES.keys())
        )
    odie = _get_pooch(data_home=data_home)

    cache_folder = Path(odie.path)
    dataset_folder = cache_folder / dataset

    for file_name in odie.registry:
        if file_name.startswith(dataset):
            should_patch = dataset == "GlobCurrent_example_data"
            odie.fetch(file_name, processor=_v4_compat_patch if should_patch else None)

    return dataset_folder


def _v4_compat_patch(fname, action, pup):
    """
    Patch the GlobCurrent example dataset to be compatible with v4.

    See https://www.fatiando.org/pooch/latest/processors.html#creating-your-own-processors
    """
    if action == "fetch":
        return fname
    xr.load_dataset(fname).pipe(patch_dataset_v4_compat).to_netcdf(fname)
    return fname

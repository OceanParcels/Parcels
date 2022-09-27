"""Get example scripts, notebooks, and data files."""
import argparse
import os
from datetime import datetime
from datetime import timedelta
import shutil

import pkg_resources
from tqdm import tqdm

try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen


example_data_files = (
    ["MovingEddies_data/" + fn for fn in [
        "moving_eddiesP.nc", "moving_eddiesU.nc", "moving_eddiesV.nc"]]
    + ["MITgcm_example_data/mitgcm_UV_surface_zonally_reentrant.nc", ]
    + ["OFAM_example_data/" + fn for fn in [
        "OFAM_simple_U.nc", "OFAM_simple_V.nc"]]
    + ["Peninsula_data/" + fn for fn in [
        "peninsulaU.nc", "peninsulaV.nc", "peninsulaP.nc", "peninsulaT.nc"]]
    + ["GlobCurrent_example_data/" + fn for fn in [
        "%s000000-GLOBCURRENT-L4-CUReul_hs-ALT_SUM-v02.0-fv01.0.nc" % (
            date.strftime("%Y%m%d"))
        for date in ([datetime(2002, 1, 1) + timedelta(days=x)
                     for x in range(0, 365)] + [datetime(2003, 1, 1)])]]
    + ["DecayingMovingEddy_data/" + fn for fn in [
        "decaying_moving_eddyU.nc", "decaying_moving_eddyV.nc"]]
    + ["NemoCurvilinear_data/" + fn for fn in [
        "U_purely_zonal-ORCA025_grid_U.nc4", "V_purely_zonal-ORCA025_grid_V.nc4",
        "mesh_mask.nc4"]]
    + ["NemoNorthSeaORCA025-N006_data/" + fn for fn in [
        "ORCA025-N06_20000104d05U.nc", "ORCA025-N06_20000109d05U.nc", "ORCA025-N06_20000114d05U.nc", "ORCA025-N06_20000119d05U.nc", "ORCA025-N06_20000124d05U.nc", "ORCA025-N06_20000129d05U.nc",
        "ORCA025-N06_20000104d05V.nc", "ORCA025-N06_20000109d05V.nc", "ORCA025-N06_20000114d05V.nc", "ORCA025-N06_20000119d05V.nc", "ORCA025-N06_20000124d05V.nc", "ORCA025-N06_20000129d05V.nc",
        "ORCA025-N06_20000104d05W.nc", "ORCA025-N06_20000109d05W.nc", "ORCA025-N06_20000114d05W.nc", "ORCA025-N06_20000119d05W.nc", "ORCA025-N06_20000124d05W.nc", "ORCA025-N06_20000129d05W.nc",
        "coordinates.nc"]]
    + ["POPSouthernOcean_data/" + fn for fn in ["t.x1_SAMOC_flux.169000.nc", "t.x1_SAMOC_flux.169001.nc",
                                                "t.x1_SAMOC_flux.169002.nc", "t.x1_SAMOC_flux.169003.nc",
                                                "t.x1_SAMOC_flux.169004.nc", "t.x1_SAMOC_flux.169005.nc"]]
    + ["SWASH_data/" + fn for fn in ["field_0065532.nc", "field_0065537.nc", "field_0065542.nc", "field_0065548.nc", "field_0065552.nc", "field_0065557.nc"]]
    + ["WOA_data/" + fn for fn in ["woa18_decav_t%.2d_04.nc" % m
                                   for m in range(1, 13)]])

example_data_url = "http://oceanparcels.org/examples-data"


def _maybe_create_dir(path):
    """Create directory (and parents) if they don't exist."""
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def copy_data_and_examples_from_package_to(target_path):
    """Copy example data from Parcels directory.

    Return thos parths of the list `file_names` that were not found in the
    package.

    """
    examples_in_package = pkg_resources.resource_filename("parcels", "examples")
    try:
        shutil.copytree(examples_in_package, target_path)
    except Exception as e:
        print(e)
        pass


def _still_to_download(file_names, target_path):
    """Only return the files that are not yet present on disk."""
    for fn in list(file_names):
        if os.path.exists(os.path.join(target_path, fn)):
            file_names.remove(fn)
    return file_names


def download_files(source_url, file_names, target_path):
    """Mirror file_names from source_url to target_path."""
    _maybe_create_dir(target_path)
    print("Downloading %s ..." % (source_url.split("/")[-1]))
    for filename in tqdm(file_names):
        _maybe_create_dir(os.path.join(target_path, os.path.dirname(filename)))
        if not os.path.exists(os.path.join(target_path, filename)):
            download_url = source_url + "/" + filename
            src = urlopen(download_url)
            with open(os.path.join(target_path, filename), 'wb') as dst:
                dst.write(src.read())


def main(target_path=None):
    """Get example scripts, example notebooks, and example data.

    Copy the examples from the package directory and get the example data either
    from the package directory or from the Parcels website.
    """
    if target_path is None:
        # get targe directory
        parser = argparse.ArgumentParser(
            description="Get Parcels example data.")
        parser.add_argument(
            "target_path",
            help="Where to put the tutorials?  (This path will be created.)")
        args = parser.parse_args()
        target_path = args.target_path

    if os.path.exists(os.path.join(target_path, "MovingEddies_data")):
        print("Error: {} already exists.".format(target_path))
        return

    # copy data and examples
    copy_data_and_examples_from_package_to(target_path)

    # try downloading remaining files
    remaining_example_data_files = _still_to_download(
        example_data_files, target_path)
    download_files(example_data_url, remaining_example_data_files, target_path)


if __name__ == "__main__":
    main()

"""Get example scripts, notebooks, and data files."""

import argparse
from datetime import datetime, timedelta
from glob import glob
import json
import os
import pkg_resources
from progressbar import ProgressBar
try:
    # For Python 3.0 and later
    from urllib.request import urlopen
except ImportError:
    # Fall back to Python 2's urllib2
    from urllib2 import urlopen
import shutil
import sys

example_data_files = (
    ["MovingEddies_data/" + fn for fn in [
        "moving_eddiesP.nc", "moving_eddiesU.nc", "moving_eddiesV.nc"]]
    + ["OFAM_example_data/" + fn for fn in [
        "OFAM_simple_U.nc", "OFAM_simple_V.nc"]]
    + ["Peninsula_data/" + fn for fn in [
        "peninsulaU.nc", "peninsulaV.nc", "peninsulaP.nc"]]
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
        "ORCA025-N06_20000104d05U.nc", "ORCA025-N06_20000109d05U.nc",
        "ORCA025-N06_20000104d05V.nc", "ORCA025-N06_20000109d05V.nc",
        "ORCA025-N06_20000104d05W.nc", "ORCA025-N06_20000109d05W.nc",
        "coordinates.nc"]])

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


def set_jupyter_kernel_to_python_version(path, python_version=2):
    """Set notebook kernelspec to desired python version.

    This also drops all other meta data from the notebook.
    """
    for file_name in glob(os.path.join(path, "*.ipynb")):

        with open(file_name, 'r') as f:
            notebook_data = json.load(f)

        notebook_data['metadata'] = {"kernelspec": {
            "display_name": "Python {}".format(python_version),
            "language": "python",
            "name": "python{}".format(python_version)}}

        with open(file_name, 'w') as f:
            json.dump(notebook_data, f, indent=2)


def _still_to_download(file_names, target_path):
    """Only return the files that are not yet present on disk."""
    for fn in list(file_names):
        if os.path.exists(os.path.join(target_path, fn)):
            file_names.remove(fn)
    return file_names


def download_files(source_url, file_names, target_path):
    """Mirror file_names from source_url to target_path."""
    _maybe_create_dir(target_path)
    pbar = ProgressBar()
    print("Downloading %s ..." % (source_url.split("/")[-1]))
    for filename in pbar(file_names):
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

    if os.path.exists(target_path):
        print("Error: {} already exists.".format(target_path))
        return

    # copy data and examples
    copy_data_and_examples_from_package_to(target_path)

    # make sure the notebooks use the correct python version
    set_jupyter_kernel_to_python_version(
        target_path,
        python_version=sys.version_info[0])

    # try downloading remaining files
    remaining_example_data_files = _still_to_download(
        example_data_files, target_path)
    download_files(example_data_url, remaining_example_data_files, target_path)


if __name__ == "__main__":
    main()

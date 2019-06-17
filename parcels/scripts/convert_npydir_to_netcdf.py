from parcels import ParticleFile
import numpy as np
from os import path
from argparse import ArgumentParser


def convert_npydir_to_netcdf(tempwritedir):
    """Convert npy files in tempwritedir to a NetCDF file
    :param tempwritedir: directory where the temporary npy files are stored (can be obtained from ParticleFile.tempwritedir attribute)
    """

    pyset_file = path.join(tempwritedir, 'pset_info.npy')
    if not path.isdir(tempwritedir):
        raise ValueError('Output directory "%s" does not exist' % tempwritedir)
    if not path.isfile(pyset_file):
        raise ValueError('Output directory "%s" does not contain a pset_info.npy file' % tempwritedir)

    pset_info = np.load(pyset_file, allow_pickle=True).item()
    pfile = ParticleFile(None, None, pset_info=pset_info)
    pfile.export()
    pfile.dataset.close()


def main(tempwritedir=None):
    if tempwritedir is None:
        p = ArgumentParser(description="""Script to convert temporary npy output files to NetCDF""")
        p.add_argument('tempwritedir', help='Name of directory where temporary npy files are stored')
        args = p.parse_args()
        tempwritedir = args.tempwritedir

    convert_npydir_to_netcdf(tempwritedir)


if __name__ == "__main__":
    main()

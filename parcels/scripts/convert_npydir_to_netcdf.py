from parcels import ParticleFile
import numpy as np
from os import path
from glob import glob
from argparse import ArgumentParser


def convert_npydir_to_netcdf(tempwritedir, tempwritedir_base):
    """Convert npy files in tempwritedir to a NetCDF file
    :param tempwritedir: directory where the temporary npy files are stored (can be obtained from ParticleFile.tempwritedir attribute)
    """

    pyset_file = path.join(tempwritedir, 'pset_info.npy')
    if not path.isdir(tempwritedir):
        raise ValueError('Output directory "%s" does not exist' % tempwritedir)
    if not path.isfile(pyset_file):
        raise ValueError('Output directory "%s" does not contain a pset_info.npy file' % tempwritedir)

    pset_info = np.load(pyset_file, allow_pickle=True).item()
    pfile = ParticleFile(None, None, pset_info=pset_info, tempwritedir=tempwritedir_base)
    pfile.export()
    pfile.dataset.close()


def main(tempwritedir_base=None):
    if tempwritedir_base is None:
        p = ArgumentParser(description="""Script to convert temporary npy output files to NetCDF""")
        p.add_argument('tempwritedir', help='Name of directory where temporary npy files are stored')
        args = p.parse_args()
        tempwritedir_base = args.tempwritedir

    temp_names = sorted(glob("%s/*/" % tempwritedir_base), key=lambda x: int(x[:-1].rsplit('/', 1)[1]))

    convert_npydir_to_netcdf(temp_names[0], tempwritedir_base)


if __name__ == "__main__":
    main()

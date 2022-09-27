from argparse import ArgumentParser
from glob import glob
from os import path

import numpy as np

# == here those classes need to be impported to parse available ParticleFile classes and create the type from its name == #
from parcels import ParticleFile, ParticleFileSOA, ParticleFileAOS  # NOQA


def convert_npydir_to_netcdf(tempwritedir_base, delete_tempfiles=False, pfile_class=None):
    """Convert npy files in tempwritedir to a NetCDF file
    :param tempwritedir_base: directory where the directories for temporary npy files
            are stored (can be obtained from ParticleFile.tempwritedir_base attribute)
    """

    tempwritedir = sorted(glob(path.join("%s" % tempwritedir_base, "*")),
                          key=lambda x: int(path.basename(x)))[0]
    pyset_file = path.join(tempwritedir, 'pset_info.npy')
    if not path.isdir(tempwritedir):
        raise ValueError('Output directory "%s" does not exist' % tempwritedir)
    if not path.isfile(pyset_file):
        raise ValueError('Output directory "%s" does not contain a pset_info.npy file' % tempwritedir)

    pset_info = np.load(pyset_file, allow_pickle=True).item()
    pfconstructor = ParticleFile if pfile_class is None else pfile_class
    pfile = pfconstructor(None, None, pset_info=pset_info, tempwritedir=tempwritedir_base, convert_at_end=False)
    pfile.close(delete_tempfiles)


def main(tempwritedir_base=None, delete_tempfiles=False):
    if tempwritedir_base is None:
        p = ArgumentParser(description="""Script to convert temporary npy output files to NetCDF""")
        p.add_argument('tempwritedir', help='Name of directory where temporary npy files are stored '
                                            '(not including numbered subdirectories)')
        p.add_argument('-d', '--delete_tempfiles', default=False,
                       help='Flag to delete temporary files at end of call (default False)')
        p.add_argument('-c', '--pfclass_name', default='ParticleFileSOA',
                       help='Class name of the stored particle file (default ParticleFileSOA)')
        args = p.parse_args()
        tempwritedir_base = args.tempwritedir
        pfclass = ParticleFile
        if hasattr(args, 'delete_tempfiles'):
            delete_tempfiles = args.delete_tempfiles
        if hasattr(args, 'pfclass_name'):
            try:
                pfclass = locals()[args.pfclass_name]
            except:
                pfclass = ParticleFile

    convert_npydir_to_netcdf(tempwritedir_base, delete_tempfiles, pfile_class=pfclass)


if __name__ == "__main__":
    main()

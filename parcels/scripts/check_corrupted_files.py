"""
Function to check if the data of a variable in a field list is corrupted
"""

import numpy as np
import xarray as xr
import os


def check_corrupted_files(filelist, fieldname, outputfile=None):
    """
    param filelist: list of files which contain the field with name 'fieldname'
    param fieldname: name of the field that is checked
    param outputfile: list of corrupted files is written into a output text file
    """

    if outputfile is None:
        outputfile = 'CorruptedFiles_field_' + fieldname + '.txt'

    if os.path.isfile(outputfile):
        print('Outputfile exists already. Please delete or choose another name for the output file.')
        return 0

    myfile = open(outputfile, 'w')

    for f in filelist:
        print('Load file: ', f)
        try:
            ds = xr.open_dataset(f, decode_cf=False)
            np.array(getattr(ds, fieldname))
        except:
            myfile.write("%s\n" % f)
            print('file is corrupted:  ', f)

    myfile.close()

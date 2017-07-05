from netCDF4 import Dataset
import numpy as np
from progressbar import ProgressBar
from argparse import ArgumentParser


def convert_IndexedOutputToArray(file_in, file_out):
    """Script to convert a trajectory file as outputted by Parcels
     in Indexed format to the easier-to-handle Array format

     :param file_in: name of the input file in Indexed format
     :param file_out: name of the output file"""

    pfile_in = Dataset(file_in, 'r')
    if 'trajectory' in pfile_in.dimensions:
        print file_in+' appears to be in array format already. Doing nothing..'
        return

    class IndexedTrajectories(object):
        """IndexedTrajectories class that holds info on the indices where the
        individual particle trajectories start and end"""
        def __init__(self, pfile):
            self.ids = pfile.variables['trajectory'][:]
            self.indices = np.argsort(self.ids)
            self.ids = self.ids[self.indices]
            self.starts = np.insert([x + 1 for x in np.where(np.diff(self.ids) > 0)], 0, 0)
            self.ends = np.append(self.starts[1:], [len(self.ids)-1])
            self.lengths = self.ends - self.starts
            self.nid = len(self.starts)
            self.nobs = np.max(self.lengths)
            for i in range(self.nid):
                # Test whether all ids in a trajectory are the same
                assert all(self.ids[self.starts[i]:self.ends[i]] == self.ids[self.starts[i]])

    trajs = IndexedTrajectories(pfile_in)

    pfile_out = Dataset("%s" % file_out, "w", format="NETCDF4")
    pfile_out.createDimension("obs", trajs.nobs)
    pfile_out.createDimension("trajectory", trajs.nid)
    coords = ("trajectory", "obs")

    id = pfile_out.createVariable("trajectory", "i4", ("trajectory",))
    id.long_name = "Unique identifier for each particle"
    id.cf_role = "trajectory_id"
    id[:] = np.array([trajs.ids[p] for p in trajs.starts])

    # create dict of all variables, except 'trajectory' as that is already created above
    var = {}
    for v in pfile_in.variables:
        if str(v) != 'trajectory':
            varin = pfile_in.variables[v]
            var[v] = pfile_out.createVariable(v, "f4", coords, fill_value=np.nan)
            # copy all attributes, except Fill_Value which is set automatically
            var[v].setncatts({k: varin.getncattr(k) for k in varin.ncattrs() if k != '_FillValue'})

    pbar = ProgressBar()
    for i in pbar(range(trajs.nid)):
        ii = np.sort(trajs.indices[trajs.starts[i]:trajs.ends[i]])
        for v in var:
            var[v][i, 0:trajs.lengths[i]] = pfile_in.variables[v][ii]

    pfile_out.sync()


if __name__ == "__main__":
    p = ArgumentParser(description="""Converting Indexed Parcels output to Array format""")
    p.add_argument('-i', '--file_in', type=str,
                   help='Name of input file in indexed form')
    p.add_argument('-o', '--file_out', type=str,
                   help='Name of output file in array form')
    args = p.parse_args()
    convert_IndexedOutputToArray(file_in=args.file_in, file_out=args.file_out)

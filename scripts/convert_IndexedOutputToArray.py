from netCDF4 import Dataset
import numpy as np
from progressbar import ProgressBar
from argparse import ArgumentParser


def convert_IndexedOutputToArray(file_in, file_out):
    pfile_in = Dataset(file_in, 'r')
    if 'trajectory' in pfile_in.dimensions:
        print file_in+' appears to be in array format already. Doing nothing..'
        return

    traj_ids = pfile_in.variables['trajectory'][:]
    traj_indices = np.argsort(traj_ids)
    traj_ids = traj_ids[traj_indices]
    traj_starts = np.insert([x + 1 for x in np.where(np.diff(traj_ids) > 0)], 0, 0)
    traj_ends = np.append(traj_starts[1:], [len(traj_ids)-1])
    traj_lens = traj_ends - traj_starts
    nid = len(traj_starts)
    nobs = np.max(traj_lens)
    for i in range(nid):
        assert all(traj_ids[traj_starts[i]:traj_ends[i]] == traj_ids[traj_starts[i]])

    pfile_out = Dataset("%s" % file_out, "w", format="NETCDF4")
    pfile_out.createDimension("obs", nobs)
    pfile_out.createDimension("trajectory", nid)
    coords = ("trajectory", "obs")

    id = pfile_out.createVariable("trajectory", "i4", ("trajectory",))
    id.long_name = "Unique identifier for each particle"
    id.cf_role = "trajectory_id"
    id[:] = np.array([traj_ids[p] for p in traj_starts])

    var = {}
    for v in pfile_in.variables:
        if str(v) != 'trajectory':
            varin = pfile_in.variables[v]
            var[v] = pfile_out.createVariable(v, "f4", coords, fill_value=np.nan)
            var[v].setncatts({k: varin.getncattr(k) for k in varin.ncattrs() if k != '_FillValue'})

    pbar = ProgressBar()
    for i in pbar(range(nid)):
        ii = np.sort(traj_indices[traj_starts[i]:traj_ends[i]])
        for v in var:
            var[v][i, 0:traj_lens[i]] = pfile_in.variables[v][ii]

    pfile_out.sync()


if __name__ == "__main__":
    p = ArgumentParser(description="""Converting Indexed Parcels output to Array format""")
    p.add_argument('-i', '--file_in', type=str,
                   help='Name of input file in indexed form')
    p.add_argument('-o', '--file_out', type=str,
                   help='Name of output file in array form')
    args = p.parse_args()
    convert_IndexedOutputToArray(file_in=args.file_in, file_out=args.file_out)

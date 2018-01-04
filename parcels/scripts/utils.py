import numpy as np
import xarray as xr
from os import path
from netCDF4 import Dataset


def compute_curvilinear_rotation_angles(mesh_filename, rotation_angles_filename, variables, dimensions):

    if path.isfile(rotation_angles_filename) and path.getmtime(rotation_angles_filename) > path.getmtime(mesh_filename):
        return

    dataset = xr.open_dataset(mesh_filename)
    lonU = np.squeeze(getattr(dataset, dimensions['U']['lon']).values)
    latU = np.squeeze(getattr(dataset, dimensions['U']['lat']).values)
    lonV = np.squeeze(getattr(dataset, dimensions['V']['lon']).values)
    latV = np.squeeze(getattr(dataset, dimensions['V']['lat']).values)
    lonF = np.squeeze(getattr(dataset, dimensions['F']['lon']).values)
    latF = np.squeeze(getattr(dataset, dimensions['F']['lat']).values)
    dataset.close()

    rad = np.pi / 180.
    rpi = np.pi

    zxnpu = - 2. * np.cos(rad*lonU[1:, 1:]) * np.tan(rpi/4. - rad*latU[1:, 1:]/2.)
    zynpu = - 2. * np.sin(rad*lonU[1:, 1:]) * np.tan(rpi/4. - rad*latU[1:, 1:]/2.)
    znnpu = zxnpu*zxnpu + zynpu*zynpu

    zxnpv = - 2. * np.cos(rad*lonV[1:, 1:]) * np.tan(rpi/4. - rad*latV[1:, 1:]/2.)
    zynpv = - 2. * np.sin(rad*lonV[1:, 1:]) * np.tan(rpi/4. - rad*latV[1:, 1:]/2.)
    znnpv = zxnpv*zxnpv + zynpv*zynpv

    zxffu = 2. * np.cos(rad*lonF[1:, 1:]) * np.tan(rpi/4. - rad*latF[1:, 1:]/2.) \
        - 2. * np.cos(rad*lonF[:-1, 1:]) * np.tan(rpi/4. - rad*latF[:-1, 1:]/2.)
    zyffu = 2. * np.sin(rad*lonF[1:, 1:]) * np.tan(rpi/4. - rad*latF[1:, 1:]/2.) \
        - 2. * np.sin(rad*lonF[:-1, 1:]) * np.tan(rpi/4. - rad*latF[:-1, 1:]/2.)
    znffu = np.sqrt(znnpu * (zxffu*zxffu + zyffu*zyffu))
    znffu = np.maximum(znffu, 1.e-14)

    zxffv = 2. * np.cos(rad*lonF[1:, 1:]) * np.tan(rpi/4. - rad*latF[1:, 1:]/2.) \
        - 2. * np.cos(rad*lonF[1:, :-1]) * np.tan(rpi/4. - rad*latF[1:, :-1]/2.)
    zyffv = 2. * np.sin(rad*lonF[1:, 1:]) * np.tan(rpi/4. - rad*latF[1:, 1:]/2.) \
        - 2. * np.sin(rad*lonF[1:, :-1]) * np.tan(rpi/4. - rad*latF[1:, :-1]/2.)
    znffv = np.sqrt(znnpv * (zxffv*zxffv + zyffv*zyffv))
    znffv = np.maximum(znffv, 1.e-14)

    gsinu = (zxnpu*zyffu - zynpu*zxffu) / znffu
    gcosu = (zxnpu*zxffu + zynpu*zyffu) / znffu
    gsinv = (zxnpv*zxffv + zynpv*zyffv) / znffv
    gcosv = -(zxnpv*zyffv - zynpv*zxffv) / znffv

    # ** netCDF4 writing, since xArray is bugged **
    lonU = lonU[1:, 1:]
    latU = latU[1:, 1:]
    lonV = lonV[1:, 1:]
    latV = latV[1:, 1:]

    subDataset = Dataset(rotation_angles_filename, 'w', format='NETCDF4')
    subDataset.createDimension('x', lonU.shape[1])
    subDataset.createDimension('y', lonU.shape[0])
    lonUVar = subDataset.createVariable(dimensions['U']['lon'], 'f8', ('y', 'x',))
    latUVar = subDataset.createVariable(dimensions['U']['lat'], 'f8', ('y', 'x',))
    lonUVar.valid_min = np.min(lonU)
    lonUVar.valid_max = np.max(lonU)
    lonUVar[:] = lonU
    latUVar[:] = latU
    lonVVar = subDataset.createVariable(dimensions['V']['lon'], 'f8', ('y', 'x',))
    latVVar = subDataset.createVariable(dimensions['V']['lat'], 'f8', ('y', 'x',))
    lonVVar.valid_min = np.min(lonV)
    lonVVar.valid_max = np.max(lonV)
    lonVVar[:] = lonV
    latVVar[:] = latV

    cosUVar = subDataset.createVariable(variables['cosU'], 'f8', ('y', 'x',))
    cosUVar[:] = gcosu
    sinUVar = subDataset.createVariable(variables['sinU'], 'f8', ('y', 'x',))
    sinUVar[:] = gsinu
    cosVVar = subDataset.createVariable(variables['cosV'], 'f8', ('y', 'x',))
    cosVVar[:] = gcosv
    sinVVar = subDataset.createVariable(variables['sinV'], 'f8', ('y', 'x',))
    sinVVar[:] = gsinv

    subDataset.close()
    # ** end netCDF4 writing **

    # lonUArray = xr.DataArray(lonU[1:, 1:],
    #                          name='lonU',
    #                          dims=('y', 'x'),
    #                          attrs={'valid_min': np.min(lonU),
    #                                 'valid_max': np.max(lonU)})
    # latUArray = xr.DataArray(latU[1:, 1:],
    #                          name='latU',
    #                          dims=('y', 'x'),
    #                          attrs={'valid_min': np.min(latU),
    #                                 'valid_max': np.max(latU)})
    # coords = {lonUArray.name: lonUArray,
    #           latUArray.name: latUArray}
    # cUArray = xr.DataArray(gcosu, name='cosU', coords=coords, dims=('y', 'x'))
    # sUArray = xr.DataArray(gsinu, name='sinU', coords=coords, dims=('y', 'x'))

    # lonVArray = xr.DataArray(lonV[1:, 1:],
    #                          name='lonV',
    #                          dims=('y', 'x'),
    #                          attrs={'valid_min': np.min(lonV),
    #                                 'valid_max': np.max(lonV)})
    # latVArray = xr.DataArray(latV[1:, 1:],
    #                          name='latV',
    #                          dims=('y', 'x'),
    #                          attrs={'valid_min': np.min(latV),
    #                                 'valid_max': np.max(latV)})
    # coords = {lonVArray.name: lonVArray,
    #           latVArray.name: latVArray}
    # cVArray = xr.DataArray(gcosv, name='cosV', coords=coords, dims=('y', 'x'))
    # sVArray = xr.DataArray(gsinv, name='sinV', coords=coords, dims=('y', 'x'))

    # dataset = xr.Dataset()
    # dataset[cUArray.name] = cUArray
    # dataset[sUArray.name] = sUArray
    # dataset[cVArray.name] = cVArray
    # dataset[sVArray.name] = sVArray
    # dataset.to_netcdf(path=angles_filename, engine='scipy')

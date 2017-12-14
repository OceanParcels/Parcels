import numpy as np
import xarray as xr


def compute_curvilinear_rotation_angles(mesh_filename, angles_filename):

    dataset = xr.open_dataset(mesh_filename)
    lonU = dataset.glamu.values[0, 0, :, :]
    latU = dataset.gphiu.values[0, 0, :, :]
    lonV = dataset.glamv.values[0, 0, :, :]
    latV = dataset.gphiv.values[0, 0, :, :]
    lonF = dataset.glamf.values[0, 0, :, :]
    latF = dataset.gphif.values[0, 0, :, :]

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

    zxffv = 2. * np.cos(rad*lonF[1:, :-1]) * np.tan(rpi/4. - rad*latF[1:, :-1]/2.) \
        - 2. * np.cos(rad*lonF[1:, 1:]) * np.tan(rpi/4. - rad*latF[1:, 1:]/2.)
    zyffv = 2. * np.sin(rad*lonF[1:, :-1]) * np.tan(rpi/4. - rad*latF[1:, :-1]/2.) \
        - 2. * np.sin(rad*lonF[1:, 1:]) * np.tan(rpi/4. - rad*latF[1:, 1:]/2.)
    znffv = np.sqrt(znnpv * (zxffv*zxffv + zyffv*zyffv))
    znffv = np.maximum(znffv, 1.e-14)

    gsinu = (zxnpu*zyffu - zynpu*zxffu) / znffu
    gcosu = (zxnpu*zxffu + zynpu*zyffu) / znffu
    gsinv = (zxnpv*zyffv - zynpv*zxffv) / znffv
    gcosv = (zxnpv*zxffv + zynpv*zyffv) / znffv

    lonUArray = xr.DataArray(lonU[1:, 1:],
                             name='lonU',
                             dims=('y', 'x'),
                             attrs={'valid_min': np.min(lonU),
                                    'valid_max': np.max(lonU)})
    latUArray = xr.DataArray(latU[1:, 1:],
                             name='latU',
                             dims=('y', 'x'),
                             attrs={'valid_min': np.min(latU),
                                    'valid_max': np.max(latU)})
    coords = {lonUArray.name: lonUArray,
              latUArray.name: latUArray}
    cUArray = xr.DataArray(gcosu, name='cosU', coords=coords, dims=('y', 'x'))
    sUArray = xr.DataArray(gsinu, name='sinU', coords=coords, dims=('y', 'x'))

    lonVArray = xr.DataArray(lonV[1:, 1:],
                             name='lonV',
                             dims=('y', 'x'),
                             attrs={'valid_min': np.min(lonV),
                                    'valid_max': np.max(lonV)})
    latVArray = xr.DataArray(latV[1:, 1:],
                             name='latV',
                             dims=('y', 'x'),
                             attrs={'valid_min': np.min(latV),
                                    'valid_max': np.max(latV)})
    coords = {lonVArray.name: lonVArray,
              latVArray.name: latVArray}
    cVArray = xr.DataArray(gcosv, name='cosV', coords=coords, dims=('y', 'x'))
    sVArray = xr.DataArray(gsinv, name='sinV', coords=coords, dims=('y', 'x'))

    dataset = xr.Dataset()
    dataset[cUArray.name] = cUArray
    dataset[sUArray.name] = sUArray
    dataset[cVArray.name] = cVArray
    dataset[sVArray.name] = sVArray
    dataset.to_netcdf(path=angles_filename, engine='scipy')

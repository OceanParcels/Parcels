# flake8: noqa
import numpy as np


def phi1D_lin(xsi):
    phi = [1-xsi,
           xsi]

    return phi


def phi1D_quad(xsi):
    phi = [2*xsi**2-3*xsi+1,
           -4*xsi**2+4*xsi,
           2*xsi**2-xsi]

    return phi


def phi2D_lin(xsi, eta):
    phi = [(1-xsi) * (1-eta),
              xsi  * (1-eta),
              xsi  *    eta ,
           (1-xsi) *    eta ]

    return phi


def phi3D_lin(xsi, eta, zet):
    phi = [(1-xsi) * (1-eta) * (1-zet),
              xsi  * (1-eta) * (1-zet),
              xsi  *    eta  * (1-zet),
           (1-xsi) *    eta  * (1-zet),
           (1-xsi) * (1-eta) *    zet ,
              xsi  * (1-eta) *    zet ,
              xsi  *    eta  *    zet ,
           (1-xsi) *    eta  *    zet ]

    return phi


def dphidxsi3D_lin(xsi, eta, zet):
    dphidxsi = [ - (1-eta) * (1-zet),
                   (1-eta) * (1-zet),
                   (  eta) * (1-zet),
                 - (  eta) * (1-zet),
                 - (1-eta) * (  zet),
                   (1-eta) * (  zet),
                   (  eta) * (  zet),
                 - (  eta) * (  zet)]
    dphideta = [ - (1-xsi) * (1-zet),
                 - (  xsi) * (1-zet),
                   (  xsi) * (1-zet),
                   (1-xsi) * (1-zet),
                 - (1-xsi) * (  zet),
                 - (  xsi) * (  zet),
                   (  xsi) * (  zet),
                   (1-xsi) * (  zet)]
    dphidzet = [ - (1-xsi) * (1-eta),
                 - (  xsi) * (1-eta),
                 - (  xsi) * (  eta),
                 - (1-xsi) * (  eta),
                   (1-xsi) * (1-eta),
                   (  xsi) * (1-eta),
                   (  xsi) * (  eta),
                   (1-xsi) * (  eta)]

    return dphidxsi, dphideta, dphidzet


def dxdxsi3D_lin(hexa_x, hexa_y, hexa_z, xsi, eta, zet, mesh):
    dphidxsi, dphideta, dphidzet = dphidxsi3D_lin(xsi, eta, zet)

    if mesh == 'spherical':
        deg2m = 1852 * 60.
        rad = np.pi / 180.
        lat = (1-xsi) * (1-eta) * hexa_y[0] + \
                 xsi  * (1-eta) * hexa_y[1] + \
                 xsi  *    eta  * hexa_y[2] + \
              (1-xsi) *    eta  * hexa_y[3]
        jac_lon = deg2m * np.cos(rad * lat)
        jac_lat = deg2m
    else:
        jac_lon = 1
        jac_lat = 1

    dxdxsi = np.dot(hexa_x, dphidxsi) * jac_lon
    dxdeta = np.dot(hexa_x, dphideta) * jac_lon
    dxdzet = np.dot(hexa_x, dphidzet) * jac_lon
    dydxsi = np.dot(hexa_y, dphidxsi) * jac_lat
    dydeta = np.dot(hexa_y, dphideta) * jac_lat
    dydzet = np.dot(hexa_y, dphidzet) * jac_lat
    dzdxsi = np.dot(hexa_z, dphidxsi)
    dzdeta = np.dot(hexa_z, dphideta)
    dzdzet = np.dot(hexa_z, dphidzet)

    return dxdxsi, dxdeta, dxdzet, dydxsi, dydeta, dydzet, dzdxsi, dzdeta, dzdzet


def jacobian3D_lin(hexa_x, hexa_y, hexa_z, xsi, eta, zet, mesh):
    dxdxsi, dxdeta, dxdzet, dydxsi, dydeta, dydzet, dzdxsi, dzdeta, dzdzet = dxdxsi3D_lin(hexa_x, hexa_y, hexa_z, xsi, eta, zet, mesh)

    jac = dxdxsi * (dydeta*dzdzet - dzdeta*dydzet)\
        - dxdeta * (dydxsi*dzdzet - dzdxsi*dydzet)\
        + dxdzet * (dydxsi*dzdeta - dzdxsi*dydeta)
    return jac


def jacobian3D_lin_face(hexa_x, hexa_y, hexa_z, xsi, eta, zet, orientation, mesh):
    dxdxsi, dxdeta, dxdzet, dydxsi, dydeta, dydzet, dzdxsi, dzdeta, dzdzet = dxdxsi3D_lin(hexa_x, hexa_y, hexa_z, xsi, eta, zet, mesh)

    if orientation == 'zonal':
        j = [dydeta*dzdzet-dydzet*dzdeta,
            -dxdeta*dzdzet+dxdzet*dzdeta,
             dxdeta*dydzet-dxdzet*dydeta]
    elif orientation == 'meridional':
        j = [dydxsi*dzdzet-dydzet*dzdxsi,
            -dxdxsi*dzdzet+dxdzet*dzdxsi,
             dxdxsi*dydzet-dxdzet*dydxsi]
    elif orientation == 'vertical':
        j = [dydxsi*dzdeta-dydeta*dzdxsi,
            -dxdxsi*dzdeta+dxdeta*dzdxsi,
             dxdxsi*dydeta-dxdeta*dydxsi]

    jac = np.sqrt(j[0]**2+j[1]**2+j[2]**2)
    return jac


def dphidxsi2D_lin(xsi, eta):
    dphidxsi = [-(1-eta),
                  1-eta,
                    eta,
                -   eta]
    dphideta = [-(1-xsi),
                -   xsi,
                    xsi,
                  1-xsi]

    return dphidxsi, dphideta


def dxdxsi2D_lin(quad_x, quad_y, xsi, eta,):
    dphidxsi, dphideta = dphidxsi2D_lin(xsi, eta)

    dxdxsi = np.dot(quad_x, dphidxsi)
    dxdeta = np.dot(quad_x, dphideta)
    dydxsi = np.dot(quad_y, dphidxsi)
    dydeta = np.dot(quad_y, dphideta)

    return dxdxsi, dxdeta, dydxsi, dydeta


def jacobian2D_lin(quad_x, quad_y, xsi, eta):
    dxdxsi, dxdeta, dydxsi, dydeta = dxdxsi2D_lin(quad_x, quad_y, xsi, eta)

    jac = dxdxsi*dydeta - dxdeta*dydxsi
    return jac


def length2d_lin_edge(quad_x, quad_y, ids):
    xe = [quad_x[ids[0]], quad_x[ids[1]]]
    ye = [quad_y[ids[0]], quad_y[ids[1]]]
    return np.sqrt((xe[1]-xe[0])**2+(ye[1]-ye[0])**2)


def interpolate(phi, f, xsi):
    return np.dot(phi(xsi), f)

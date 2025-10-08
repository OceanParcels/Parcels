"""Collection of pre-built eos sea water property kernels."""

import math

__all__ = ["AdiabticTemperatureGradient", "PressureFromLatDepth", "PtempFromTemp", "TempFromPtemp", "UNESCODensity"]


def PressureFromLatDepth(particle, fieldset, time):  # pragma: no cover
    """
    Calculates pressure in dbars from depth in meters and latitude.

    Returns
    -------
    p : array_like
        pressure [db]

    References
    ----------
    1. Saunders, Peter M., 1981: Practical Conversion of Pressure to Depth.
       J. Phys. Oceanogr., 11, 573-574.
       doi: 10.1175/1520-0485(1981)011<0573:PCOPTD>2.0.CO;2
    """
    # Angle conversions.
    deg2rad = math.pi / 180.0

    X = math.sin(max(particle.lat * deg2rad, -1 * particle.lat * deg2rad))
    C1 = 5.92e-3 + math.pow(X, 2) * 5.25e-3
    particle.pressure = ((1 - C1) - math.pow(((math.pow((1 - C1), 2)) - (8.84e-6 * particle.depth)), 0.5)) / 4.42e-6


def AdiabticTemperatureGradient(particle, fieldset, time):  # pragma: no cover
    """Calculates adiabatic temperature gradient as per UNESCO 1983 routines.


    Parameters
    ----------
    particle :
        The particle object with the following attributes:
            - S : array_like
                Salinity in psu (PSS-78).
            - T : array_like
                Temperature in ℃ (ITS-90).
            - pressure : array_like
                Pressure in db.
    fieldset :
        The fieldset object

    Returns
    -------
    array_like
        Adiabatic temperature gradient in ℃ db⁻¹.


    References
    ----------
    1. Fofonoff, P. and Millard, R.C. Jr UNESCO 1983. Algorithms for
       computation of fundamental properties of seawater. UNESCO Tech. Pap. in
       Mar. Sci., No. 44, 53 pp.
       http://unesdoc.unesco.org/images/0005/000598/059832eb.pdf

    2. Bryden, H. 1973. New Polynomials for thermal expansion, adiabatic
       temperature gradient and potential temperature of sea water. Deep-Sea
       Res. Vol20,401-408. doi:10.1016/0011-7471(73)90063-6

    """
    s, t, pres = particle.S, particle.T, particle.pressure

    T68 = t * 1.00024

    a = [3.5803e-5, 8.5258e-6, -6.836e-8, 6.6228e-10]
    b = [1.8932e-6, -4.2393e-8]
    c = [1.8741e-8, -6.7795e-10, 8.733e-12, -5.4481e-14]
    d = [-1.1351e-10, 2.7759e-12]
    e = [-4.6206e-13, 1.8676e-14, -2.1687e-16]
    particle.adtg = (
        a[0]
        + (a[1] + (a[2] + a[3] * T68) * T68) * T68
        + (b[0] + b[1] * T68) * (s - 35)
        + ((c[0] + (c[1] + (c[2] + c[3] * T68) * T68) * T68) + (d[0] + d[1] * T68) * (s - 35)) * pres
        + (e[0] + (e[1] + e[2] * T68) * T68) * pres * pres
    )


def PtempFromTemp(particle, fieldset, time):  # pragma: no cover
    """
    Calculates potential temperature as per UNESCO 1983 report.

    Parameters
    ----------
    particle :
        The particle object with the following attributes:
            - S : array_like
                Salinity in psu (PSS-78).
            - T : array_like
                Temperature in ℃ (ITS-90).
            - pressure : array_like
                Pressure in db.
    fieldset :
        The fieldset object with the following attributes:
            - refpressure : array_like, optional
                Reference pressure in db (default is 0).
    time : float
        Simulation time (not used in this function but required for consistency with other kernels).

    Returns
    -------
    array_like
        Potential temperature relative to reference pressure in ℃ (ITS-90).


    References
    ----------
    1. Fofonoff, P. and Millard, R.C. Jr UNESCO 1983. Algorithms for
       computation of fundamental properties of seawater. UNESCO Tech. Pap. in
       Mar. Sci., No. 44, 53 pp.  Eqn.(31) p.39.
       http://unesdoc.unesco.org/images/0005/000598/059832eb.pdf

    2. Bryden, H. 1973. New Polynomials for thermal expansion, adiabatic
       temperature gradient and potential temperature of sea water. Deep-Sea
       Res. Vol20,401-408. doi:10.1016/0011-7471(73)90063-6

    """
    s = fieldset.psu_salinity[time, particle.depth, particle.lat, particle.lon]
    t = fieldset.temperature[time, particle.depth, particle.lat, particle.lon]
    pres, pr = particle.pressure, fieldset.refpressure

    # First calculate the adiabatic temperature gradient adtg
    # Convert ITS-90 temperature to IPTS-68
    T68 = t * 1.00024

    a = [3.5803e-5, 8.5258e-6, -6.836e-8, 6.6228e-10]
    b = [1.8932e-6, -4.2393e-8]
    c = [1.8741e-8, -6.7795e-10, 8.733e-12, -5.4481e-14]
    d = [-1.1351e-10, 2.7759e-12]
    e = [-4.6206e-13, 1.8676e-14, -2.1687e-16]
    adtg = (
        a[0]
        + (a[1] + (a[2] + a[3] * T68) * T68) * T68
        + (b[0] + b[1] * T68) * (s - 35)
        + ((c[0] + (c[1] + (c[2] + c[3] * T68) * T68) * T68) + (d[0] + d[1] * T68) * (s - 35)) * pres
        + (e[0] + (e[1] + e[2] * T68) * T68) * pres * pres
    )

    # Theta1.
    del_P = pr - pres
    del_th = del_P * adtg
    th = T68 + 0.5 * del_th
    q = del_th

    pprime = pres + 0.5 * del_P
    adtg = (
        a[0]
        + (a[1] + (a[2] + a[3] * th) * th) * th
        + (b[0] + b[1] * th) * (s - 35)
        + ((c[0] + (c[1] + (c[2] + c[3] * th) * th) * th) + (d[0] + d[1] * th) * (s - 35)) * pprime
        + (e[0] + (e[1] + e[2] * th) * th) * pprime * pprime
    )

    # Theta2.
    del_th = del_P * adtg
    th = th + (1 - 1 / 2**0.5) * (del_th - q)
    q = (2 - 2**0.5) * del_th + (-2 + 3 / 2**0.5) * q

    # Theta3.
    adtg = (
        a[0]
        + (a[1] + (a[2] + a[3] * th) * th) * th
        + (b[0] + b[1] * th) * (s - 35)
        + ((c[0] + (c[1] + (c[2] + c[3] * th) * th) * th) + (d[0] + d[1] * th) * (s - 35)) * pprime
        + (e[0] + (e[1] + e[2] * th) * th) * pprime * pprime
    )

    del_th = del_P * adtg
    th = th + (1 + 1 / 2**0.5) * (del_th - q)
    q = (2 + 2**0.5) * del_th + (-2 - 3 / 2**0.5) * q

    # Theta4.
    pprime = pres + del_P
    adtg = (
        a[0]
        + (a[1] + (a[2] + a[3] * th) * th) * th
        + (b[0] + b[1] * th) * (s - 35)
        + ((c[0] + (c[1] + (c[2] + c[3] * th) * th) * th) + (d[0] + d[1] * th) * (s - 35)) * pprime
        + (e[0] + (e[1] + e[2] * th) * th) * pprime * pprime
    )

    del_th = del_P * adtg
    particle.potemp = (th + (del_th - 2 * q) / 6) / 1.00024


def TempFromPtemp(particle, fieldset, time):  # pragma: no cover
    """
    Calculates temperature from potential temperature at the reference
    pressure PR and in situ pressure P.

    Parameters
    ----------
    particle :
        The particle object with the following attributes:
            - S : array_like
                Salinity in psu (PSS-78).
            - T : array_like
                Potential temperature in ℃ (ITS-90).
            - pressure : array_like
                Pressure in db.
    fieldset :
        The fieldset object with the following attributes:
            - refpressure : array_like, optional
                Reference pressure in db (default is 0).
    time : float
        Simulation time (not used in this function but required for consistency with other kernels).

    Returns
    -------
    array_like
        Temperature in ℃ (ITS-90).

    References
    ----------
    1. Fofonoff, P. and Millard, R.C. Jr UNESCO 1983. Algorithms for
       computation of fundamental properties of seawater. UNESCO Tech. Pap. in
       Mar. Sci., No. 44, 53 pp.  Eqn.(31) p.39.
       http://unesdoc.unesco.org/images/0005/000598/059832eb.pdf

    2. Bryden, H. 1973. New Polynomials for thermal expansion, adiabatic
       temperature gradient and potential temperature of sea water. Deep-Sea
       Res.  Vol20,401-408. doi:10.1016/0011-7471(73)90063-6

    """
    s = fieldset.psu_salinity[time, particle.depth, particle.lat, particle.lon]
    t = fieldset.potemperature[time, particle.depth, particle.lat, particle.lon]
    pres, pr = fieldset.refpressure, particle.pressure  # The order should be switched here

    # Convert ITS-90 temperature to IPTS-68
    T68 = t * 1.00024

    a = [3.5803e-5, 8.5258e-6, -6.836e-8, 6.6228e-10]
    b = [1.8932e-6, -4.2393e-8]
    c = [1.8741e-8, -6.7795e-10, 8.733e-12, -5.4481e-14]
    d = [-1.1351e-10, 2.7759e-12]
    e = [-4.6206e-13, 1.8676e-14, -2.1687e-16]
    adtg = (
        a[0]
        + (a[1] + (a[2] + a[3] * T68) * T68) * T68
        + (b[0] + b[1] * T68) * (s - 35)
        + ((c[0] + (c[1] + (c[2] + c[3] * T68) * T68) * T68) + (d[0] + d[1] * T68) * (s - 35)) * pres
        + (e[0] + (e[1] + e[2] * T68) * T68) * pres * pres
    )

    # Theta1.
    del_P = pr - pres
    del_th = del_P * adtg
    th = T68 + 0.5 * del_th
    q = del_th

    pprime = pres + 0.5 * del_P
    adtg = (
        a[0]
        + (a[1] + (a[2] + a[3] * th) * th) * th
        + (b[0] + b[1] * th) * (s - 35)
        + ((c[0] + (c[1] + (c[2] + c[3] * th) * th) * th) + (d[0] + d[1] * th) * (s - 35)) * pprime
        + (e[0] + (e[1] + e[2] * th) * th) * pprime * pprime
    )

    # Theta2.
    del_th = del_P * adtg
    th = th + (1 - 1 / 2**0.5) * (del_th - q)
    q = (2 - 2**0.5) * del_th + (-2 + 3 / 2**0.5) * q

    # Theta3.
    adtg = (
        a[0]
        + (a[1] + (a[2] + a[3] * th) * th) * th
        + (b[0] + b[1] * th) * (s - 35)
        + ((c[0] + (c[1] + (c[2] + c[3] * th) * th) * th) + (d[0] + d[1] * th) * (s - 35)) * pprime
        + (e[0] + (e[1] + e[2] * th) * th) * pprime * pprime
    )

    del_th = del_P * adtg
    th = th + (1 + 1 / 2**0.5) * (del_th - q)
    q = (2 + 2**0.5) * del_th + (-2 - 3 / 2**0.5) * q

    # Theta4.
    pprime = pres + del_P
    adtg = (
        a[0]
        + (a[1] + (a[2] + a[3] * th) * th) * th
        + (b[0] + b[1] * th) * (s - 35)
        + ((c[0] + (c[1] + (c[2] + c[3] * th) * th) * th) + (d[0] + d[1] * th) * (s - 35)) * pprime
        + (e[0] + (e[1] + e[2] * th) * th) * pprime * pprime
    )

    del_th = del_P * adtg

    particle.temp = (th + (del_th - 2 * q) / 6) / 1.00024


def UNESCODensity(particle, fieldset, time):  # pragma: no cover
    # This is a kernel which calculates the UNESCO density
    # (https://link.springer.com/content/pdf/bbm%3A978-3-319-18908-6%2F1.pdf),
    # from pressure, temperature and salinity.
    # density in [kg/m3] if temperature in degrees C, salinity in PSU,
    # pressure in bar.

    a0 = 999.842594
    a1 = 0.06793953
    a2 = -0.009095290
    a3 = 0.0001001685
    a4 = -0.000001120083
    a5 = 0.000000006536332

    S = fieldset.psu_salinity[time, particle.depth, particle.lat, particle.lon]  # salinity
    T = fieldset.cons_temperature[time, particle.depth, particle.lat, particle.lon]  # temperature
    P = fieldset.cons_pressure[time, particle.depth, particle.lat, particle.lon]  # pressure

    rsmow = a0 + a1 * T + a2 * math.pow(T, 2) + a3 * math.pow(T, 3) + a4 * math.pow(T, 4) + a5 * math.pow(T, 5)

    b0 = 0.82449
    b1 = -0.0040899
    b2 = 0.000076438
    b3 = -0.00000082467
    b_four = 0.0000000053875

    c0 = -0.0057246
    c1 = 0.00010227
    c2 = -0.0000016546

    d0 = 0.00048314

    B1 = b0 + b1 * T + b2 * math.pow(T, 2) + b3 * math.pow(T, 3) + b_four * math.pow(T, 4)
    C1 = c0 + c1 * T + c2 * math.pow(T, 2)

    rho_st0 = rsmow + B1 * S + C1 * math.pow(S, 1.5) + d0 * math.pow(S, 2)

    e0 = 19652.21
    e1 = 148.4206
    e2 = -2.327105
    e3 = 0.01360477
    e4 = -0.00005155288

    f0 = 54.6746
    f1 = -0.603459
    f2 = 0.01099870
    f3 = -0.00006167

    g0 = 0.07944
    g1 = 0.016483
    g2 = -0.00053009

    Kw = e0 + e1 * T + e2 * math.pow(T, 2) + e3 * math.pow(T, 3) + e4 * math.pow(T, 4)
    F1 = f0 + f1 * T + f2 * math.pow(T, 2) + f3 * math.pow(T, 3)
    G1 = g0 + g1 * T + g2 * math.pow(T, 2)

    K_ST0 = Kw + F1 * S + G1 * math.pow(S, 1.5)

    h0 = 3.2399
    h1 = 0.00143713
    h2 = 0.000116092
    h3 = -0.000000577905

    i0 = 0.0022838
    i1 = -0.000010981
    i2 = -0.0000016078

    j0 = 0.000191075

    k0 = 0.0000850935
    k1 = -0.00000612293
    k2 = 0.000000052787

    m0 = -0.00000099348
    m1 = 0.000000020816
    m2 = 0.00000000091697

    Aw = h0 + h1 * T + h2 * math.pow(T, 2) + h3 * math.pow(T, 3)
    A1 = Aw + (i0 + i1 * T + i2 * math.pow(T, 2)) * S + j0 * math.pow(S, 1.5)
    Bw = k0 + k1 * T + k2 * math.pow(T, 2)
    B2 = Bw + (m0 + m1 * T + m2 * math.pow(T, 2)) * S

    K_STp = K_ST0 + A1 * P + B2 * math.pow(T, 2)

    particle.density = rho_st0 / (1 - (P / K_STp))

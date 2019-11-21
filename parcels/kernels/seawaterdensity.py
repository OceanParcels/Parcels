"""Collection of pre-built sea water density kernels"""
import math


__all__ = ['polyTEOS10_bsq', 'UNESCO_Density']


def polyTEOS10_bsq(particle, fieldset, time):
    # calculates density based on the polyTEOS10-bsq algorithm from Appendix A.2 of
    # https://www.sciencedirect.com/science/article/pii/S1463500315000566
    # requires fieldset.abs_salinity and fieldset.cons_temperature Fields in the fieldset
    # and a particle.density Variable in the ParticleSet
    #
    # References:
    #  Roquet, F., Madec, G., McDougall, T. J., Barker, P. M., 2014: Accurate
    #   polynomial expressions for the density and specific volume of
    #   seawater using the TEOS-10 standard. Ocean Modelling.
    #  McDougall, T. J., D. R. Jackett, D. G. Wright and R. Feistel, 2003:
    #   Accurate and computationally efficient algorithms for potential
    #   temperature and density of seawater.  Journal of Atmospheric and
    #   Oceanic Technology, 20, 730-741.

    Z = - particle.depth  # note: use negative depths!
    SA = fieldset.abs_salinity[time, particle.depth, particle.lat, particle.lon]
    CT = fieldset.cons_temperature[time, particle.depth, particle.lat, particle.lon]

    SAu = 40 * 35.16504 / 35
    CTu = 40
    Zu = 1e4
    deltaS = 32
    R000 = 8.0189615746e+02
    R100 = 8.6672408165e+02
    R200 = -1.7864682637e+03
    R300 = 2.0375295546e+03
    R400 = -1.2849161071e+03
    R500 = 4.3227585684e+02
    R600 = -6.0579916612e+01
    R010 = 2.6010145068e+01
    R110 = -6.5281885265e+01
    R210 = 8.1770425108e+01
    R310 = -5.6888046321e+01
    R410 = 1.7681814114e+01
    R510 = -1.9193502195e+00
    R020 = -3.7074170417e+01
    R120 = 6.1548258127e+01
    R220 = -6.0362551501e+01
    R320 = 2.9130021253e+01
    R420 = -5.4723692739e+00
    R030 = 2.1661789529e+01
    R130 = -3.3449108469e+01
    R230 = 1.9717078466e+01
    R330 = -3.1742946532e+00
    R040 = -8.3627885467e+00
    R140 = 1.1311538584e+01
    R240 = -5.3563304045e+00
    R050 = 5.4048723791e-01
    R150 = 4.8169980163e-01
    R060 = -1.9083568888e-01
    R001 = 1.9681925209e+01
    R101 = -4.2549998214e+01
    R201 = 5.0774768218e+01
    R301 = -3.0938076334e+01
    R401 = 6.6051753097e+00
    R011 = -1.3336301113e+01
    R111 = -4.4870114575e+00
    R211 = 5.0042598061e+00
    R311 = -6.5399043664e-01
    R021 = 6.7080479603e+00
    R121 = 3.5063081279e+00
    R221 = -1.8795372996e+00
    R031 = -2.4649669534e+00
    R131 = -5.5077101279e-01
    R041 = 5.5927935970e-01
    R002 = 2.0660924175e+00
    R102 = -4.9527603989e+00
    R202 = 2.5019633244e+00
    R012 = 2.0564311499e+00
    R112 = -2.1311365518e-01
    R022 = -1.2419983026e+00
    R003 = -2.3342758797e-02
    R103 = -1.8507636718e-02
    R013 = 3.7969820455e-01
    ss = math.sqrt((SA + deltaS) / SAu)
    tt = CT / CTu
    zz = -Z / Zu
    rz3 = R013 * tt + R103 * ss + R003
    rz2 = (R022 * tt + R112 * ss + R012) * tt + (R202 * ss + R102) * ss + R002
    rz1 = (((R041 * tt + R131 * ss + R031) * tt + (R221 * ss + R121) * ss + R021) * tt + ((R311 * ss + R211) * ss + R111) * ss + R011) * tt + (((R401 * ss + R301) * ss + R201) * ss + R101) * ss + R001
    rz0 = (((((R060 * tt + R150 * ss + R050) * tt + (R240 * ss + R140) * ss + R040) * tt + ((R330 * ss + R230) * ss + R130) * ss + R030) * tt + (((R420 * ss + R320) * ss + R220) * ss + R120) * ss + R020) * tt + ((((R510 * ss + R410) * ss + R310) * ss + R210) * ss + R110) * ss + R010) * tt + (((((R600 * ss + R500) * ss + R400) * ss + R300) * ss + R200) * ss + R100) * ss + R000
    particle.density = ((rz3 * zz + rz2) * zz + rz1) * zz + rz0


def UNESCO_Density(particle, fieldset, time):
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
    p = fieldset.cons_pressure[time, particle.depth, particle.lat, particle.lon]  # pressure

    rsmow = a0 + a1*T + a2*math.pow(T, 2) + a3*math.pow(T, 3) +     \
        a4*math.pow(T, 4) + a5*math.pow(T, 5)

    b0 = 0.82449
    b1 = -0.0040899
    b2 = 0.000076438
    b3 = -0.00000082467
    b_four = 0.0000000053875

    c0 = -0.0057246
    c1 = 0.00010227
    c2 = -0.0000016546

    d0 = 0.00048314

    B1 = b0 + b1*T + b2*math.pow(T, 2) + b3*math.pow(T, 3) + b_four*math.pow(T, 4)
    C1 = c0 + c1*T + c2*math.pow(T, 2)

    rho_st0 = rsmow + B1*S + C1*math.pow(S, 1.5) + d0*math.pow(S, 2)

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

    Kw = e0 + e1*T + e2*math.pow(T, 2) + e3*math.pow(T, 3) + e4*math.pow(T, 4)
    F1 = f0 + f1*T + f2*math.pow(T, 2) + f3*math.pow(T, 3)
    G1 = g0 + g1*T + g2*math.pow(T, 2)

    K_ST0 = Kw + F1*S + G1*math.pow(S, 1.5)

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

    Aw = h0 + h1*T + h2*math.pow(T, 2) + h3*math.pow(T, 3)
    A1 = Aw + (i0 + i1*T + i2*math.pow(T, 2))*S + j0*math.pow(S, 1.5)
    Bw = k0 + k1*T + k2*math.pow(T, 2)
    B2 = Bw + (m0 + m1*T + m2*math.pow(T, 2))*S

    K_STp = K_ST0 + A1*p + B2*math.pow(T, 2)

    particle.density = rho_st0/(1-(p/K_STp))

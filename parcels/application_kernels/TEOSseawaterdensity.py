"""Collection of pre-built sea water density kernels."""

import math

__all__ = ["PolyTEOS10_bsq"]


def PolyTEOS10_bsq(particle, fieldset, time):  # pragma: no cover
    """Calculates density based on the polyTEOS10-bsq algorithm from Appendix A.2 of
    https://www.sciencedirect.com/science/article/pii/S1463500315000566
    requires fieldset.abs_salinity and fieldset.cons_temperature Fields in the fieldset
    and a particle.density Variable in the ParticleSet

    References
    ----------
    1. Roquet, F., Madec, G., McDougall, T. J., Barker, P. M., 2014: Accurate
       polynomial expressions for the density and specific volume of
       seawater using the TEOS-10 standard. Ocean Modelling.

    2. McDougall, T. J., D. R. Jackett, D. G. Wright and R. Feistel, 2003:
       Accurate and computationally efficient algorithms for potential
       temperature and density of seawater.  Journal of Atmospheric and
       Oceanic Technology, 20, 730-741.

    """
    Z = -math.fabs(particle.depth)  # Z needs to be negative
    SA = fieldset.abs_salinity[time, particle.depth, particle.lat, particle.lon]
    CT = fieldset.cons_temperature[time, particle.depth, particle.lat, particle.lon]

    SAu = 40 * 35.16504 / 35
    CTu = 40
    Zu = 1e4
    deltaS = 32
    R000 = 8.0189615746e02
    R100 = 8.6672408165e02
    R200 = -1.7864682637e03
    R300 = 2.0375295546e03
    R400 = -1.2849161071e03
    R500 = 4.3227585684e02
    R600 = -6.0579916612e01
    R010 = 2.6010145068e01
    R110 = -6.5281885265e01
    R210 = 8.1770425108e01
    R310 = -5.6888046321e01
    R410 = 1.7681814114e01
    R510 = -1.9193502195e00
    R020 = -3.7074170417e01
    R120 = 6.1548258127e01
    R220 = -6.0362551501e01
    R320 = 2.9130021253e01
    R420 = -5.4723692739e00
    R030 = 2.1661789529e01
    R130 = -3.3449108469e01
    R230 = 1.9717078466e01
    R330 = -3.1742946532e00
    R040 = -8.3627885467e00
    R140 = 1.1311538584e01
    R240 = -5.3563304045e00
    R050 = 5.4048723791e-01
    R150 = 4.8169980163e-01
    R060 = -1.9083568888e-01
    R001 = 1.9681925209e01
    R101 = -4.2549998214e01
    R201 = 5.0774768218e01
    R301 = -3.0938076334e01
    R401 = 6.6051753097e00
    R011 = -1.3336301113e01
    R111 = -4.4870114575e00
    R211 = 5.0042598061e00
    R311 = -6.5399043664e-01
    R021 = 6.7080479603e00
    R121 = 3.5063081279e00
    R221 = -1.8795372996e00
    R031 = -2.4649669534e00
    R131 = -5.5077101279e-01
    R041 = 5.5927935970e-01
    R002 = 2.0660924175e00
    R102 = -4.9527603989e00
    R202 = 2.5019633244e00
    R012 = 2.0564311499e00
    R112 = -2.1311365518e-01
    R022 = -1.2419983026e00
    R003 = -2.3342758797e-02
    R103 = -1.8507636718e-02
    R013 = 3.7969820455e-01
    ss = math.sqrt((SA + deltaS) / SAu)
    tt = CT / CTu
    zz = -Z / Zu
    rz3 = R013 * tt + R103 * ss + R003
    rz2 = (R022 * tt + R112 * ss + R012) * tt + (R202 * ss + R102) * ss + R002
    rz1 = (((R041 * tt + R131 * ss + R031) * tt + (R221 * ss + R121) * ss + R021) * tt + ((R311 * ss + R211) * ss + R111) * ss + R011) * tt + (((R401 * ss + R301) * ss + R201) * ss + R101) * ss + R001  # fmt: skip
    rz0 = (((((R060 * tt + R150 * ss + R050) * tt + (R240 * ss + R140) * ss + R040) * tt + ((R330 * ss + R230) * ss + R130) * ss + R030) * tt + (((R420 * ss + R320) * ss + R220) * ss + R120) * ss + R020) * tt + ((((R510 * ss + R410) * ss + R310) * ss + R210) * ss + R110) * ss + R010) * tt + (((((R600 * ss + R500) * ss + R400) * ss + R300) * ss + R200) * ss + R100) * ss + R000  # fmt: skip
    particle.density = ((rz3 * zz + rz2) * zz + rz1) * zz + rz0

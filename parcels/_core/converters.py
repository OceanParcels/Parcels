from __future__ import annotations

from math import pi

import numpy as np
import numpy.typing as npt

__all__ = [
    "Geographic",
    "GeographicPolar",
    "GeographicPolarSquare",
    "GeographicSquare",
    "UnitConverter",
    "_convert_to_flat_array",
    "_unitconverters_map",
]


def _convert_to_flat_array(var: npt.ArrayLike) -> npt.NDArray:
    """Convert lists and single integers/floats to one-dimensional numpy arrays

    Parameters
    ----------
    var : Array
        list or numeric to convert to a one-dimensional numpy array
    """
    return np.array(var).flatten()


class UnitConverter:
    """Interface class for spatial unit conversion during field sampling that performs no conversion."""

    source_unit: str | None = None
    target_unit: str | None = None

    def to_target(self, value, z, y, x):
        return value

    def to_source(self, value, z, y, x):
        return value


class Geographic(UnitConverter):
    """Unit converter from geometric to geographic coordinates (m to degree)"""

    source_unit = "m"
    target_unit = "degree"

    def to_target(self, value, z, y, x):
        return value / 1000.0 / 1.852 / 60.0

    def to_source(self, value, z, y, x):
        return value * 1000.0 * 1.852 * 60.0


class GeographicPolar(UnitConverter):
    """Unit converter from geometric to geographic coordinates (m to degree)
    with a correction to account for narrower grid cells closer to the poles.
    """

    source_unit = "m"
    target_unit = "degree"

    def to_target(self, value, z, y, x):
        return value / 1000.0 / 1.852 / 60.0 / np.cos(y * pi / 180)

    def to_source(self, value, z, y, x):
        return value * 1000.0 * 1.852 * 60.0 * np.cos(y * pi / 180)


class GeographicSquare(UnitConverter):
    """Square distance converter from geometric to geographic coordinates (m2 to degree2)"""

    source_unit = "m2"
    target_unit = "degree2"

    def to_target(self, value, z, y, x):
        return value / pow(1000.0 * 1.852 * 60.0, 2)

    def to_source(self, value, z, y, x):
        return value * pow(1000.0 * 1.852 * 60.0, 2)


class GeographicPolarSquare(UnitConverter):
    """Square distance converter from geometric to geographic coordinates (m2 to degree2)
    with a correction to account for narrower grid cells closer to the poles.
    """

    source_unit = "m2"
    target_unit = "degree2"

    def to_target(self, value, z, y, x):
        return value / pow(1000.0 * 1.852 * 60.0 * np.cos(y * pi / 180), 2)

    def to_source(self, value, z, y, x):
        return value * pow(1000.0 * 1.852 * 60.0 * np.cos(y * pi / 180), 2)


_unitconverters_map = {
    "U": GeographicPolar(),
    "V": Geographic(),
    "Kh_zonal": GeographicPolarSquare(),
    "Kh_meridional": GeographicSquare(),
}

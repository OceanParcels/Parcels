import itertools

import numpy as np

_FULL_MASK = slice(None)


class ParticleData:
    """Class to encapsulate and provide access to particle data based on a mask variable.

    Notes
    -----
    Dev note: No public methods or attributes should be defined in this class, instead being defined as helper functions in this class. This is to avoid shadowing user provided attribute names (as the public attributes on this class correspond directly with the particle variables defined by the user).

    """

    def __init__(self, data: dict[str, np.ndarray], mask=_FULL_MASK):
        _assert_particledata_same_nparticles(data)

        self._data = data
        self._mask = mask

    def _set_mask(self, mask):
        self._mask = mask

    def __getattr__(self, name):
        return self._data[name][self._mask]

    def __setattr__(self, name, value):
        if name in ["_data", "_mask"]:
            object.__setattr__(self, name, value)
        else:
            self._data[name][self._mask] = value


def _assert_particledata_same_nparticles(data):
    for (left_name, left_array), (right_name, right_array) in itertools.pairwise(data.items()):
        nparticles_left = left_array.shape[0]
        nparticles_right = right_array.shape[0]
        if nparticles_left != nparticles_right:
            raise ValueError(
                f"Particle data has mismatching number of particles. Attribute {left_name!r} has {nparticles_left} particles, but attribute {right_name!r} has {nparticles_left} particles"
            )


my_dict = {
    "x": np.array([1, 2, 3]),
    "y": np.array([1, 2, 3]),
}
pd = ParticleData(my_dict)

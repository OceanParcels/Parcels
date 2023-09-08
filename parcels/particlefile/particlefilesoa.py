"""Module controlling the writing of ParticleSets to NetCDF file"""
import numpy as np

from parcels.particlefile.baseparticlefile import BaseParticleFile

__all__ = ['ParticleFileSOA']


class ParticleFileSOA(BaseParticleFile):
    """Initialise trajectory output.

    Parameters
    ----------
    name : str
        Basename of the output file.  This can also be a Zarr store.
    particleset :
        ParticleSet to output
    outputdt :
        Interval which dictates the update frequency of file output
        while ParticleFile is given as an argument of ParticleSet.execute()
        It is either a timedelta object or a positive double.
    chunks :
        Tuple (trajs, obs) to control the size of chunks in the zarr output.

    Returns
    -------
    ParticleFileSOA
        ParticleFile object that can be used to write particle data to file

    """

    def __init__(self, name, particleset, outputdt=np.infty, chunks=None, write_ondelete=False):
        super().__init__(
            name=name, particleset=particleset, outputdt=outputdt, chunks=chunks)

    def _reserved_var_names(self):
        """Returns the reserved dimension names not to be written just once."""
        return ['time', 'lat', 'lon', 'depth', 'id']

"""Module controlling the writing of ParticleSets to NetCDF file"""
import numpy as np

from parcels.particlefile.baseparticlefile import BaseParticleFile

__all__ = ['ParticleFileSOA']


class ParticleFileSOA(BaseParticleFile):
    """Initialise trajectory output.

    Parameters
    ----------
    name :
        Basename of the output file.  This can also be a Zarr store.
    particleset :
        ParticleSet to output
    outputdt :
        Interval which dictates the update frequency of file output
        while ParticleFile is given as an argument of ParticleSet.execute()
        It is either a timedelta object or a positive double.
    chunks :
        Tuple (trajs, obs) to control the size of chunks in the zarr output.
    write_ondelete :
        Boolean to write particle data only when they are deleted. Default is False

    Returns
    -------
    ParticleFileSOA
        ParticleFile object that can be used to write particle data to file

    """

    def __init__(self, name, particleset, outputdt=np.infty, chunks=None, write_ondelete=False):
        super().__init__(
            name=name, particleset=particleset, outputdt=outputdt, chunks=chunks, write_ondelete=write_ondelete
        )

    def __del__(self):
        super().__del__()

    def _reserved_var_names(self):
        """Returns the reserved dimension names not to be written just once."""
        return ['time', 'lat', 'lon', 'depth', 'id']

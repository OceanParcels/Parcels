"""Module controlling the writing of ParticleSets to NetCDF file"""
import numpy as np

from parcels.particlefile.baseparticlefile import BaseParticleFile

__all__ = ["ParticleFileSOA"]


class ParticleFileSOA(BaseParticleFile):
    """Initialise trajectory output.

    :param name: Basename of the output file
    :param particleset: ParticleSet to output
    :param outputdt: Interval which dictates the update frequency of file output
                     while ParticleFile is given as an argument of ParticleSet.execute()
                     It is either a timedelta object or a positive double.
    :param chunks: Tuple (trajs, obs) to control the size of chunks in the zarr output.
    :param write_ondelete: Boolean to write particle data only when they are deleted. Default is False
    :param create_new_zarrfile: Boolean to determine if we need to set up a new Zarr store. Default is True
    """

    def __init__(
        self,
        name,
        particleset,
        outputdt=np.infty,
        chunks=None,
        write_ondelete=False,
        create_new_zarrfile=True,
    ):
        super(ParticleFileSOA, self).__init__(
            name=name,
            particleset=particleset,
            outputdt=outputdt,
            chunks=chunks,
            write_ondelete=write_ondelete,
            create_new_zarrfile=create_new_zarrfile,
        )

    def __del__(self):
        super(ParticleFileSOA, self).__del__()

    def _reserved_var_names(self):
        """
        returns the reserved dimension names not to be written just once.
        """
        return ["time", "lat", "lon", "depth", "id"]

import warnings

__all__ = ["FieldSetWarning", "FileWarning", "KernelWarning", "ParticleSetWarning"]


class FieldSetWarning(UserWarning):
    """Warning that is raised when there are issues in the construction of the FieldSet or its Grid.

    These warnings are often caused by issues in the input data dimensions
    or options selected when loading data into a FieldSet.
    """

    pass


class ParticleSetWarning(UserWarning):
    """Warning that is raised when there are issues in the construction of the ParticleSet."""

    pass


class FileWarning(UserWarning):
    """Warning that is raised when there are issues with input or output files.

    These warnings can be related to file chunking, naming, or decoding issues.
    Chunking issues in particular may negatively impact performance
    (see also https://docs.oceanparcels.org/en/latest/examples/documentation_MPI.html#Chunking-the-FieldSet-with-dask)
    """

    pass


class KernelWarning(RuntimeWarning):
    """Warning that is raised when there are issues with the Kernel.

    These warnings often result from issues in the FieldSet or user-defined Kernel
    that are passed into the Parcels Kernel loop.
    """

    pass


def _deprecated_param_netcdf_decodewarning():
    warnings.warn(
        "The 'netcdf_decodewarning' argument is deprecated in v3.1.0 and will be removed completely in a future release. "
        "The parameter no longer has any effect, please use the Python warnings module to control warnings, "
        "e.g., warnings.filterwarnings('ignore', category=parcels.FileWarning). "
        "See also https://docs.oceanparcels.org/en/latest/examples/tutorial_nemo_3D.html",
        DeprecationWarning,
        stacklevel=2,
    )

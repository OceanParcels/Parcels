import warnings

from parcels._fieldfilebuffer import DaskFileBuffer, DeferredDaskFileBuffer, NetcdfFileBuffer, _FileBuffer

__all__ = ["DaskFileBuffer", "DeferredDaskFileBuffer", "NetcdfFileBuffer", "_FileBuffer"]

warnings.warn(
    "The `parcels.fieldfilebuffer` module is deprecated as it has been marked private. "
    "Users are not expected to use it in their scripts. See https://github.com/OceanParcels/Parcels/issues/1773 "
    "to continue discussion.",
    DeprecationWarning,
    stacklevel=2,
)  # TODO: Remove 6 months after v3.1.1

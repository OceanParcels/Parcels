import os
import subprocess
import warnings

try:
    from parcels._version_setup import version
except ModuleNotFoundError:
    try:
        version = (
            subprocess.check_output(
                ["git", "-C", os.path.dirname(__file__), "describe", "--tags"],
                stderr=subprocess.PIPE,
            )
            .decode("ascii")
            .strip()
        )
    except subprocess.SubprocessError:
        warnings.warn(
            "Looks like you're trying to do a development install of parcels. "
            "This needs to be in a git repo so that version information is available. "
            "Setting version to 'unknown'.",
            stacklevel=2,
        )
        version = "unknown"

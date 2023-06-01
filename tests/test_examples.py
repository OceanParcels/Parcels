import runpy
import sys
from pathlib import Path

import pytest

# TODO: Broaden scope to all `.py` scripts in `example_folder`?
example_folder = (Path(__file__).parent / "../docs/examples").resolve()
example_fnames = [
    "example_decaying_moving_eddy.py",
    "example_moving_eddies.py",
    # "example_nemo_curvilinear.py", #! Imports cartopy, which is not installed in CI
    "example_peninsula.py",
    "example_radial_rotation.py",
    "example_stommel.py",
]


@pytest.mark.parametrize("example_fname", example_fnames)
def test_example_script(example_fname):
    script = str(example_folder / example_fname)

    # Clear sys.argv, otherwise pytest pollutes it with its own arguments.
    sys.argv = [sys.argv[0]]

    runpy.run_path(script, run_name="__main__")

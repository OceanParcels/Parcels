import runpy
import sys
from pathlib import Path

import pytest

example_folder = (Path(__file__).parent / "../docs/examples").resolve()
example_fnames = [path.name for path in example_folder.glob("*.py")]


@pytest.mark.parametrize("example_fname", example_fnames)
def test_example_script(example_fname):
    script = str(example_folder / example_fname)

    # Clear sys.argv, otherwise pytest pollutes it with its own arguments.
    sys.argv = [sys.argv[0]]

    runpy.run_path(script, run_name="__main__")

import os
import runpy
import shutil
import sys
import time
from pathlib import Path

import pytest

example_folder = (Path(__file__).parent / "../docs/examples").resolve()
example_fnames = [path.name for path in example_folder.glob("*.py")]


@pytest.fixture(autouse=True)
def cleanup_generated_data_files():
    """Clean up generated data files from test run.

    Records current folder contents before test, and cleans up any generated `.nc` files
    and `.zarr` folders afterwards. For safety this is non-recursive. This function is
    only necessary as the scripts being run aren't native pytest tests, so they don't
    have access to the `tmpdir` fixture.

    """
    folder_contents = os.listdir()
    yield
    time.sleep(0.1)  # Buffer so that files are closed before we try to delete them.
    for fname in os.listdir():
        if fname in folder_contents:
            continue
        if not (fname.endswith(".nc") or fname.endswith(".zarr")):
            continue

        path = Path(fname)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        print(f"Removed {path}")


@pytest.mark.parametrize("example_fname", example_fnames)
def test_example_script(example_fname):
    script = str(example_folder / example_fname)

    # Clear sys.argv, otherwise pytest pollutes it with its own arguments.
    sys.argv = [sys.argv[0]]

    runpy.run_path(script, run_name="__main__")

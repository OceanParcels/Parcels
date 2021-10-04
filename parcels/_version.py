import subprocess
import os
version = subprocess.check_output(['git', '-C', os.path.dirname(__file__), 'describe', '--tags']).decode('ascii').strip()

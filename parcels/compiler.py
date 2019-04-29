import subprocess
from os import path, getenv
from tempfile import gettempdir
from struct import calcsize
try:
    from pathlib import Path
except ImportError:
    from pathlib2 import Path  # python 2 backport
try:
    from os import getuid
except:
    # Windows does not have getuid(), so define to simply return 'tmp'
    def getuid():
        return 'tmp'


def get_package_dir():
    return path.abspath(path.dirname(__file__))


def get_cache_dir():
    directory = path.join(gettempdir(), "parcels-%s" % getuid())
    Path(directory).mkdir(exist_ok=True)
    return directory


class Compiler(object):
    """A compiler object for creating and loading shared libraries.

    :arg cc: C compiler executable (uses environment variable ``CC`` if not provided).
    :arg cppargs: A list of arguments to the C compiler (optional).
    :arg ldargs: A list of arguments to the linker (optional)."""

    def __init__(self, cc=None, cppargs=None, ldargs=None):
        if cppargs is None:
            cppargs = []
        if ldargs is None:
            ldargs = []

        self._cc = getenv('CC') if cc is None else cc
        self._cppargs = cppargs
        self._ldargs = ldargs

    def compile(self, src, obj, log):
        cc = [self._cc] + self._cppargs + ['-o', obj, src] + self._ldargs
        with open(log, 'w') as logfile:
            logfile.write("Compiling: %s\n" % " ".join(cc))
            try:
                subprocess.check_call(cc, stdout=logfile, stderr=logfile)
            except OSError:
                err = """OSError during compilation
Please check if compiler exists: %s""" % self._cc
                raise RuntimeError(err)
            except subprocess.CalledProcessError:
                with open(log, 'r') as logfile2:
                    err = """Error during compilation:
Compilation command: %s
Source file: %s
Log file: %s

Log output: %s""" % (" ".join(cc), src, logfile.name, logfile2.read())
                raise RuntimeError(err)


class GNUCompiler(Compiler):
    """A compiler object for the GNU Linux toolchain.

    :arg cppargs: A list of arguments to pass to the C compiler
         (optional).
    :arg ldargs: A list of arguments to pass to the linker (optional)."""
    def __init__(self, cppargs=None, ldargs=None):
        if cppargs is None:
            cppargs = []
        if ldargs is None:
            ldargs = []

        opt_flags = ['-g', '-O3']
        arch_flag = ['-m64' if calcsize("P") == 8 else '-m32']
        cppargs = ['-Wall', '-fPIC', '-I%s' % path.join(get_package_dir(), 'include')] + opt_flags + cppargs
        cppargs += arch_flag
        ldargs = ['-shared'] + ldargs + arch_flag
        super(GNUCompiler, self).__init__("gcc", cppargs=cppargs, ldargs=ldargs)

import os
import subprocess
from struct import calcsize

try:
    from mpi4py import MPI
except:
    MPI = None

class CCompiler(object):
    """A compiler object for creating and loading shared libraries.

    :arg cc: C compiler executable (uses environment variable ``CC`` if not provided).
    :arg cppargs: A list of arguments to the C compiler (optional).
    :arg ldargs: A list of arguments to the linker (optional)."""

    # support_libraries = []
    # support_inc_folders = []
    # support_lib_folders = []

    def __init__(self, cc=None, cppargs=None, ldargs=None, incdirs=None, libdirs=None, libs=None):
        if cppargs is None:
            cppargs = []
        if ldargs is None:
            ldargs = []
        #if incdirs is None:
        #    incdirs = []
        #if libdirs is None:
        #    libdirs = []
        #if libs is None:
        #    libs = []

        self._cc = os.getenv('CC') if cc is None else cc
        self._cppargs = cppargs
        self._ldargs = ldargs

        # self.support_inc_folders += incdirs
        # self.support_lib_folders += libdirs
        # self.support_libraries += libs

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


class GNUCompiler(CCompiler):
    """A compiler object for the GNU Linux toolchain.

    :arg cppargs: A list of arguments to pass to the C compiler
         (optional).
    :arg ldargs: A list of arguments to pass to the linker (optional)."""
    def __init__(self, cppargs=None, ldargs=None, incdirs=None, libdirs=None, libs=None):
        if cppargs is None:
            cppargs = []
        if ldargs is None:
            ldargs = []

        Iflags = []
        if incdirs is not None and isinstance(incdirs, list):
            for i, dir in enumerate(incdirs):
                Iflags.append("-I"+dir)
        Lflags = []
        if libdirs is not None and isinstance(libdirs, list):
            for i, dir in enumerate(libdirs):
                Lflags.append("-L"+dir)
        lflags = []
        if libs is not None and isinstance(libs, list):
            for i, lib in enumerate(libs):
                lflags.append("-l" +lib)

        opt_flags = ['-g', '-O3']
        arch_flag = ['-m64' if calcsize("P") == 8 else '-m32']
        cppargs = ['-Wall', '-fPIC'] + Iflags + opt_flags + cppargs
        cppargs += arch_flag
        ldargs = ['-shared'] + Lflags + lflags + ldargs + arch_flag
        #compiler = "mpicc" if MPI else "gcc"
        cc_env = os.getenv('CC')
        compiler = "mpicc" if MPI else "gcc" if cc_env is None else cc_env

        super(GNUCompiler, self).__init__(compiler, cppargs=cppargs, ldargs=ldargs, incdirs=incdirs, libdirs=libdirs, libs=libs)

    def compile(self, src, obj, log):
        lib_pathfile = os.path.basename(obj)
        lib_pathdir = os.path.dirname(obj)
        if lib_pathfile[0:3] != "lib":
            lib_pathfile = "lib"+lib_pathfile
            obj = os.path.join(lib_pathdir, lib_pathfile)

        super(GNUCompiler, self).compile(src, obj, log)
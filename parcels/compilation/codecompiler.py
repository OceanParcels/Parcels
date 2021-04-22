import os
import subprocess
from struct import calcsize

try:
    from mpi4py import MPI
except:
    MPI = None


class Compiler_parameters(object):
    def __init__(self):
        self._compiler = ""
        self._cppargs = []
        self._ldargs = []
        self._incdirs = []
        self._libdirs = []
        self._libs = []
        self._dynlib_ext = ""
        self._stclib_ext = ""
        self._obj_ext = ""
        self._exe_ext = ""

    @property
    def compiler(self):
        return self._compiler

    @property
    def cppargs(self):
        return self._cppargs

    @property
    def ldargs(self):
        return self._ldargs

    @property
    def incdirs(self):
        return self._incdirs

    @property
    def libdirs(self):
        return self._libdirs

    @property
    def libs(self):
        return self._libs

    @property
    def dynlib_ext(self):
        return self._dynlib_ext

    @property
    def stclib_ext(self):
        return self._stclib_ext

    @property
    def obj_ext(self):
        return self._obj_ext

    @property
    def exe_ext(self):
        return self._exe_ext


class GNU_parameters(Compiler_parameters):
    def __init__(self, cppargs=None, ldargs=None, incdirs=None, libdirs=None, libs=None):
        super(GNU_parameters, self).__init__()
        if cppargs is None:
            cppargs = []
        if ldargs is None:
            ldargs = []
        if incdirs is None:
            incdirs = []
        if libdirs is None:
            libdirs = []
        if libs is None:
            libs = []
        libs.append("m")

        Iflags = []
        if isinstance(incdirs, list):
            for i, dir in enumerate(incdirs):
                Iflags.append("-I"+dir)
        Lflags = []
        if isinstance(libdirs, list):
            for i, dir in enumerate(libdirs):
                Lflags.append("-L"+dir)
        lflags = []
        if isinstance(libs, list):
            for i, lib in enumerate(libs):
                lflags.append("-l" + lib)

        cc_env = os.getenv('CC')
        mpicc = None
        if MPI:
            mpicc_env = os.getenv('MPICC')
            mpicc = mpicc_env
            mpicc = "mpicc" if mpicc is None and os._exists("mpicc") else None
            mpicc = "mpiCC" if mpicc is None and os._exists("mpiCC") else None
            os.system("%s --version" % (mpicc))
        self._compiler = mpicc if MPI and mpicc is not None else cc_env if cc_env is not None else "gcc"
        opt_flags = ['-g', '-O3']
        arch_flag = ['-m64' if calcsize("P") == 8 else '-m32']
        self._cppargs = ['-Wall', '-fPIC', '-std=gnu11']
        self._cppargs += Iflags
        self._cppargs += opt_flags + cppargs + arch_flag
        self._ldargs = ['-shared']
        self._ldargs += Lflags
        self._ldargs += lflags
        self._ldargs += ldargs
        if len(Lflags) > 0:
            self._ldargs += ['-Wl, -rpath=%s' % (":".join(libdirs))]
        self._ldargs += arch_flag
        self._incdirs = incdirs
        self._libdirs = libdirs
        self._libs = libs
        self._dynlib_ext = "so"
        self._stclib_ext = "a"
        self._obj_ext = "o"
        self._exe_ext = ""


class Clang_parameters(Compiler_parameters):
    def __init__(self, cppargs=None, ldargs=None, incdirs=None, libdirs=None, libs=None):
        super(Clang_parameters, self).__init__()
        if cppargs is None:
            cppargs = []
        if ldargs is None:
            ldargs = []
        if incdirs is None:
            incdirs = []
        if libdirs is None:
            libdirs = []
        if libs is None:
            libs = []
        self._compiler = "cc"
        self._cppargs = cppargs
        self._ldargs = ldargs
        self._incdirs = incdirs
        self._libdirs = libdirs
        self._libs = libs
        self._dynlib_ext = "dynlib"
        self._stclib_ext = "a"
        self._obj_ext = "o"
        self._exe_ext = "exe"


class MinGW_parameters(Compiler_parameters):
    def __init__(self, cppargs=None, ldargs=None, incdirs=None, libdirs=None, libs=None):
        super(MinGW_parameters, self).__init__()
        if cppargs is None:
            cppargs = []
        if ldargs is None:
            ldargs = []
        if incdirs is None:
            incdirs = []
        if libdirs is None:
            libdirs = []
        if libs is None:
            libs = []
        self._compiler = "gcc"
        self._cppargs = cppargs
        self._ldargs = ldargs
        self._incdirs = incdirs
        self._libdirs = libdirs
        self._libs = libs
        self._dynlib_ext = "so"
        self._stclib_ext = "a"
        self._obj_ext = "o"
        self._exe_ext = "exe"


class VS_parameters(Compiler_parameters):
    def __init__(self, cppargs=None, ldargs=None, incdirs=None, libdirs=None, libs=None):
        super(VS_parameters, self).__init__()
        if cppargs is None:
            cppargs = []
        if ldargs is None:
            ldargs = []
        if incdirs is None:
            incdirs = []
        if libdirs is None:
            libdirs = []
        if libs is None:
            libs = []
        self._compiler = "cl"
        self._cppargs = cppargs
        self._ldargs = ldargs
        self._incdirs = incdirs
        self._libdirs = libdirs
        self._libs = libs
        self._dynlib_ext = "dll"
        self._stclib_ext = "lib"
        self._obj_ext = "obj"
        self._exe_ext = "exe"


class CCompiler(object):
    """A compiler object for creating and loading shared libraries.

    :arg cc: C compiler executable (uses environment variable ``CC`` if not provided).
    :arg cppargs: A list of arguments to the C compiler (optional).
    :arg ldargs: A list of arguments to the linker (optional)."""

    def __init__(self, cc=None, cppargs=None, ldargs=None, incdirs=None, libdirs=None, libs=None, tmp_dir=os.getcwd()):
        if cppargs is None:
            cppargs = []
        if ldargs is None:
            ldargs = []

        self._cc = os.getenv('CC') if cc is None else cc
        self._cppargs = cppargs
        self._ldargs = ldargs
        self._dynlib_ext = ""
        self._stclib_ext = ""
        self._obj_ext = ""
        self._exe_ext = ""
        self._tmp_dir = tmp_dir
        self._incdirs = incdirs
        self._libdirs = libdirs  # only possible for already-compiled, external libraries
        self._libs = libs  # only possible for already-compiled, external libraries

    def compile(self, src, obj, log):
        pass

    def _create_compile_process_(self, cmd, src, log):
        with open(log, 'w') as logfile:
            try:
                subprocess.check_call(cmd, stdout=logfile, stderr=logfile)
            except OSError:
                err = """OSError during compilation
Please check if compiler exists: %s""" % self._cc
                raise RuntimeError(err)
            except subprocess.CalledProcessError:
                with open(log, 'r') as logfile2:
                    err = """Error during compilation:
Compilation command: %s
Source/Destination file: %s
Log file: %s

Log output: %s""" % (" ".join(cmd), src, logfile.name, logfile2.read())
                raise RuntimeError(err)
        return True


class CCompiler_SS(CCompiler):
    """
    single-stage C-compiler; used for a SINGLE source file
    """
    def __init__(self, cc=None, cppargs=None, ldargs=None, incdirs=None, libdirs=None, libs=None, tmp_dir=os.getcwd()):
        super(CCompiler_SS, self).__init__(cc=cc, cppargs=cppargs, ldargs=ldargs, incdirs=incdirs, libdirs=libdirs, libs=libs, tmp_dir=tmp_dir)

    def __str__(self):
        output = "[CCompiler_SS]: "
        output += "('cc': {}), ".format(self._cc)
        output += "('cppargs': {}), ".format(self._cppargs)
        output += "('ldargs': {}), ".format(self._ldargs)
        output += "('incdirs': {}), ".format(self._incdirs)
        output += "('libdirs': {}), ".format(self._libdirs)
        output += "('libs': {}), ".format(self._libs)
        output += "('tmp_dir': {}), ".format(self._tmp_dir)
        return output

    def compile(self, src, obj, log):
        cc = [self._cc] + self._cppargs + ['-o', obj, src] + self._ldargs
        with open(log, 'w') as logfile:
            logfile.write("Compiling: %s\n" % " ".join(cc))
        self._create_compile_process_(cc, src, log)


class GNUCompiler_SS(CCompiler_SS):
    """A compiler object for the GNU Linux toolchain.

    :arg cppargs: A list of arguments to pass to the C compiler
         (optional).
    :arg ldargs: A list of arguments to pass to the linker (optional)."""
    def __init__(self, cppargs=None, ldargs=None, incdirs=None, libdirs=None, libs=None, tmp_dir=os.getcwd()):
        c_params = GNU_parameters(cppargs, ldargs, incdirs, libdirs, libs)
        super(GNUCompiler_SS, self).__init__(c_params.compiler, cppargs=c_params.cppargs, ldargs=c_params.ldargs, incdirs=c_params.incdirs, libdirs=c_params.libdirs, libs=c_params.libs, tmp_dir=tmp_dir)
        self._dynlib_ext = c_params.dynlib_ext
        self._stclib_ext = c_params.stclib_ext
        self._obj_ext = c_params.obj_ext
        self._exe_ext = c_params.exe_ext

    def compile(self, src, obj, log):
        lib_pathfile = os.path.basename(obj)
        lib_pathdir = os.path.dirname(obj)
        obj = os.path.join(lib_pathdir, lib_pathfile)

        super(GNUCompiler_SS, self).compile(src, obj, log)


GNUCompiler = GNUCompiler_SS

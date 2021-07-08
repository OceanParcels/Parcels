import os  # noqa
import sys
import _ctypes
import numpy.ctypeslib as npct
from parcels.tools import get_cache_dir, get_package_dir
from .codecompiler import *  # noqa: F401
from parcels.tools.loggers import logger

try:
    from mpi4py import MPI
except:
    MPI = None

try:
    from mpi4py import MPI
except:
    MPI = None

__all__ = ['LibraryRegisterC', 'InterfaceC']


class LibraryRegisterC:
    _data = {}

    def __init__(self):
        self._data = {}

    def __del__(self):
        if self._data is None or len(self._data) <= 0:
            self._data = None
        else:
            self.clear()
            self._data = None

    def clear(self):
        # comment: we cannot pop it here, cause then there's no entry for the nodes in the self._data list to deregister from
        # while len(self._data) > 0:
        for item in self._data.items():
            # entry = self._data.popitem()
            # libname = item[0]
            entry = item[1]
            # logger.info("Closing library '{}' ...".format(entry.basename))
            # while entry.is_registered() and entry.is_loaded():
            if entry.is_registered() and entry.is_loaded():
                # logger.info("Library '{}' has still {} entities registered.".format(libname, entry.register_count))
                # stdout.write("Library '{}' has still {} entities registered.".format(libname, entry.register_count))
                entry.close()
            entry.unload_library()
            entry.cleanup_files()
        while len(self._data) > 0:
            entry = self._data.popitem()
            # libname = entry[0]
            entry = entry[1]
            # logger.info("Deleteing library '{}'".format(libname))
            del entry

    def add_entry(self, libname, interface_c_instance):
        if not self.is_created(libname):
            self._data[libname] = interface_c_instance

    def load(self, libname, src_dir=get_package_dir()):
        if libname is None or (libname in self._data.keys() and self._data[libname].is_loaded()):
            return
        if libname not in self._data.keys():
            # cppargs = ['-DDOUBLE_COORD_VARIABLES'] if self.lonlatdepth_dtype == np.float64 else None
            return
        if not self._data[libname].is_compiled():
            self._data[libname].compile_library()
        if not self._data[libname].is_loaded():
            self._data[libname].load_library()

    def unload(self, libname):
        if libname in self._data.keys():
            self._data[libname].unload_library()

    def remove(self, libname):
        if libname in self._data.keys():
            del self._data[libname]

    def is_created(self, libname):
        return libname in self._data.keys()

    def is_registered(self, libname):
        return self._data[libname].is_registered()

    def is_loaded(self, libname):
        return self._data[libname].is_loaded()

    def is_compiled(self, libname):
        return self._data[libname].is_compiled()

    def __getitem__(self, item):
        return self.get(item)

    def get(self, libname):
        if libname in self._data.keys():
            return self._data[libname]
        return None

    def register(self, libname, close_callback=None):
        # logger.info("Library registration called.")
        if libname in self._data.keys():
            self._data[libname].register(close_callback)
        #     logger.info("Library '{}' registered {} times.".format(libname, self._data[libname].register_count))

    def deregister(self, libname):
        # logger.info("Library deregistration called.")
        if libname in self._data.keys():
            self._data[libname].unregister()
        #     logger.info("Library '{}' deregistered - remaining registrations: {}.".format(libname, self._data[libname].register_count))


class InterfaceC(object):
    basename = ""
    compiled = False
    loaded = False
    compiler = None
    libc = None
    register_count = 0
    close_cb = None

    def __init__(self, c_file_name, compiler, src_dir=get_package_dir()):
        self.basename = c_file_name
        src_pathfile = c_file_name
        if isinstance(self.basename, list) and not isinstance(self.basename, str) and len(self.basename) > 0:
            self.basename = self.basename[0]
        lib_path = self.basename
        lib_pathfile = os.path.basename(self.basename)
        lib_pathdir = os.path.dirname(self.basename)
        libext = 'dll' if sys.platform == 'win32' else 'so'
        # == handle case that compiler auto-prefixed 'lib' with the libfile == #
        if lib_pathfile[0:3] != "lib":
            lib_pathfile = "lib"+lib_pathfile
            lib_path = os.path.join(lib_pathdir, lib_pathfile)
        # == handle case where multiple simultaneous instances of node-library are required == #
        libinstance = 0
        while os.path.exists("%s-%d.%s" % (os.path.join(get_cache_dir(), lib_path), libinstance, libext)):
            libinstance += 1
        lib_pathfile = "%s-%d" % (lib_pathfile, libinstance)
        lib_path = os.path.join(lib_pathdir, lib_pathfile)
        # == handle multi-lib in an MPI setup == #
        if MPI and MPI.COMM_WORLD.Get_size() > 1:
            lib_pathfile = "%s_%d" % (lib_pathfile, MPI.COMM_WORLD.Get_rank())
            lib_path = os.path.join(lib_pathdir, lib_pathfile)
        if isinstance(src_pathfile, list):
            self.src_file = []
            if isinstance(src_dir, list) and not isinstance(src_dir, str):
                for fdir, fname in zip(src_dir, src_pathfile):
                    self.src_file.append("%s.c" % os.path.join(fdir, fname))
            else:
                for fname in src_pathfile:
                    self.src_file = "%s.c" % os.path.join(src_dir, fname)
        else:
            self.src_file = "%s.c" % os.path.join(src_dir, src_pathfile)
        self.lib_file = "%s.%s" % (os.path.join(get_cache_dir(), lib_path), libext)
        self.log_file = "%s.log" % os.path.join(get_cache_dir(), self.basename)
        if os.path.exists(self.lib_file):
            # logger.info("library path of '{}': library already compiled.".format(basename))
            self.compiled = True
        else:
            # logger.info("library path of '{}': library still needs to be compiled.".format(basename))
            pass

        self.compiler = compiler
        self.compiled = False
        self.loaded = False
        self.libc = None
        self.register_count = 0
        self.close_cb = []

    def __del__(self):
        self.unload_library()
        self.cleanup_files()

    def is_compiled(self):
        return self.compiled

    def is_loaded(self):
        return self.loaded

    def is_registered(self):
        return self.register_count > 0

    def get_library_path(self):
        return self.lib_file

    def get_library_dir(self):
        return os.path.dirname(self.lib_file)

    def get_library_basename(self):
        return self.basename

    def get_library_filename(self):
        return os.path.basename(self.lib_file)

    def get_library_extension(self):
        return os.path.splitext(self.lib_file)[1]

    def compile_library(self):
        """ Writes kernel code to file and compiles it."""
        if not self.compiled:
            self.compiler.compile(self.src_file, self.lib_file, self.log_file)
            # logger.info("Compiled %s ==> %s" % (self.name, self.lib_file))
            # self._cleanup_files = finalize(self, package_globals.cleanup_remove_files, self.lib_file, self.log_file)
            self.compiled = True

    def cleanup_files(self):
        if os.path.isfile(self.lib_file) and self.compiled:
            [os.remove(s) for s in [self.lib_file, self.log_file] if os._exists(s)]
        self.compiled = False

    def unload_library(self):
        if self.libc is not None and self.compiled and self.loaded:
            try:
                _ctypes.FreeLibrary(self.libc._handle) if sys.platform == 'win32' else _ctypes.dlclose(self.libc._handle)
            except (OSError, ) as e:
                logger.error("{}".format(e))
            del self.libc
            self.libc = None
            self.loaded = False

    def load_library(self):
        if self.libc is None and self.compiled and not self.loaded:
            libdir = os.path.dirname(self.lib_file)
            libfile = os.path.basename(self.lib_file)
            liblist = libfile.split('.')
            del liblist[-1]
            libfile = ""
            for entry in liblist:
                libfile += entry
            # self.libc = npct.load_library(self.lib_file, '.')
            self.libc = npct.load_library(libfile, libdir)
            # self.libc = _ctypes.LoadLibrary(self.lib_file) if sys.platform == 'win32' else _ctypes.dlopen(self.lib_file)
            # self._cleanup_lib = finalize(self, package_globals.cleanup_unload_lib, self.libc)
            self.loaded = True

    def register(self, close_callback=None):
        if self.libc is not None and self.compiled and self.loaded:
            self.register_count += 1
        if close_callback is not None:
            self.close_cb.append(close_callback)
        # logger.info("library '{}' - registered instances: {}".format(self.basename, self.register_count))

    def unregister(self):
        if self.register_count > 0:
            self.register_count -= 1
        # logger.info("library '{}' - registered instances: {}".format(self.basename, self.register_count))

    def load_functions(self, function_param_array=None):
        """

        :param function_name_array: array of dictionary {"name": str, "return": type, "arguments": [type, ...]}
        :return: dict (function_name -> function_handler)
        """
        if function_param_array is None:
            function_param_array = []
        result = None
        if self.libc is None or not self.compiled or not self.loaded:
            return result
        result = dict()
        for function_param in function_param_array:
            if isinstance(function_param, dict) and \
                    isinstance(function_param["name"], str) and \
                    isinstance(function_param["return"], type) or function_param["return"] is None and \
                    isinstance(function_param["arguments"], list):
                try:
                    result[function_param["name"]] = self.libc[function_param["name"]]
                    result[function_param["name"]].restype = function_param["return"]
                    result[function_param["name"]].argtypes = function_param["arguments"]
                except (AttributeError, ValueError, KeyError, IndexError) as e:
                    result = None
                    logger.error("Failed to load function '{}' from library '{}.".format(function_param["name"], self.basename))
                    e.print_stack()
        return result

    def close(self):
        if self.close_cb is not None and len(self.close_cb) > 0:
            for close_func in self.close_cb:
                try:
                    close_func()
                except:
                    pass
        if self.register_count <= 0:
            logger.error("Closing interface for '{}' library failed - {} non-revertable links.".format(self.basename, self.register_count))

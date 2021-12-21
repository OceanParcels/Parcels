import os  # noqa
import sys  # noqa
import platform # noqa
import _ctypes
import random as PyRandom
import numpy.ctypeslib as npct
from parcels.tools import get_cache_dir, get_package_dir
from .codecompiler import *  # noqa: F401
from .codecompiler import CCompiler_MS
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
    """
    This class is an interface that allows objects to register to one and the same c-library in the background via ctypes,
    while the library is only compiled one. That is important when having a large number of objects that need to register to
    one and the same library.
    """
    _data = {}

    def __init__(self):
        """
        LibraryRegisterC - Constructor
        """
        self._data = {}

    def __del__(self):
        """
        LibraryRegisterC - Destructor
        """
        if self._data is None or len(self._data) <= 0:
            self._data = None
        else:
            self.clear()
            self._data = None

    def clear(self):
        """
        This functions clears the registry by detaching and closing the connection of registered objects to the managed
        C-libraries, and then remove the compiled C-libraries themselves.
        """
        for item in self._data.items():
            entry = item[1]
            if entry.isregistered() and entry.isloaded():
                entry.close()
            entry.unload_library()
            entry.cleanup_files()
        while len(self._data) > 0:
            entry = self._data.popitem()
            entry = entry[1]
            del entry

    def add_entry(self, libname, interface_c_instance):
        """
        Adds an interface to a distinct C-library, represented as one interface.
        :arg libname: name of the C-library itself; for a library called 'libexample.so' or 'example.dll', :arg name equals 'example'
        :arg interface_c_instance: the C-library interface itself
        """
        if not self.iscreated(libname):
            self._data[libname] = interface_c_instance

    def load(self, libname):  # , src_dir=get_package_dir()
        """
        Loads a distinct C-library via its C-library interface.
        :arg libname: name of the C-library itself; for a library called 'libexample.so' or 'example.dll', :arg name equals 'example'
        """
        if libname is None or (libname in self._data.keys() and self._data[libname].isloaded()):
            return
        if libname not in self._data.keys():
            return
        if not self._data[libname].iscompiled():
            self._data[libname].compile_library()
        if not self._data[libname].isloaded():
            self._data[libname].load_library()

    def unload(self, libname):
        """
        Unloads a given C-library via its registered C-Interface.
        :arg libname: name of the C-library itself; for a library called 'libexample.so' or 'example.dll', :arg name equals 'example'
        """
        if libname in self._data.keys():
            self._data[libname].unload_library()

    def remove(self, libname):
        """
        This function removes a given C-Interface. Note that this function plainly deleted the library - it does not
        (prior to it) unload the library in an ordered manner.
        :arg libname: name of the C-library itself; for a library called 'libexample.so' or 'example.dll', :arg name equals 'example'
        """
        if libname in self._data.keys():
            del self._data[libname]

    def iscreated(self, libname):
        """
        :arg libname: name of the C-library itself; for a library called 'libexample.so' or 'example.dll', :arg name equals 'example'
        :returns if the given library is created (i.e. attached to the registry) or not
        """
        return libname in self._data.keys()

    def isregistered(self, libname):
        """
        :arg libname: name of the C-library itself; for a library called 'libexample.so' or 'example.dll', :arg name equals 'example'
        :returns if the given library is registered (by any external object) or not
        """
        return self._data[libname].isregistered()

    def isloaded(self, libname):
        """
        :arg libname: name of the C-library itself; for a library called 'libexample.so' or 'example.dll', :arg name equals 'example'
        :returns if the given library is loaded or not
        """
        return self._data[libname].isloaded()

    def iscompiled(self, libname):
        """
        :arg libname: name of the C-library itself; for a library called 'libexample.so' or 'example.dll', :arg name equals 'example'
        :returns if the given library is compiled or not
        """
        return self._data[libname].iscompiled()

    def __getitem__(self, item):
        """
        This function retrieves a distinct C-library interface.
        :arg item: name of the C-library itself; for a library called 'libexample.so' or 'example.dll', :arg name equals 'example'
        :returns requested C-library interface
        """
        return self.get(item)

    def get(self, libname):
        """
        This function retrieves a distinct C-library interface. It is a safe request with a prior containment check.
        :arg libname: name of the C-library itself; for a library called 'libexample.so' or 'example.dll', :arg name equals 'example'
        :returns requested C-library interface
        """
        if libname in self._data.keys():
            return self._data[libname]
        return None

    def register(self, libname, close_callback=None):
        """
        This functions allows for an external object to register itself to a distinct library, including defining a
        callback function that allows it to execute an ordered clean-up.
        :arg libname: name of the C-library itself; for a library called 'libexample.so' or 'example.dll', :arg name equals 'example'
        :arg close_callback: a callback-function of the object, called when the external object is deregistered.
        """
        if libname in self._data.keys():
            self._data[libname].register(close_callback)

    def deregister(self, libname):
        """
        This function allows for an external function to deregister itself from a distinct library. If a callback function
        previously has been registered, it is executed upon deregistration.
        :arg libname: name of the C-library itself; for a library called 'libexample.so' or 'example.dll', :arg name equals 'example'

        """
        if libname in self._data.keys():
            self._data[libname].unregister()


class InterfaceC(object):
    """

    """
    basename = ""
    compiled = False
    loaded = False
    compiler = None
    libc = None
    register_count = 0
    close_cb = None

    def __init__(self, c_file_name, compiler, src_dir=get_package_dir()):
        """
        C-Interface - Constructor
        """
        self.basename = c_file_name
        src_pathfile = c_file_name
        if isinstance(self.basename, list) and not isinstance(self.basename, str) and len(self.basename) > 0:
            self.basename = self.basename[0]
        lib_path = self.basename
        lib_pathfile = os.path.basename(self.basename)
        lib_pathdir = os.path.dirname(self.basename)
        libext = 'dll' if sys.platform == 'win32' else 'so'

        # == handle case where multiple simultaneous instances of node-library are required == #
        libinstance = PyRandom.randint(0, (2**32)-1)
        while os.path.exists("%s-%d.%s" % (os.path.join(get_cache_dir(), lib_path), libinstance, libext)):
            libinstance = PyRandom.randint(0, (2**32)-1)
        lib_pathfile = "%s-%d" % (lib_pathfile, libinstance)
        lib_path = lib_pathfile if os.path.sep not in self.basename else os.path.join(lib_pathdir, lib_pathfile)

        # == handle multi-lib in an MPI setup == #
        if MPI and MPI.COMM_WORLD.Get_size() > 1:
            lib_pathfile = "%s_%d" % (lib_pathfile, MPI.COMM_WORLD.Get_rank())
            lib_path = lib_pathfile if os.path.sep not in self.basename else os.path.join(lib_pathdir, lib_pathfile)
        if isinstance(src_pathfile, list) and not isinstance(src_pathfile, str):
            self.src_file = []
            if isinstance(src_dir, list) and not isinstance(src_dir, str):
                for fdir, fname in zip(src_dir, src_pathfile):
                    self.src_file.append("%s.c" % (os.path.join(fdir, fname)))
            else:
                for fname in src_pathfile:
                    self.src_file = "%s.c" % os.path.join(src_dir, fname)
        else:
            self.src_file = "%s.c" % (os.path.join(src_dir, src_pathfile), )
        self.lib_file = "%s.%s" % (os.path.join(get_cache_dir(), lib_path), libext)
        self.log_file = "%s.log" % (os.path.join(get_cache_dir(), lib_path), )
        if os.path.exists(self.lib_file):
            self.compiled = True

        self.compiler = compiler
        self.compiled = False
        self.loaded = False
        self.libc = None
        self.register_count = 0
        self.close_cb = []

    def __del__(self):
        """
        C-Interface - Destructor
        """
        self.unload_library()
        self.cleanup_files()

    def iscompiled(self):
        """
        :returns if this C-library is compiled or not
        """
        return self.compiled

    def isloaded(self):
        """
        :returns if this C-library is loaded or not
        """
        return self.loaded

    def isregistered(self):
        """
        :returns if any external object has been registered with this library or not
        """
        return self.register_count > 0

    def get_library_path(self):
        """
        :returns the path to the library-file, incl. the library filename itself
        """
        return self.lib_file

    def get_library_dir(self):
        """
        :returns the path to the directory where the library-file is located
        """
        return os.path.dirname(self.lib_file)

    def get_library_basename(self):
        """
        :returns the name of the library filename, without the parent directory or the file extension
        """
        return self.basename

    def get_library_filename(self):
        """
        :returns the library filename, without the parent directory or the file extension; commonly similar
                 to 'get_library_basename()', but can include the 'lib'-suffix on UNIX platforms
        """
        return os.path.basename(self.lib_file)

    def get_library_extension(self):
        """
        :returns the file extension of the library
        """
        return os.path.splitext(self.lib_file)[1]

    def compile_library(self):
        """ Writes kernel code to file and compiles it."""
        if not self.compiled:
            src_file = self.src_file if not isinstance(self.compiler, CCompiler_MS) or (isinstance(self.src_file, list) and not isinstance(self.src_file, str)) else [self.src_file, ]
            compiled_fpath = self.compiler.compile(src=src_file, obj=self.lib_file, log=self.log_file)
            logger.info("Compiled %s ==> %s" % (self.basename, compiled_fpath))
            assert os.path.exists(compiled_fpath)
            self.lib_file = compiled_fpath
            self.compiled = True

    def cleanup_files(self):
        """
        Removes all files (i.e. lib-file and log-file) of the associated library
        """
        if os.path.isfile(self.lib_file) and self.compiled:
            [os.remove(s) for s in [self.lib_file, self.log_file] if os.path.exists(s)]
        self.compiled = False

    def unload_library(self):
        """
        Unloads the C-library
        """
        if self.libc is not None and self.compiled and self.loaded:
            try:
                _ctypes.FreeLibrary(self.libc._handle) if sys.platform == 'win32' else _ctypes.dlclose(self.libc._handle)
            except (OSError, ) as e:
                logger.error("{}".format(e))
            del self.libc
            self.libc = None
            self.loaded = False

    def load_library(self):
        """
        Loads the (compiled) C-library
        """
        if self.libc is None and self.compiled and not self.loaded:
            libdir = os.path.dirname(self.lib_file)
            libfile = os.path.basename(self.lib_file)
            libfile = libfile[3:len(libfile)] if libfile[0:3] == "lib" and sys.platform in ['darwin', 'win32'] else libfile
            liblist = libfile.split('.')
            del liblist[-1]
            try:
                self.libc = npct.load_library(self.lib_file, '.')
                logger.info("Loaded %s library (%s)" % (self.basename, self.lib_file))
            except (OSError, ) as e:
                self.loaded = False
                from glob import glob
                libext = 'dll' if sys.platform == 'win32' else 'so'
                alllibfiles = sorted(glob(os.path.join(libdir, "*.%s" % (libext, ))))
                logger.error("Did not locate {} in folder {} among files: {}".format(libfile, libdir, alllibfiles))
                raise e

            self.loaded = True if self.libc is not None else False

    def register(self, close_callback=None):
        """
        Registers an external object to this C-library, with a callback executed on deregistration.
        :arg callback: function of the external object, executed upon deregistration
        """
        if self.libc is not None and self.compiled and self.loaded:
            self.register_count += 1
        if close_callback is not None:
            self.close_cb.append(close_callback)

    def unregister(self):
        """
        Deregisters an external object from this C-library.
        """
        if self.register_count > 0:
            self.register_count -= 1

    def load_functions(self, function_param_array=None):
        """
        A function to return a ctype-interface to the requested library functions, which can be called fromout Python
        :param function_name_array: array of dictionary {"name": str, "return": type, "arguments": [type, ...]}
        :return: dict (function_name -> function_handler)
        """
        if function_param_array is None:
            function_param_array = []
        result = None
        if self.libc is None or not self.compiled or not self.loaded:
            logger.error("Trying to call functions from dead library '{}' - returning empty result.".format(self.basename))
            return result
        result = dict()
        for function_param in function_param_array:
            if isinstance(function_param, dict) and \
                    isinstance(function_param["name"], str) and \
                    (isinstance(function_param["return"], type) or function_param["return"] is None) and \
                    (isinstance(function_param["arguments"], list) or function_param["arguments"] is None):
                try:
                    result[function_param["name"]] = getattr(self.libc, function_param["name"])
                    result[function_param["name"]].restype = function_param["return"]
                    if function_param["arguments"] is not None:
                        result[function_param["name"]].argtypes = function_param["arguments"]
                except (AttributeError, ValueError, KeyError, IndexError) as e:
                    result = None
                    logger.error("Failed to load function '{}' from library '{}.".format(function_param["name"], self.basename))
                    e.print_stack()
        return result

    def close(self):
        """
        Closes this interface, while also executing remaining closing callbacks.
        """
        if self.close_cb is not None and len(self.close_cb) > 0:
            for close_func in self.close_cb:
                try:
                    close_func()
                except:
                    pass
        if self.register_count <= 0:
            logger.error("Closing interface for '{}' library failed - {} non-revertable links.".format(self.basename, self.register_count))

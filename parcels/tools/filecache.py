"""
issue # 1126
"""
# import sys
import os
from copy import deepcopy
import fnmatch
import errno
import numpy as np
import math
import portalocker
import threading
import _pickle as cPickle
from time import sleep
from random import uniform
from shutil import copyfile, copy, copy2, rmtree  # noqa
from .global_statics import get_cache_dir
from tempfile import gettempdir
from .loggers import logger

DEBUG = True


def file_check_lock_busy(filepath):
    """

    :param filepath: file to be checked
    :return: True if locked or busy; False if free
    """
    result = False
    try:
        fp = open(filepath)
        fp.close()
    except IOError as e:
        if e.errno in [errno.EACCES, errno.EBUSY]:
            result = True
    return result


def file_check_OK(filepath):
    """

    :param filepath: file to be checked
    :return: if there is no OS or IO error, all is fine (True)
    """
    result = True
    try:
        fp = open(filepath)
        fp.close()
    except (IOError, OSError):
        result = False
    return result


def get_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


def lock_open_file_sync(filepath, filemode):
    """

    :param filepath: path to file to be lock-opened
    :return: file handle to file
    """
    fh = None
    fh_locked = False
    while not fh_locked:
        try:
            fh = open(filepath, filemode)
            portalocker.lock(fh, portalocker.LOCK_EX)
            fh_locked = True
        except (portalocker.LockException, portalocker.AlreadyLocked, IOError):  # as e
            fh.close()
            sleeptime = uniform(0.1, 0.3)
            sleep(sleeptime)
            fh_locked = False
    return fh


def unlock_close_file_sync(filehandle):
    """

    :param filehandle:
    :return:
    """
    filehandle.flush()
    portalocker.unlock(filehandle)
    filehandle.close()
    return


def get_compute_env():
    """

    :return: tuple of: computer_env, cache_head_dir, data_head_dir
    """
    computer_env = "local/unspecified"
    cache_head = ""
    data_head = ""
    USERNAME = os.environ.get('USER')
    if os.name != 'posix':
        data_head = os.path.join(gettempdir(), "data")
        cache_head = os.path.join(gettempdir(), "{}".format(USERNAME))
        computer_env = "local/non-posix"
    # elif os.uname()[1] in ['science-bs35', 'science-bs36', 'science-bs37', 'science-bs38', 'science-bs39', 'science-bs40', 'science-bs41', 'science-bs42']:  # Gemini
    elif fnmatch.fnmatchcase(os.uname()[1], "science-bs*"):  # Gemini
        data_head = "/data/oceanparcels/input_data"
        cache_head = "/scratch/{}".format(USERNAME)
        computer_env = "Gemini"
    elif os.uname()[1] in ["lorenz.science.uu.nl", ] or fnmatch.fnmatchcase(os.uname()[1], "node*"):  # Lorenz
        data_head = "/storage/shared/oceanparcels/input_data"
        cache_head = "/scratch/{}".format(USERNAME)
        computer_env = "Lorenz"
    elif fnmatch.fnmatchcase(os.uname()[1], "*.bullx*"):  # Cartesius
        data_head = "/projects/0/topios/hydrodynamic_data"
        cache_head = "/scratch/local/{}".format(USERNAME)
        computer_env = "Cartesius"
    elif fnmatch.fnmatchcase(os.uname()[1], "int*.snellius.*") or fnmatch.fnmatchcase(os.uname()[1], "fcn*") or fnmatch.fnmatchcase(os.uname()[1], "tcn*") or fnmatch.fnmatchcase(os.uname()[1], "gcn*") or fnmatch.fnmatchcase(os.uname()[1], "hcn*"):  # Snellius
        data_head = "/projects/0/topios/hydrodynamic_data"
        cache_head = "/scratch-local/{}".format(USERNAME)
        computer_env = "Snellius"
    else:  # local setup
        data_head = "/data"
        cache_head = "/tmp/{}".format(USERNAME)
        computer_env = "local/{}".format(os.name)
    cache_head = os.path.join(cache_head, os.path.split(get_cache_dir())[1])
    # print("running on {} (uname: {}) - cache head directory: {} - data head directory: {}".format(computer_env, os.uname()[1], cache_head, data_head))
    return computer_env, cache_head, data_head


class FieldFileCache(object):
    # needs to be a background process because the FieldFileBuffers will just straight access the cached location;
    # the background process needs to assure that the files are actually there
    _cache_top_dir = "/data"  # needs to be calculated
    _computer_env = "local"  # needs to be calculated
    _occupation_file = "available_files.pkl"  # pickled file
    _process_file = "loaded_files.pkl"
    _ti_file = "timeindices.pkl"
    _field_names = []  # name of all fields under management (for easy iteration)
    _original_top_dirs = {}  # dict per field to their related top directory
    _original_filepaths = {}  # complete file paths to original storage locations
    _global_files = {}  # all files that need to be loaded (at some point)
    _available_files = {}  # files currently loaded into cache
    _prev_processed_files = {}  # mapping of files (via index) and if they've already been loaded or not in previous update
    _processed_files = {}  # mapping of files (via index) and if they've already been loaded or not
    _tis = {}  # currently treated timestep per field - differs between MPI PU's
    _last_loaded_tis = {}  # highest loaded timestep
    _changeflags = {}  # indicates where new data have been requested - differs between MPI PU's
    _cache_upper_limit = 20*1024*1024*1024
    _cache_lower_limit = int(3.5*2014*1024*1024)
    _use_thread = False
    _caching_started = False
    _remove_cache_top_dir = False
    _named_copy = False
    _start_ti = {}
    _end_ti = {}
    _periodic_wrap = {}  # for each field, shows if not to wrap (0), wrap on first-to-last (-1) or on last-to-first (+1)
    _do_wrapping = {}
    _occupation_files_lock = None
    _processed_files_lock = None
    _ti_files_lock = None
    _periodic_wrap_lock = None
    _stopped = None

    def __init__(self, cache_upper_limit=20*1024*1024*1024, cache_lower_limit=3.5*2014*1024*1024, use_thread=False, cache_top_dir=None, remove_cache_dir=True, debug=False):
        computer_env, cache_head, data_head = get_compute_env()
        global DEBUG
        DEBUG = debug
        self._cache_top_dir = cache_top_dir if cache_top_dir is not None and type(cache_top_dir) is str else cache_head
        self._cache_top_dir = os.path.join(self._cache_top_dir, str(os.getpid()))
        if not os.path.exists(self.cache_top_dir):
            os.makedirs(self.cache_top_dir, exist_ok=True)
        self._computer_env = computer_env
        self._occupation_file = "available_files.pkl"
        self._process_file = "loaded_files.pkl"
        self._ti_file = "timeindices.pkl"
        self._field_names = []
        self._var_names = {}
        self._original_top_dirs = {}
        self._original_filepaths = {}
        self._global_files = {}
        self._available_files = {}
        self._processed_files = {}
        self._prev_processed_files = {}
        self._tis = {}
        self._last_loaded_tis = {}
        self._changeflags = {}
        self._cache_upper_limit = int(cache_upper_limit)
        self._cache_lower_limit = int(cache_lower_limit)
        self._use_thread = use_thread
        self._caching_started = False
        self._remove_cache_top_dir = remove_cache_dir
        self._named_copy = False
        self._sim_dt = 1.0
        self._start_ti = {}
        self._end_ti = {}
        self._periodic_wrap = {}
        self._do_wrapping = {}
        self._T = None
        self._occupation_files_lock = None
        self._processed_files_lock = None
        self._ti_files_lock = None
        self._periodic_wrap_lock = None
        self._stopped = None

    def __del__(self):
        if self._caching_started:
            self.stop_caching()

    @property
    def prev_processed_files(self):
        return self._prev_processed_files

    @property
    def cache_top_dir(self):
        return self._cache_top_dir

    @cache_top_dir.setter
    def cache_top_dir(self, cache_dir_path):
        self._cache_top_dir = cache_dir_path

    @property
    def named_copy(self):
        return self._named_copy

    def enable_named_copy(self):
        self._named_copy = True if not self._caching_started else self._named_copy

    def disable_named_copy(self):
        self._named_copy = False if not self._caching_started else self._named_copy

    @property
    def use_thread(self):
        return self._use_thread

    @use_thread.setter
    def use_thread(self, flag):
        self._use_thread = flag

    def enable_threading(self):
        self._use_thread = True

    def disable_threading(self):
        self._use_thread = False

    @property
    def caching_started(self):
        return self._caching_started

    @caching_started.setter
    def caching_started(self, flag):
        raise AttributeError("Flag for caching being started cannot be set from outside the class")

    def _initialize_comm_files_(self):
        process_tis = None
        create_ti_dict = not os.path.exists(os.path.join(self._cache_top_dir, self._ti_file))
        if create_ti_dict:
            process_tis = {}
        else:
            fh_time_indices = lock_open_file_sync(os.path.join(self._cache_top_dir, self._ti_file), filemode="rb")
            process_tis = cPickle.load(fh_time_indices)
            unlock_close_file_sync(fh_time_indices)
        process_tis[os.getpid()] = self._tis

        if DEBUG:
            logger.info("Dumping 'processed' dict into {} ...".format(os.path.join(self._cache_top_dir, self._process_file)))
        fh_processed = lock_open_file_sync(os.path.join(self._cache_top_dir, self._process_file), filemode="wb")
        if DEBUG:
            logger.info("Dumping 'available' dict into {} ...".format(os.path.join(self._cache_top_dir, self._occupation_file)))
        fh_available = lock_open_file_sync(os.path.join(self._cache_top_dir, self._occupation_file), filemode="wb")
        if DEBUG:
            logger.info("Dumping 'time_indices' dict into {} ...".format(os.path.join(self._cache_top_dir, self._ti_file)))
        fh_time_indices = lock_open_file_sync(os.path.join(self._cache_top_dir, self._ti_file), filemode="wb")
        cPickle.dump(self._available_files, fh_available)
        cPickle.dump(self._processed_files, fh_processed)
        cPickle.dump(process_tis, fh_time_indices)
        unlock_close_file_sync(fh_available)
        unlock_close_file_sync(fh_processed)
        unlock_close_file_sync(fh_time_indices)

    def start_caching(self, signdt):
        if DEBUG:
            logger.info("Start caching ...")
        for name in self._field_names:
            self._start_ti[name] = len(self._global_files[name])-1 if signdt < 0 else 0
            self._end_ti[name] = 0 if signdt < 0 else len(self._global_files[name]) - 1
            self._tis[name] = self._start_ti[name] - int(signdt)
            self._last_loaded_tis[name] = self._tis[name]
        self._sim_dt = signdt

        self._initialize_comm_files_()

        if self._use_thread:
            self._start_thread()
            self._T.caching_started = True
        self._caching_started = True

        # if self._use_thread:
        #     self._processed_files_lock.acquire()
        # fh = lock_open_file_sync(os.path.join(self._cache_top_dir, self._process_file), filemode="wb")
        # fh_tis = lock_open_file_sync(os.path.join(self._cache_top_dir, self._ti_file), filemode="wb")
        # for name in self._field_names:
        #     self._tis[name] = self._start_ti[name]
        #     self._processed_files[name][self._tis[name]] += 1
        #     self._changeflags[name] = True
        # cPickle.dump(self._processed_files, fh)
        # cPickle.dump(process_tis, fh_tis)
        # unlock_close_file_sync(fh)
        # unlock_close_file_sync(fh_tis)
        # if self._use_thread:
        #     self._processed_files_lock.release()

    def stop_caching(self):
        if not self._caching_started:
            return
        if self._use_thread:
            if self._occupation_files_lock.locked():
                self._occupation_files_lock.release()
            if self._processed_files_lock.locked():
                self._processed_files_lock.release()
            if self._ti_files_lock.locked():
                self._ti_files_lock.release()
            if self._periodic_wrap_lock.locked():
                self._periodic_wrap_lock.release()
            # self._stopped.set()
            self._T.stop_execution()
            self._T.join()
        self._caching_started = False
        if self._remove_cache_top_dir and os.path.exists(self.cache_top_dir):
            logger.warn("Removing cache folder '{}' ...".format(self._cache_top_dir))
            rmtree(self.cache_top_dir, ignore_errors=False)

    def _start_thread(self):
        if DEBUG:
            logger.info("Creating caching thread ...")
        self._stopped = threading.Event()
        self._occupation_files_lock = threading.Lock()
        self._processed_files_lock = threading.Lock()
        self._ti_files_lock = threading.Lock()
        self._periodic_wrap_lock = threading.Lock()
        self._T = FieldFileCacheThread(self._cache_top_dir, self._computer_env, self._occupation_file, self._process_file, self._ti_file,
                                       self._field_names, self._var_names, self._original_top_dirs, self._original_filepaths, self._global_files,
                                       self._available_files, self._processed_files, self._tis, self._last_loaded_tis, self._changeflags,
                                       self._cache_upper_limit, self._cache_lower_limit, self._named_copy, self._sim_dt, self._start_ti,
                                       self._end_ti, self._do_wrapping, self._periodic_wrap, self._occupation_files_lock,
                                       self._processed_files_lock, self._ti_files_lock, self._periodic_wrap_lock, self._stopped)
        if DEBUG:
            logger.info("Caching thread created.")
        self._T.start()
        if DEBUG:
            logger.info("Caching thread started.")
        sleep(0.5)

    def nc_copy(self, src_filepath, dst_filepath):
        """
        Copies and merges source file fields in destination file
        :param src_filepath:
        :param dst_filepath:
        :return: None
        """
        field_names = self.fields_in_file(dst_filepath)
        var_string = ""
        for fname in field_names:
            var_string += "{},".format(self._var_names[fname])
        var_string = var_string[:-1] if var_string[-1] == ',' else var_string
        # Options:
        # -q -> quench; no print-outs at all
        # -4 -> output in NetCDF 4 format
        # --fix_rec_dmn=all -> remove the 'unlimited' dimension from the data to focus just on 1 file
        cmd = "ncks -4 --fix_rec_dmn=all -q -v {} {} {}".format(var_string, src_filepath, dst_filepath)
        if DEBUG:
            logger.info("copy file via command: '{}'".format(cmd))
        if os.system(cmd) != 0:
            # raise OSError("Failure executing NetCDF kitchen sink '{}'.".format(cmd))
            if os.path.exists(dst_filepath):
                os.remove(dst_filepath)

    def update_processed_files(self):
        """
        :return: None
        """
        for key in self._processed_files.keys():
            assert key in self._prev_processed_files
            assert len(self._processed_files) == len(self._prev_processed_files)
            self._prev_processed_files[key] = deepcopy(self._processed_files[key])

    def is_field_added(self, name):
        """

        :param name: Name of the Field to be added
        :return: if Field is already added to the cache
        """
        return (name in self._field_names)

    def fields_in_file(self, filepath, skip_field=None):
        """
        Checks with fields contain the requested file. This is mainly to fuse the field-copy, especially when using
        'ncks;.
        :param filepath: path to a Field file
        :param skip_field: (Optional) excludes the given field name in the result list
        :return: list of field names
        """
        field_name_results = []
        for name in self._field_names:
            if skip_field is not None and name == skip_field:
                continue
            if filepath in self._global_files[name]:
                field_name_results.append(name)
        return field_name_results

    def add_field(self, name, varname, files, do_wrapping=False):
        """
        Adds files of a field to the cache and returns the (cached) file paths
        :param name: Name of the Field to be added
        :param files: Field data files
        :return: list of new file paths with directory to cache
        """
        field_name = name
        if self.is_field_added(field_name):
            field_name_index = -1
            while self.is_field_added(field_name):
                field_name_index += 1
                field_name = "%s%d" % (name, field_name_index)

        dirname = files[0]
        is_top_dir = False
        while not is_top_dir:
            dirname = os.path.abspath(os.path.join(dirname, os.pardir))
            if DEBUG:
                logger.info("Checking path: {}".format(dirname))
            is_top_dir = np.all([dirname in dname for dname in files])
            if DEBUG:
                logger.info("'{}' is common head: {}".format(dirname, is_top_dir))
        topdirname = dirname
        source_paths = []
        destination_paths = []
        for dname in files:
            source_paths.append(dname)
            fname = os.path.split(dname)[1]
            if self._named_copy:
                ofname = fname
                fname = "{}_{}".format(field_name, ofname)
            destination_paths.append(os.path.join(self._cache_top_dir, fname))
        self._field_names.append(field_name)
        self._var_names[field_name] = varname
        self._original_top_dirs[field_name] = topdirname
        self._original_filepaths[field_name] = source_paths
        self._global_files[field_name] = destination_paths
        self._available_files[field_name] = []
        self._processed_files[field_name] = np.zeros(len(destination_paths), dtype=np.int16)
        self._prev_processed_files[field_name] = np.zeros(len(destination_paths), dtype=np.int16)
        self._periodic_wrap[field_name] = 0
        self._do_wrapping[field_name] = do_wrapping

        self._tis[field_name] = 0
        self._last_loaded_tis[field_name] = 0
        self._changeflags[field_name] = True

        return destination_paths, field_name

    def update_next(self, name, ti):
        # if not self.caching_started:
        #     self.star
        while not self.caching_started:
            logger.warn("FieldFileCacheThread not started")
            sleep(0.1)
        if self._use_thread:
            self._T.request_next(name, ti=ti)
        else:
            self.request_next(name, ti=ti)
        if not self._use_thread:
            self._load_cache()

    def request_next(self, name, ti):
        """
        registers the request for a file in the cache, so that it is loaded (if not already available).
        Also controls the periodic wrapping.
        :param name: name of the registered field
        :param ti: requested time index (i.e. index of a timestamp) if the field file
        :return: None
        """
        # ==== WHAT HAPPENS IF THE SAME TI IS REQUESTED TWICE ? ==== #
        if DEBUG:
            logger.info("{}: requested timestep {} for field '{}'.".format(str(type(self).__name__), ti, name))
        if self._use_thread:
            self._processed_files_lock.acquire()
            self._ti_files_lock.acquire()
        fh = lock_open_file_sync(os.path.join(self._cache_top_dir, self._process_file), filemode="rb")
        fh_tis = lock_open_file_sync(os.path.join(self._cache_top_dir, self._ti_file), filemode="rb")
        self._processed_files = cPickle.load(fh)
        process_tis = cPickle.load(fh_tis)
        if DEBUG:
            logger.info("{}: current timestep {} for field '{}'.".format(str(type(self).__name__), self._tis[name], name))

        if self._use_thread and np.any(list(self._do_wrapping.values())):
            self._periodic_wrap_lock.acquire()
        changed_timestep = False
        if not self._do_wrapping[name]:
            ti = max(ti, self._end_ti[name]) if self._start_ti[name] > 0 else min(ti, self._end_ti[name])
        else:
            ti_len = abs(self._end_ti[name]-self._start_ti[name]) + 1
            ti = (ti + ti_len) % ti_len
        if DEBUG:
            logger.info("{}: [corrected] request-timestep {} for field '{}'.".format(str(type(self).__name__), ti, name))
        assert (ti >= 0) and (ti < len(self._global_files[name])), "Requested index is outside the valid index range."

        ti_delta = int(math.copysign(1, ti - self._tis[name])) if int(ti - self._tis[name]) != 0 else 0
        if self._do_wrapping[name]:
            normal_delta = -1 if self._start_ti[name] > 0 else 1
            self._periodic_wrap[name] = 0 if (ti_delta == normal_delta) or (ti_delta == 0) else normal_delta
            ti_delta = normal_delta
            if DEBUG and self._periodic_wrap[name] != 0:
                logger.info("{}: detected a periodic wrap at ti {} -> {} for field '{}'.".format(str(type(self).__name__), self._tis[name], ti, name))

        while self._tis[name] != ti:
            self._tis[name] += ti_delta
            if self._do_wrapping[name]:
                self._tis[name] = self._start_ti[name] if ti_delta > 0 and (self._tis[name] > self._end_ti[name]) else self._tis[name]
                self._tis[name] = self._start_ti[name] if ti_delta < 0 and (self._tis[name] < self._end_ti[name]) else self._tis[name]
            else:
                self._tis[name] = min(len(self._global_files[name])-1, max(0, self._tis[name]))
            self._processed_files[name][self._tis[name]] += 1
            changed_timestep = True
            if DEBUG:
                logger.info("{}: loading initiated timestep {} (requested {}; timedelta {}) in field '{}'.".format(str(type(self).__name__), self._tis[name], ti, ti_delta, name))
        process_tis[os.getpid()] = self._tis
        if self._use_thread and np.any(list(self._do_wrapping.values())):
            self._periodic_wrap_lock.release()

        unlock_close_file_sync(fh)
        unlock_close_file_sync(fh_tis)
        fh = lock_open_file_sync(os.path.join(self._cache_top_dir, self._process_file), filemode="wb")
        fh_tis = lock_open_file_sync(os.path.join(self._cache_top_dir, self._ti_file), filemode="wb")
        cPickle.dump(self._processed_files, fh)
        cPickle.dump(process_tis, fh_tis)
        unlock_close_file_sync(fh)
        unlock_close_file_sync(fh_tis)
        if self._use_thread:
            self._processed_files_lock.release()
            self._ti_files_lock.release()

        self._changeflags[name] |= changed_timestep

    def request_single(self, name, ti):
        """
        back-up function to obtain field data before simulation (e.g. fieldset sampling; time loading; etc.).
        Hard-copy single file to cache.
        Do not call in actual simulation because of performance drop.
        :param name: name of the registered field
        :param ti: requested time index (i.e. index of a timestamp) if the field file
        :return:
        """
        ti_len = len(self._global_files[name])
        ti = (ti + ti_len) % ti_len
        assert (ti >= 0) and (ti < ti_len), "Requested index is outside the valid index range."
        if DEBUG:
            logger.info("{} (request-single): requested timestep {} for field '{}'.".format(str(type(self).__name__), ti, name))

        if not os.path.exists(os.path.join(self._cache_top_dir, self._occupation_file)):
            self._initialize_comm_files_()

        if self._use_thread and self.caching_started:
            self._occupation_files_lock.acquire()
        fh_available = lock_open_file_sync(os.path.join(self._cache_top_dir, self._occupation_file), filemode="rb")
        self._available_files = cPickle.load(fh_available)
        unlock_close_file_sync(fh_available)
        if self._use_thread and self.caching_started:
            self._occupation_files_lock.release()

        if self._use_thread and self.caching_started:
            self._ti_files_lock.acquire()
        fh_tis = lock_open_file_sync(os.path.join(self._cache_top_dir, self._ti_file), filemode="rb")
        process_tis = cPickle.load(fh_tis)
        unlock_close_file_sync(fh_tis)
        if self._use_thread and self.caching_started:
            self._ti_files_lock.release()

        if DEBUG:
            logger.info("{}: loading requested timestep {} in field '{}'.".format(str(type(self).__name__), ti, name))
        copy2(self._original_filepaths[name][ti], self._global_files[name][ti], follow_symlinks=True)
        while os.path.getsize(self._global_files[name][ti]) != os.path.getsize(self._original_filepaths[name][ti]):
            sleeptime = uniform(0.1, 0.3)
            sleep(sleeptime)

        if DEBUG:
            logger.info("{}: loaded file for timestep {} in field '{}'.".format(str(type(self).__name__), ti, name))
        self._tis[name] = ti
        process_tis[os.getpid()] = self._tis
        if self._global_files[name][ti] not in self._available_files[name]:
            self._available_files[name].append(self._global_files[name][ti])

        if self._use_thread and self.caching_started:
            self._ti_files_lock.acquire()
        fh_tis = lock_open_file_sync(os.path.join(self._cache_top_dir, self._ti_file), filemode="wb")
        cPickle.dump(process_tis, fh_tis)
        unlock_close_file_sync(fh_tis)
        if self._use_thread and self.caching_started:
            self._ti_files_lock.release()

        if self._use_thread and self.caching_started:
            self._occupation_files_lock.acquire()
        fh_available = lock_open_file_sync(os.path.join(self._cache_top_dir, self._occupation_file), filemode="wb")
        cPickle.dump(self._available_files, fh_available)
        unlock_close_file_sync(fh_available)
        if self._use_thread and self.caching_started:
            self._occupation_files_lock.release()
        return True

    def is_ready(self, filepath, name_hint=None):
        """
        Checks if a requested file in :param filepath for the given :param name_hint is available in cache.
        Super-function also forwarding the check to the caching thread.
        :param filepath: requested file path
        :param name_hint: name if the field the requested file belongs to
        :return: boolean if file is available (True) or not (False)
        """
        if name_hint is None:
            for name in self._field_names:
                if filepath in self._global_files[name]:
                    name_hint = name
            assert name_hint is not None, "Requested field not part of the cache requests."
        if self._use_thread:
            return self._T.is_file_available(filepath, name_hint)
        else:
            return self.is_file_available(filepath, name_hint)

    def is_file_available(self, filepath, name):
        """
        Checks if a requested file in :param filepath for the given :param name_hint is available in cache.
        :param filepath: requested file path
        :param name_hint: name if the field the requested file belongs to
        :return: boolean if file is available (True) or not (False)
        """
        if self._use_thread:
            self._occupation_files_lock.acquire()
        fh_available = lock_open_file_sync(os.path.join(self._cache_top_dir, self._occupation_file), filemode="rb")
        self._available_files = cPickle.load(fh_available)
        unlock_close_file_sync(fh_available)
        if self._use_thread:
            self._occupation_files_lock.release()
        file_available_check = filepath in self._available_files[name]
        file_exists_check = os.path.exists(filepath)
        file_ok_check = file_check_OK(filepath)
        if DEBUG:
            logger.info("Available files in cache: {}".format(self._available_files[name]))
            logger.info("File to locate: {}".format(filepath))
            logger.info("File located ?: {}".format(file_available_check))
            logger.info("File exists ?: {}".format(file_exists_check))
            logger.info("File OK ?: {}".format(file_ok_check))
        return file_available_check and file_exists_check and file_ok_check

    def renew_cache(self, name):
        """
        Just initiates a new cache_load process by setting the changeflags of field :param name.
        Super-function also forwarding the request to the caching thread.
        :param name: name of the field to the renewed in cache
        :return: None
        """
        while not self.caching_started:
            logger.warn("FieldFileCacheThread not started")
            sleep(0.1)
        if self._use_thread:
            self._T.reset_changeflag(name=name)
        else:
            self.reset_changeflag(name=name)

    def reset_changeflag(self, name):
        """
        Just initiates a new cache_load process by setting the changeflags of field :param name.
        :param name: name of the field to the renewed in cache
        :return: None
        """
        self._changeflags[name] = True

    def restart_cache(self, name=None):
        """
        Re-initializes the cache after the caching has started if the simulation is completely reset.
        Super-function also forwarding the request to the caching thread.
        :param name: name of the field the cache is to be restarted for
        :return:
        """
        while not self.caching_started:
            logger.warn("FieldFileCacheThread not started")
            sleep(0.1)
        if name is None:
            for fname in self._field_names:
                if self._use_thread:
                    self._T.call_restart_cache(name=fname)
                else:
                    self.call_restart_cache(name=fname)
        else:
            if self._use_thread:
                self._T.call_restart_cache(name=name)
            else:
                self.call_restart_cache(name=name)

    def call_restart_cache(self, name):
        """
        Re-initializes the cache after the caching has started if the simulation is completely reset.
        :param name: name of the field the cache is to be restarted for
        :return:
        """
        if self._use_thread:
            self._ti_files_lock.acquire()
        fh_time_indices = lock_open_file_sync(os.path.join(self._cache_top_dir, self._ti_file), filemode="rb")
        process_tis = cPickle.load(fh_time_indices)
        unlock_close_file_sync(fh_time_indices)
        if self._use_thread:
            self._ti_files_lock.release()
        if DEBUG:
            logger.info("All processes' time indices: {}".format(process_tis))

        for name in self._field_names:
            self._prev_processed_files[name] = deepcopy(self._processed_files[name])
            self._processed_files[name] = np.zeros(len(self._global_files[name]), dtype=np.int16)
            self._periodic_wrap[name] = 0
            self._tis[name] = 0
            self._last_loaded_tis[name] = 0
            self._changeflags[name] = True

        if self._use_thread:
            self._ti_files_lock.acquire()
        fh_tis = lock_open_file_sync(os.path.join(self._cache_top_dir, self._ti_file), filemode="wb")
        cPickle.dump(process_tis, fh_tis)
        unlock_close_file_sync(fh_tis)
        if self._use_thread:
            self._ti_files_lock.release()

    def _load_cache(self):
        """
        Updates the cache. Procedure as follows:
        1. Check if a new data have been requested - if no change occurred: return/skip
        2. Lock-open the file - processed files and available files
        2.1 Collect all cache files not further required (to be deleted) -> processed_files
        2.2 Remove the files from cache disk
        2.3 Remove the filepaths from the available file list
        2.4 Check cache file size; if cache size >= lower limit:
        2.4.1 write new processed files and available files to disk: pass
        2.5 else: In while loop (cache size < upper limit)
        2.5.1 copy files of the next timestep to cache
        2.5.2 increment _last_loaded_tis
        2.5.3 add copied filepaths to available files
        2.5.4 Check cache file size
        2.5.5 If cache size < upper_limit: to to 2.5
        2.5.6 else: break
        2.6 Write new processed files and available files to disk
        2.7 Unlock-close files - processed files and available files
        2.7 reset changeflags

        :return: None
        """
        num_changed_fields = np.sum(list(self._changeflags.values()))
        if num_changed_fields <= 0:
            return
        if DEBUG:
            logger.info("# changed field: {}".format(num_changed_fields))
        if self._use_thread:
            self._processed_files_lock.acquire()
        fh_processed = lock_open_file_sync(os.path.join(self._cache_top_dir, self._process_file), filemode="rb")
        self._processed_files = cPickle.load(fh_processed)
        unlock_close_file_sync(fh_processed)
        if self._use_thread:
            self._processed_files_lock.release()
        if self._use_thread:
            self._ti_files_lock.acquire()
        fh_time_indices = lock_open_file_sync(os.path.join(self._cache_top_dir, self._ti_file), filemode="rb")
        process_tis = cPickle.load(fh_time_indices)
        unlock_close_file_sync(fh_time_indices)
        if self._use_thread:
            self._ti_files_lock.release()
        if DEBUG:
            logger.info("All processes' time indices: {}".format(process_tis))
        if self._use_thread:
            self._occupation_files_lock.acquire()
        fh_available = lock_open_file_sync(os.path.join(self._cache_top_dir, self._occupation_file), filemode="rb")
        self._available_files = cPickle.load(fh_available)
        unlock_close_file_sync(fh_available)
        if self._use_thread:
            self._occupation_files_lock.release()

        cache_range_indices = {}
        signdt = 1 if max(list(self._start_ti.values())) == 0 else -1
        indices = {}
        cacheclean = {}
        files_to_keep = {}
        global_files_to_keep = []
        if self._use_thread and np.any(list(self._do_wrapping.values())):
            self._periodic_wrap_lock.acquire()
        for name in self._field_names:
            if self._periodic_wrap[name] != 0 and self._do_wrapping[name]:
                self._prev_processed_files[name][:] -= 1
                self._processed_files[name][:] -= 1
            if DEBUG:
                logger.info("field '{}':  prev_processed_files = {}".format(name, self.prev_processed_files[name]))
            current_ti = self._tis[name]
            prev_processed = np.where(self.prev_processed_files[name] > 0)[0]
            progress_ti_before = (prev_processed.max() if signdt > 0 else prev_processed.min()) if np.any(self.prev_processed_files[name] > 0) else self._start_ti[name]
            if DEBUG:
                logger.info("field '{}':  _processed_files = {}".format(name, self._processed_files[name]))
            now_processed = np.where(self._processed_files[name] > 0)[0]
            progress_ti_now = (now_processed.max() if signdt > 0 else now_processed.min()) if np.any(self._processed_files[name] > 0) else self._start_ti[name]
            if progress_ti_now != progress_ti_before:
                self._changeflags[name] |= True
            if DEBUG:
                logger.info("field '{}': progress ti before = {}, progress ti now = {}".format(name, progress_ti_before, progress_ti_now))
            last_ti = len(self._global_files[name])-1
            past_keep_index = max(progress_ti_before-1, 0) if signdt > 0 else min(progress_ti_before+1, last_ti)
            if self._do_wrapping[name]:
                future_keep_index = progress_ti_before-2 if signdt > 0 else progress_ti_before+2  # purely storage limited
                future_keep_index = (future_keep_index + len(self._global_files[name])) % len(self._global_files[name])
            else:
                # future_keep_index = min(progress_ti_now+5, last_ti) if signdt > 0 else max(progress_ti_now-5, 0)  # look-ahead index limit
                future_keep_index = self._end_ti[name]  # purely storage limited
            past_keep_index = min(past_keep_index, max(current_ti - 1, 0)) if signdt > 0 else max(past_keep_index, min(current_ti + 1, last_ti))  # clamping to what is currently processed
            future_keep_index = max(future_keep_index, min(current_ti + 1, last_ti)) if signdt > 0 else min(future_keep_index, max(current_ti - 1, 0))
            if DEBUG:
                logger.info("field '{}' (before cleanup): past-ti = {}, future-ti = {}".format(name, past_keep_index, future_keep_index))
            cache_range_indices[name] = (past_keep_index, future_keep_index)
            files_to_keep[name] = list(dict.fromkeys(self._global_files[name][past_keep_index:future_keep_index]))
            global_files_to_keep += files_to_keep[name]
            indices[name] = self._start_ti[name]
            cacheclean[name] = not self._changeflags[name]
            if self._do_wrapping[name]:
                self._periodic_wrap[name] = 0
        if self._use_thread and np.any(list(self._do_wrapping.values())):
            self._periodic_wrap_lock.release()
        global_files_to_keep = list(dict.fromkeys(global_files_to_keep))

        cache_size = get_size(self._cache_top_dir)
        while (cache_size > self._cache_lower_limit) and (not np.all(list(cacheclean.values()))):
            for name in self._field_names:
                if cacheclean[name]:
                    continue
                past_keep_index = cache_range_indices[name][0]
                i = indices[name]
                indices[name] += signdt
                if self._do_wrapping[name]:
                    indices[name] = (indices[name] + len(self._global_files[name])) % len(self._global_files[name])
                else:
                    indices[name] = (min(indices[name], self._end_ti[name]) if signdt > 0 else max(indices[name], self._end_ti[name]))
                # if (signdt > 0 and (i >= past_keep_index or i >= self._tis[name])) or (signdt < 0 and (i <= past_keep_index or i <= self._tis[name])) or self._global_files[name][self._tis[name]] == self._global_files[name][i]:
                if (signdt > 0 and i >= past_keep_index) or (signdt < 0 and i <= past_keep_index) or self._global_files[name][self._tis[name]] == self._global_files[name][i]:
                    indices[name] = i
                    cacheclean[name] = True
                    continue

                if self._use_thread:
                    self._occupation_files_lock.acquire()
                fh_available = lock_open_file_sync(os.path.join(self._cache_top_dir, self._occupation_file), filemode="rb")
                self._available_files = cPickle.load(fh_available)
                unlock_close_file_sync(fh_available)
                if self._use_thread:
                    self._occupation_files_lock.release()
                if (self._global_files[name][i] in self._available_files[name]):
                    # if os.path.exists(self._global_files[name][i]) and not file_check_lock_busy(self._global_files[name][i]) and self._global_files[name][i] not in files_to_keep[name]:
                    if os.path.exists(self._global_files[name][i]) and not file_check_lock_busy(self._global_files[name][i]) and self._global_files[name][i] not in global_files_to_keep:
                        if DEBUG:
                            logger.info("Removing file '{}' with index={} ...".format(self._global_files[name][i], i))
                        os.remove(self._global_files[name][i])
                        self._available_files[name].remove(self._global_files[name][i])
                    else:  # file still in use -> lowest usable index
                        indices[name] = i
                        cacheclean[name] = True
                else:
                    pass  # skip because the file is not in cache
                if self._use_thread:
                    self._occupation_files_lock.acquire()
                fh_available = lock_open_file_sync(os.path.join(self._cache_top_dir, self._occupation_file), filemode="wb")
                cPickle.dump(self._available_files, fh_available)
                unlock_close_file_sync(fh_available)
                if self._use_thread:
                    self._occupation_files_lock.release()
            # ==== do not remove more files than necessary -> not below lower cache limit
            cache_size = get_size(self._cache_top_dir)
            if DEBUG and (np.any(list(cacheclean.values())) or (cache_size < self._cache_lower_limit)):
                logger.info("[removed cache] Current cache size: {} bytes ({} MB); cleaned fields: {}.".format(cache_size, int(cache_size/(1024*1024)), cacheclean))
            if (cache_size < self._cache_lower_limit) or np.all(list(cacheclean.values())):
                break

        for name in self._field_names:
            cache_range_indices[name] = (indices[name], cache_range_indices[name][1])
            if DEBUG:
                logger.info("field '{}' (after cleanup): past-ti = {}, future-ti = {}".format(name, cache_range_indices[name][0], cache_range_indices[name][1]))

        # for name in self._field_names:
        #     past_keep_index = cache_range_indices[name][0]
        #     future_keep_index = cache_range_indices[name][1]
        #     for i in range(self._start_ti[name], past_keep_index, signdt):
        #         fh_available = lock_open_file_sync(os.path.join(self._cache_top_dir, self._occupation_file), filemode="rb")
        #         self._available_files = cPickle.load(fh_available)
        #         unlock_close_file_sync(fh_available)
        #         if (self._global_files[name][i] in self._available_files[name]) and (os.path.exists(self._global_files[name][i])) and not file_check_lock_busy(self._global_files[name][i]):
        #             os.remove(self._global_files[name][i])
        #             self._available_files[name].remove(self._global_files[name][i])
        #         fh_available = lock_open_file_sync(os.path.join(self._cache_top_dir, self._occupation_file), filemode="wb")
        #         cPickle.dump(self._available_files, fh_available)
        #         unlock_close_file_sync(fh_available)
        #     if DEBUG:
        #         logger.info("field '{}' (after cleanup): past-ti = {}, future-ti = {}".format(name, past_keep_index, future_keep_index))
        #     cache_range_indices[name] = (past_keep_index, future_keep_index)

        cache_size = get_size(self._cache_top_dir)
        if DEBUG:
            logger.info("[before adding] Current cache size: {} bytes ({} MB).".format(cache_size, int(cache_size/(1024*1024))))
        cachefill = (cache_size >= self._cache_upper_limit)
        # while (cache_size < self._cache_upper_limit) and (not cachefill):
        while (cache_size < self._cache_upper_limit) and (not cachefill) and np.any(list(self._changeflags.values())):
            cachefill = True
            for name in self._field_names:
                if not self._do_wrapping[name] and ((cache_range_indices[name][0] > cache_range_indices[name][1] and signdt >= 0) or (cache_range_indices[name][0] < cache_range_indices[name][1] and signdt < 0)):
                    continue  # no new file to add to cache
                elif not self._changeflags[name]:
                    continue  # no new file to add to cache
                i = cache_range_indices[name][0]

                if self._use_thread:
                    self._occupation_files_lock.acquire()
                fh_available = lock_open_file_sync(os.path.join(self._cache_top_dir, self._occupation_file), filemode="rb")
                self._available_files = cPickle.load(fh_available)
                unlock_close_file_sync(fh_available)
                if self._use_thread:
                    self._occupation_files_lock.release()

                if not os.path.exists(self._global_files[name][i]):
                    if DEBUG:
                        logger.info("field '{}' - loading '{}' to '{}' ...".format(name, self._original_filepaths[name][i], self._global_files[name][i]))
                    # copyfile(self._original_filepaths[name][i], self._global_files[name][i])
                    copy2(self._original_filepaths[name][i], self._global_files[name][i], follow_symlinks=True)
                    # copy(self._original_filepaths[name][i], self._global_files[name][i], follow_symlinks=True)
                    checksize = False

                    # self.nc_copy(self._original_filepaths[name][i], self._global_files[name][i])
                    # if not os.path.exists(self._global_files[name][i]):
                    #     copy2(self._original_filepaths[name][i], self._global_files[name][i], follow_symlinks=True)
                    #     checksize = True
                    assert os.path.exists(self._global_files[name][i])
                    while checksize and (os.path.getsize(self._global_files[name][i]) != os.path.getsize(self._original_filepaths[name][i])):
                        sleeptime = uniform(0.05, 0.12)
                        sleep(sleeptime)
                    if DEBUG:
                        logger.info("field '{}' - '{}' ready.".format(name, self._global_files[name][i]))
                else:
                    if DEBUG:
                        logger.info("field '{}' - '{}' already available.".format(name, self._global_files[name][i]))
                    pass

                if self._global_files[name][i] not in self._available_files[name]:
                    self._available_files[name].append(self._global_files[name][i])
                if self._use_thread:
                    self._occupation_files_lock.acquire()
                fh_available = lock_open_file_sync(os.path.join(self._cache_top_dir, self._occupation_file), filemode="wb")
                cPickle.dump(self._available_files, fh_available)
                unlock_close_file_sync(fh_available)
                if self._use_thread:
                    self._occupation_files_lock.release()

                self._last_loaded_tis[name] = i
                cachefill &= False
                if cache_range_indices[name][0] == cache_range_indices[name][1]:
                    self._changeflags[name] = False  # no new file to add to cache in next run
                cache_range_indices[name] = (i+signdt, cache_range_indices[name][1])
                if self._do_wrapping[name]:
                    cache_range_indices[name] = (self._start_ti[name], cache_range_indices[name][1]) if (cache_range_indices[name][0] > (len(self._global_files[name]) - 1) and signdt > 0) else cache_range_indices[name]
                    cache_range_indices[name] = (self._start_ti[name], cache_range_indices[name][1]) if (cache_range_indices[name][0] < 0 and signdt < 0) else cache_range_indices[name]
            cache_size = get_size(self._cache_top_dir)
            if DEBUG and (cachefill or (cache_size > self._cache_upper_limit)):
                logger.info("[after adding] Current cache size: {} bytes ({} MB).".format(cache_size, int(cache_size/(1024*1024))))
            if (cache_size >= self._cache_upper_limit) or cachefill or not np.any(list(self._changeflags.values())):
                break

        self.update_processed_files()
        # fh_available = lock_open_file_sync(os.path.join(self._cache_top_dir, self._occupation_file), filemode="wb")
        # cPickle.dump(self._available_files, fh_available)
        # unlock_close_file_sync(fh_available)
        for name in self._field_names:
            self._changeflags[name] = False


class FieldFileCacheThread(threading.Thread, FieldFileCache):

    def __init__(self, cache_top_dir, computer_env, occupation_file, process_file, ti_file,
                 field_names, var_names, original_top_dirs, original_filepaths, global_files,
                 available_files, processed_files, tis, last_loaded_tis, changeflags,
                 cache_upper_limit, cache_lower_limit, named_copy, sim_dt, start_ti, end_ti,
                 do_wrapping, periodic_wrap, occupation_files_lock, processed_files_lock,
                 ti_files_lock, periodic_wrap_lock, stop_event):
        super(FieldFileCache, self).__init__()
        threading.Thread.__init__(self)
        self._cache_top_dir = cache_top_dir
        self._computer_env = computer_env
        self._occupation_file = occupation_file
        self._process_file = process_file
        self._ti_file = ti_file
        self._field_names = field_names
        self._var_names = var_names
        self._original_top_dirs = original_top_dirs
        self._original_filepaths = original_filepaths
        self._global_files = global_files
        self._available_files = available_files
        self._processed_files = processed_files
        self._prev_processed_files = {}
        for name in self._field_names:
            self._prev_processed_files[name] = np.zeros(len(self._global_files[name]), dtype=np.int16)
        if DEBUG:
            logger.info("FieldFileCacheThread: previous processed files - keys: {}".format(self._prev_processed_files.keys()))
            logger.info("FieldFileCacheThread: processed files - keys: {}".format(self._processed_files.keys()))
        self._tis = tis
        self._last_loaded_tis = last_loaded_tis
        self._changeflags = changeflags
        self._cache_upper_limit = cache_upper_limit
        self._cache_lower_limit = cache_lower_limit
        self._use_thread = True
        self._caching_started = False
        self._named_copy = named_copy
        self._sim_dt = sim_dt
        self._start_ti = start_ti
        self._end_ti = end_ti
        self._periodic_wrap = {}
        self._do_wrapping = do_wrapping
        self._periodic_wrap = periodic_wrap
        self._occupation_files_lock = occupation_files_lock
        self._processed_files_lock = processed_files_lock
        self._ti_files_lock = ti_files_lock
        self._periodic_wrap_lock = periodic_wrap_lock
        self._stopped = stop_event
        self._sleepreset = False

    @property
    def use_thread(self):
        return True

    @property
    def prev_processed_files(self):
        return self._prev_processed_files

    @use_thread.setter
    def use_thread(self, flag):
        pass

    def enable_threading(self):
        pass

    def disable_threading(self):
        pass

    @property
    def caching_started(self):
        return self._caching_started

    @caching_started.setter
    def caching_started(self, flag):
        self._caching_started = flag

    def start_caching(self, signdt):
        pass

    def stop_caching(self):
        pass

    def _start_thread(self):
        pass

    def reset_changeflag(self, name):
        super(FieldFileCacheThread, self).reset_changeflag(name=name)
        self._sleepreset = True

    def stop_execution(self):
        self._stopped.set()

    def run(self):
        sleeptime = 0.2  # [s]
        maintain_sleep = True
        while not self._stopped.is_set():
            if self._sleepreset:
                sleeptime = 0.2
                self._sleepreset = False
            if np.any(list(self._changeflags.values())):
                if DEBUG:
                    logger.info("FieldFileCacheThread: loading files to cache ...")
                self._load_cache()
                if DEBUG:
                    num_files_loaded = 0
                    for name in self._field_names:
                        num_files_loaded += len(self._available_files[name])
                    logger.info("FieldFileCacheThread: files loaded into cache. Cache size: {}".format(num_files_loaded))
                if not maintain_sleep:
                    sleeptime /= 2.0
                maintain_sleep = False
            else:
                sleeptime *= 2.0
                maintain_sleep = True
            sleep(sleeptime)  # <- load balancing on this parameter: normal aging first, doubling the time if no change and halfing the time after a change. 200ms min. is used when renew_cache() needed to be called.; like TCP-Reno protocoll.
            if self._stopped.is_set():
                break

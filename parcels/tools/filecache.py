"""

"""
# import sys
import os
# from glob import glob
import numpy as np
import portalocker
import threading
import _pickle as cPickle
from time import sleep
from random import uniform
from shutil import copyfile


def get_size(start_path='.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size


def lock_open_file_sync(filepath):
    """

    :param filepath: path to file to be lock-opened
    :return: file handle to file
    """
    fh = None
    fh_locked = False
    while not fh_locked:
        try:
            fh = open(filepath, "rw")
            portalocker.lock(fh, portalocker.LOCK_EX)
            fh_locked = True
        except (portalocker.LockException, portalocker.AlreadyLocked):  # as e
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


class FieldFileCache(object):
    # needs to be a background process because the FieldFileBuffers will just straight access the cached location;
    # the background process needs to assure that the files are actually there
    _cache_top_dir = "/data"
    _computer_env = "local"
    _occupation_file = "available_files.pkl"  # pickled file
    _process_file = "loaded_files.pkl"
    _field_names = []  # name of all fields under management (for easy iteration)
    _original_top_dirs = {}  # dict per field to their related top directory
    _original_filepaths = {}  # complete file paths to original storage locations
    _global_files = {}  # all files that need to be loaded (at some point)
    _available_files = {}  # files currently loaded into cache
    __prev_processed_files = {}  # mapping of files (via index) and if they've already been loaded or not in previous update
    _processed_files = {}  # mapping of files (via index) and if they've already been loaded or not
    _tis = {}  # currently treated timestep per field - differs between MPI PU's
    _last_loaded_tis = {}  # highest loaded timestep
    _changeflags = {}  # indicates where new data have been requested - differs between MPI PU's
    _cache_upper_limit = 20*1024*1024*1024
    _cache_lower_limit = int(3.5*2014*1024*1024)
    _use_thread = False

    def __init__(self, cache_upper_limit=eval(20*1024*1024*1024), cache_lower_limit=eval(3.5*2014*1024*1024), use_thread=False):
        self._cache_top_dir = "/data"  # needs to be calculated
        self._computer_env = "local"  # needs to be calculated
        self._occupation_file = "available_files.pkl"
        self._process_file = "loaded_files.pkl"
        self._field_names = []
        self._original_top_dirs = {}
        self._original_filepaths = {}
        self._global_files = {}
        self._available_files = {}
        self.__prev_processed_files = {}
        self._processed_files = {}
        self._tis = {}
        self._last_loaded_tis = {}
        self._changeflags = {}
        self._cache_upper_limit = int(cache_upper_limit)
        self._cache_lower_limit = int(cache_lower_limit)
        self._use_thread = use_thread
        self._T = None

    @property
    def cache_top_dir(self):
        return self._cache_top_dir

    @cache_top_dir.setter
    def cache_top_dir(self, cache_dir_path):
        self._cache_top_dir = cache_dir_path

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

    def start_caching(self):
        if self._use_thread:
            self._start_thread()

    def stop_caching(self):
        if self._use_thread:
            self._T.stopped.set()

    def _start_thread(self):
        self._T = FieldFileCacheThread(self._cache_top_dir, self._computer_env, self._occupation_file, self._process_file,
                                       self._field_names, self._original_top_dirs, self._original_filepaths, self._global_files,
                                       self._available_files, self._processed_files, self._tis, self._last_loaded_tis, self._changeflags,
                                       self._cache_upper_limit, self._cache_lower_limit)
        self._T.start()

    def add_field(self, name, files):
        """
        Adds files of a field to the cache and returns the (cached) file paths
        :param name:
        :param files:
        :return: list of new file paths with directory to cache
        """
        dirname = files[0]
        is_top_dir = False
        while not is_top_dir:
            dirname = os.path.join(dirname, os.pardir)
            is_top_dir = np.nonzero([dirname in dname for dname in files])[0]
        topdirname = dirname
        source_paths = []
        destination_paths = []
        for dname in files:
            source_paths.append(dname)
            fname = os.path.split(dname)[1]
            destination_paths.append(os.path.join(self._cache_top_dir, fname))
        self._field_names.append(name)
        self._original_top_dirs[name] = topdirname
        self._original_filepaths[name] = source_paths
        self._global_files[name] = destination_paths
        self._available_files[name] = []
        self._processed_files[name] = [0, ] * len(destination_paths)
        self.__prev_processed_files[name] = [0, ] * len(destination_paths)

        self._tis[name] = 0
        self._last_loaded_tis[name] = 0
        self._changeflags[name] |= True

        return destination_paths

    def update_next(self, name):
        if self._use_thread:
            self._T.request_next(name)
        else:
            self.request_next(name)
        if not self._use_thread:
            self._load_cache()

    def request_next(self, name):
        """

        :param name: name of the registered field
        :return: None
        """
        self._tis[name] += 1

        fh = lock_open_file_sync(self._process_file)
        self._processed_files = cPickle.load(fh)
        self._processed_files[name][self._tis[name]] += 1
        cPickle.dump(self._processed_files, fh)
        unlock_close_file_sync(fh)

        self._changeflags[name] |= True

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
        if np.nonzero(self._changeflags.values())[0] <= 0:
            return
        fh_processed = lock_open_file_sync(self._process_file)
        fh_available = lock_open_file_sync(self._occupation_file)
        self._processed_files = cPickle.load(fh_processed)
        self._available_files = cPickle.load(fh_available)

        cache_range_indices = {}
        for name in self._field_names:
            max_ti_before = np.where(self.__prev_processed_files[name] > 0)[0].max()
            max_ti_now = np.where(self._processed_files[name] > 0)[0].max()
            if max_ti_now > max_ti_before:
                self._changeflags[name] |= True
            lowest_keep_index = max(max_ti_before-1, 0)
            highest_keep_index = max_ti_now(max_ti_now+5, len(self._global_files[name])-1)
            for i in range(0, lowest_keep_index):
                if (self._global_files[name][i] in self._available_files[name]) and (os.path.exists(self._global_files[name][i])):
                    os.remove(self._global_files[name][i])
                    self._available_files[name].remove(self._global_files[name][i])
            start = lowest_keep_index
            for i in range(start, highest_keep_index):
                if (self._global_files[name][i] in self._available_files[name]) and (os.path.exists(self._global_files[name][i])):
                    lowest_keep_index += 1
            cache_range_indices[name] = (lowest_keep_index, highest_keep_index)

        cache_size = sum(os.path.getsize(cachefile) for cachefile in os.listdir(self._cache_top_dir) if os.path.isfile(cachefile))
        if not (cache_size >= self._cache_lower_limit):
            cachefill = False
            while (cache_size < self._cache_upper_limit) and (not cachefill):
                cachefill = True
                for name in self._field_names:
                    if cache_range_indices[name][0] >= cache_range_indices[1]:
                        continue  # no new file to add to cache
                    i = cache_range_indices[name][0]
                    copyfile(self._original_filepaths[name][i], self._global_files[name][i])
                    self._last_loaded_tis[name] = i
                    self._available_files[name].append(self._global_files[name][i])
                    cache_range_indices[name] = (i+1, cache_range_indices[name][1])
                    cachefill &= False
                cache_size = sum(os.path.getsize(cachefile) for cachefile in os.listdir(self._cache_top_dir) if os.path.isfile(cachefile))
                if (cache_size >= self._cache_upper_limit) or cachefill:
                    break

        self.__prev_processed_files = self._processed_files
        cPickle.dump(self._available_files, fh_available)
        cPickle.dump(self._processed_files, fh_processed)
        unlock_close_file_sync(fh_available)
        unlock_close_file_sync(fh_processed)
        for name in self._field_names:
            self._changeflags[name] = False


class FieldFileCacheThread(threading.Thread, FieldFileCache):

    def __init__(self, cache_top_dir, computer_env, occupation_file, process_file,
                 field_names, original_top_dirs, original_filepaths, global_files,
                 available_files, processed_files, tis, last_loaded_tis, changeflags,
                 cache_upper_limit, cache_lower_limit):
        super(threading.Thread, self).__init__()
        super(FieldFileCache, self).__init__()
        self._cache_top_dir = cache_top_dir
        self._computer_env = computer_env
        self._occupation_file = occupation_file
        self._process_file = process_file
        self._field_names = field_names
        self._original_top_dirs = original_top_dirs
        self._original_filepaths = original_filepaths
        self._global_files = global_files
        self._available_files = available_files
        self._processed_files = processed_files
        for name in self._field_names:
            self.__prev_processed_files[name] = [0, ] * len(process_file[name])
        self._tis = tis
        self._last_loaded_tis = last_loaded_tis
        self._changeflags = changeflags
        self._cache_upper_limit = cache_upper_limit
        self._cache_lower_limit = cache_lower_limit
        self._use_thread = True
        self.stopped = threading.Event()

    @property
    def use_thread(self):
        return True

    @use_thread.setter
    def use_thread(self, flag):
        pass

    def enable_threading(self):
        pass

    def disable_threading(self):
        pass

    def start_caching(self):
        pass

    def stop_caching(self):
        pass

    def _start_thread(self):
        pass

    def run(self):
        while not self.stopped.is_set():
            self._load_cache()
            sleep(0.1)
            if self.stopped.is_set():
                break

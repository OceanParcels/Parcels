"""Script to create a `logger` for Parcels"""
import logging

__all__ = ['logger', 'XarrayDecodedFilter']

warning_once_level = 25
info_once_level = 26


class DuplicateFilter(object):
    """Utility class to prevent warning_once warnings from being
    displayed more than once"""
    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        if record.levelno in [warning_once_level, info_once_level]:
            self.msgs.add(record.msg)
        return rv


def warning_once(self, message, *args, **kws):
    """Custom logging level for warnings that need to be displayed only once"""
    if self.isEnabledFor(warning_once_level):
        self._log(warning_once_level, message, args, **kws)


def info_once(self, message, *args, **kws):
    """Custom logging level for info that need to be displayed only once"""
    if self.isEnabledFor(info_once_level):
        self._log(info_once_level, message, args, **kws)


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt="%(levelname)s: %(message)s"))
logger.addHandler(handler)

logging.addLevelName(warning_once_level, "WARNING")
logging.Logger.warning_once = warning_once

logging.addLevelName(info_once_level, "INFO")
logging.Logger.info_once = info_once

dup_filter = DuplicateFilter()
logger.addFilter(dup_filter)
logger.setLevel(10)


class XarrayDecodedFilter(logging.Filter):
    """Filters the warning_once from fieldfilebuffer when cf_decoding fails"""
    def filter(self, record):
        return 'Filling values might be wrongly parsed' not in record.getMessage()

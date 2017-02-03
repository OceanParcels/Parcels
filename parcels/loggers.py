import logging

__all__ = ('logger')


class DuplicateFilter(object):
    def __init__(self):
        self.msgs = set()

    def filter(self, record):
        rv = record.msg not in self.msgs
        self.msgs.add(record.msg)
        return rv

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(fmt="%(levelname)s: %(message)s"))
logger.addHandler(handler)

dup_filter = DuplicateFilter()
logger.addFilter(dup_filter)
logger.setLevel(10)

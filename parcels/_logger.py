"""Script to create a `logger` for Parcels."""

import logging
import sys

__all__ = ["logger"]


logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter(fmt="%(levelname)s: %(message)s"))
logger.addHandler(handler)
logger.setLevel(10)

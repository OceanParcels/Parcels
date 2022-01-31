from enum import IntEnum


class FieldOutOfBoundError(Exception):
    pass


class FieldOutOfBoundSurfaceError(Exception):
    pass


class FieldSamplingError(Exception):
    pass


class TimeExtrapolationError(Exception):
    pass


class GridCode(IntEnum):
    RectilinearZGrid = 0
    RectilinearSGrid = 1
    CurvilinearZGrid = 2
    CurvilinearSGrid = 3


class GridStatus(IntEnum):
    Updated = 0
    FirstUpdated = 1
    NeedsUpdate = 2

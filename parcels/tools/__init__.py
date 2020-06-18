from .converters import *  # noqa
from .error import *  # noqa
from .interpolation_utils import *  # noqa
from .loggers import *  # noqa
from .timer import *  # noqa
from .global_statics import *  # noga
from .id_generators import *  # noga
from .performance_logger import *  # noga

global idgen
idgen = SpatioTemporalIdGenerator()
idgen.setDepthLimits(0, 100)
idgen.setTimeLine(0, 240.0)
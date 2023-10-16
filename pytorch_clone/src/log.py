import os
from enum import Enum


class LoggingLevel(Enum):
    NONE = 0
    FATAL = 1
    WARNING = 2
    DEBUG = 4
    INFO = 5


class Log:
    def __init__(self, level: LoggingLevel) -> None:
        self.level = LoggingLevel(level)

    def is_level(self, l):
        return self.level == l

    def set_level(self, l):
        self.level = l

    def print(self, *args, level_prefix=True, type=LoggingLevel.NONE, **kwargs):
        if type.value <= self.level.value:
            print(f'{type.name}:' if level_prefix else "", *args, **kwargs)

    from functools import partialmethod

    info = partialmethod(print, type=LoggingLevel.INFO)
    warn = partialmethod(print, type=LoggingLevel.WARNING)
    debug = partialmethod(print, type=LoggingLevel.DEBUG)
    fatal = partialmethod(print, type=LoggingLevel.FATAL)

    is_info = partialmethod(is_level, LoggingLevel.INFO)
    is_warn = partialmethod(is_level, LoggingLevel.WARNING)
    is_debug = partialmethod(is_level, LoggingLevel.DEBUG)
    is_fatal = partialmethod(is_level, LoggingLevel.FATAL)


LOGGER = Log(LoggingLevel.DEBUG)

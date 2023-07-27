from enum import IntEnum


class GazeEventTypeEnum(IntEnum):
    UNDEFINED = 0
    FIXATION = 1
    SACCADE = 2
    BLINK = 3

from enum import IntEnum


class GazeEventTypeEnum(IntEnum):
    UNDEFINED = 0
    FIXATION = 1
    SACCADE = 2
    PSO = 3
    SMOOTH_PURSUIT = 4
    BLINK = 5

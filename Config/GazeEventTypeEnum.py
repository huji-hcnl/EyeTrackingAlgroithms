from enum import IntEnum
from typing import Union


class GazeEventTypeEnum(IntEnum):
    UNDEFINED = 0
    FIXATION = 1
    SACCADE = 2
    PSO = 3
    SMOOTH_PURSUIT = 4
    BLINK = 5


def get_event_type(v: Union[GazeEventTypeEnum, int, str], safe: bool = True) -> GazeEventTypeEnum:
    try:
        if type(v) not in [GazeEventTypeEnum, int, str]:
            raise TypeError(f"Incompatible type: {type(v)}")
        if type(v) is GazeEventTypeEnum:
            return v
        if type(v) is int:
            return GazeEventTypeEnum(v)
        if type(v) is str:
            return GazeEventTypeEnum[v.upper()]
        return GazeEventTypeEnum(v)
    except Exception as e:
        if safe and (type(e) is ValueError or type(e) is TypeError):
            return GazeEventTypeEnum.UNDEFINED
        raise e


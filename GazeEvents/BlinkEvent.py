from GazeEvents.BaseEvent import BaseEvent
from Config.GazeEventTypeEnum import GazeEventTypeEnum


class BlinkEvent(BaseEvent):
    _EVENT_TYPE = GazeEventTypeEnum.BLINK
    MIN_DURATION = 50


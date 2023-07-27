import numpy as np
import pandas as pd
from abc import ABC
from typing import List, final

import constants as cnst
from GazeEvents.GazeEventTypeEnum import GazeEventTypeEnum


class BaseEvent(ABC):
    MIN_DURATION: float = 5        # minimum duration of an event in milliseconds
    MAX_DURATION: float = 2500     # maximum duration of an event in milliseconds
    _EVENT_TYPE: GazeEventTypeEnum
    _MINIMUM_SAMPLES_PER_EVENT: int = 2

    def __init__(self, timestamps: np.ndarray):
        if len(timestamps) < self._MINIMUM_SAMPLES_PER_EVENT:
            raise ValueError("event must be at least {} samples long".format(self._MINIMUM_SAMPLES_PER_EVENT))
        if np.isnan(timestamps).any() or np.isinf(timestamps).any():
            raise ValueError("array `timestamps` must not contain NaN or infinite values")
        if (timestamps < 0).any():
            raise ValueError("array `timestamps` must not contain negative values")
        self._timestamps = timestamps

    @final
    @property
    def start_time(self) -> float:
        # Event's start time in milliseconds
        return self._timestamps[0]

    @final
    @property
    def end_time(self) -> float:
        # Event's end time in milliseconds
        return self._timestamps[-1]

    @final
    @property
    def duration(self) -> float:
        # Event's duration in milliseconds
        return self.end_time - self.start_time

    @final
    @property
    def is_outlier(self) -> bool:
        return len(self.get_outlier_reasons()) > 0

    def get_outlier_reasons(self) -> List[str]:
        reasons = []
        if self.duration < self.MIN_DURATION:
            reasons.append(f"min_{cnst.DURATION}")
        if self.duration > self.MAX_DURATION:
            reasons.append(f"max_{cnst.DURATION}")
        return reasons

    def to_series(self) -> pd.Series:
        """
        creates a pandas Series with summary of event information.
        :return: a pd.Series with the following index:
            - start_time: event's start time in milliseconds
            - end_time: event's end time in milliseconds
            - duration: event's duration in milliseconds
            - is_outlier: boolean indicating whether the event is an outlier or not
            - outlier_reasons: a list of strings indicating the reasons why the event is an outlier
        """
        return pd.Series(data=[self._EVENT_TYPE.name, self.start_time, self.end_time, self.duration, self.is_outlier,
                               self.get_outlier_reasons()],
                         index=["event_type", "start_time", "end_time", "duration", "is_outlier", "outlier_reasons"])

    @final
    def get_timestamps(self, round_decimals: int = 1, zero_corrected: bool = True) -> np.ndarray:
        """
        Returns the timestamps of the event, rounded to the specified number of decimals.
        If zero_corrected is True, the timestamps will be relative to the first timestamp of the event.
        """
        timestamps = self._timestamps  # timestamps in milliseconds
        if zero_corrected:
            timestamps = timestamps - timestamps[0]  # start from 0
        timestamps = np.round(timestamps, decimals=round_decimals)
        return timestamps

    @classmethod
    @final
    def event_type(cls) -> GazeEventTypeEnum:
        return cls._EVENT_TYPE

    @classmethod
    @final
    def set_min_duration(cls, min_duration: float):
        cls.MIN_DURATION = min_duration

    @classmethod
    @final
    def set_max_duration(cls, max_duration: float):
        cls.MAX_DURATION = max_duration

    def __repr__(self):
        event_type = self._EVENT_TYPE.name.capitalize()
        return f"{event_type} ({self.duration:.1f} ms)"

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return False
        if self._timestamps.shape != other._timestamps.shape:
            return False
        if not np.allclose(self._timestamps, other._timestamps):
            return False
        return True


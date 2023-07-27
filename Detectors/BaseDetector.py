import numpy as np
from abc import ABC, abstractmethod
from typing import final, List, Tuple

import Utils.array_utils as arr_utils
from GazeEvents.GazeEventTypeEnum import GazeEventTypeEnum


class BaseDetector(ABC):
    _MINIMUM_TIME_WITHIN_EVENT: float = 5  # min duration of single event (in milliseconds)
    _MINIMUM_TIME_BETWEEN_IDENTICAL_EVENTS: float = 5  # min duration between identical events (in milliseconds)

    def __init__(self, sr: float):
        self._sr = sr  # sampling rate in Hz

    @final
    def detect_monocular(self, x: np.ndarray, y: np.ndarray):
        x, y = self.__verify_inputs(x, y)
        candidates = self._identify_event_candidates(x, y)
        return None

    @abstractmethod
    def _identify_event_candidates(self, x: np.ndarray, y: np.ndarray) -> List[GazeEventTypeEnum]:
        raise NotImplementedError

    @final
    def clear_short_candidates(self, candidates: List[GazeEventTypeEnum]) -> List[GazeEventTypeEnum]:
        """
        Removes candidates that are too short to be events
        :param candidates: list of candidates
        :return: list of candidates without short candidates
        """
        # TODO: implement
        return candidates

    @property
    @final
    def _minimum_samples_within_event(self) -> int:
        """ minimum number of samples within a single event """
        return int(self._MINIMUM_TIME_WITHIN_EVENT * self._sr / 1000)

    @property
    @final
    def _minimum_samples_between_identical_events(self) -> int:
        """ minimum number of samples between identical events """
        return int(self._MINIMUM_TIME_BETWEEN_IDENTICAL_EVENTS * self._sr / 1000)

    @staticmethod
    def __verify_inputs(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not arr_utils.is_one_dimensional(x):
            raise ValueError("x must be one-dimensional")
        if not arr_utils.is_one_dimensional(y):
            raise ValueError("y must be one-dimensional")
        x = x.reshape(-1)
        y = y.reshape(-1)
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        return x, y







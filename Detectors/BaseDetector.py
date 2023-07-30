import numpy as np
from abc import ABC, abstractmethod
from typing import final, List, Tuple

import constants as cnst
import Utils.array_utils as arr_utils
from GazeEvents.GazeEventTypeEnum import GazeEventTypeEnum


class BaseDetector(ABC):
    """
    Base class for gaze event detectors, that segment eye-tracking data into separate events, such as blinks, saccades,
    fixations, etc.
    The detection process is implemented in detect_candidates_monocular() and detect_candidates_binocular() and is the
    same for all detectors. Detection steps are as follows:
    1. Detecting event candidates using unique algorithms for each detector (implemented in _identify_event_candidates())
    2. Filling short chunks of event candidates with GazeEventTypeEnum.UNDEFINED
    3. Merging chunks of identical event candidates that are close to each other
    4. If binocular data is available, candidates from both eyes are merged into a single list of candidates based on
    pre-defined logic (e.g. both eyes must detect a candidate for it to be considered a binocular candidate).
    """

    _MINIMUM_TIME_WITHIN_EVENT: float = 5  # min duration of single event (in milliseconds)
    _MINIMUM_TIME_BETWEEN_IDENTICAL_EVENTS: float = 5  # min duration between identical events (in milliseconds)

    def __init__(self, sr: float):
        self._sr = sr  # sampling rate in Hz

    @final
    def detect_candidates_monocular(self, x: np.ndarray, y: np.ndarray) -> List[GazeEventTypeEnum]:
        """
        Detects event-candidates in the given gaze data from a single eye. Detection steps:
        1. Find event candidates based on each Detector's implementation of _identify_event_candidates()
        2. Fill short chunks of event candidates with GazeEventTypeEnum.UNDEFINED
        3. Merge chunks of identical event candidates that are close to each other

        :param x: x-coordinates of gaze data from a single eye
        :param y: y-coordinates of gaze data from a single eye

        :return: list of GazeEventTypeEnum values, where each value indicates the type of event that is detected at the
            corresponding index in the given gaze data
        """
        x, y = self.__verify_inputs(x, y)
        candidates = self._identify_event_candidates(x, y)
        candidates = arr_utils.fill_short_chunks(arr=candidates,
                                                 min_chunk_length=self._minimum_samples_within_event,
                                                 fill_value=GazeEventTypeEnum.UNDEFINED)
        candidates = arr_utils.merge_proximal_chunks(arr=candidates,
                                                     min_chunk_length=self._minimum_samples_between_identical_events,
                                                     allow_short_chunks_of=set())
        candidates = candidates.tolist()
        return candidates

    @final
    def detect_candidates_binocular(self,
                                    x_l: np.ndarray, y_l: np.ndarray,
                                    x_r: np.ndarray, y_r: np.ndarray,
                                    detect_by: str = 'both') -> List[GazeEventTypeEnum]:
        left_candidates = self.detect_candidates_monocular(x=x_l, y=y_l)
        right_candidates = self.detect_candidates_monocular(x=x_r, y=y_r)

        detect_by = detect_by.lower()
        if detect_by == cnst.LEFT:
            return left_candidates
        if detect_by == cnst.RIGHT:
            return right_candidates

        assert len(left_candidates) == len(right_candidates)
        if detect_by in ["both", "and"]:
            # only keep candidates that are detected by both eyes
            both_candidates = [left_cand if left_cand == right_cand else GazeEventTypeEnum.UNDEFINED
                               for left_cand, right_cand in zip(left_candidates, right_candidates)]
            return both_candidates
        if detect_by in ["either", "or"]:
            either_candidates = [left_cand or right_cand for left_cand, right_cand
                                 in zip(left_candidates, right_candidates)]
            return either_candidates

        # TODO: support more complex logic: fixations & blinks are monocular, saccades are binocular, etc.

        raise ValueError(f"invalid value for `detect_by`: {detect_by}")

    @abstractmethod
    def _identify_event_candidates(self, x: np.ndarray, y: np.ndarray) -> List[GazeEventTypeEnum]:
        raise NotImplementedError

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







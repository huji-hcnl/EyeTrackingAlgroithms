import traceback
import numpy as np
from abc import ABC, abstractmethod
from typing import final, List, Tuple, Set

import constants as cnst
import Utils.array_utils as arr_utils
from Config.GazeEventTypeEnum import GazeEventTypeEnum


class BaseDetector(ABC):
    """
    Base class for gaze event detectors, that segment eye-tracking data into separate events, such as blinks, saccades,
    fixations, etc.
    The detection process is implemented in detect_candidates_monocular() and detect_candidates_binocular() and is the
    same for all detectors. Detection steps are as follows:
    1. Verify the input is valid
    2. Calculate the sampling rate of the given timestamps
    3. Detecting blink candidates based on missing data in the recorded gaze data
    4. Detecting event candidates using unique algorithms for each detector (implemented in _identify_event_candidates())
    5. Filling short chunks of event candidates with GazeEventTypeEnum.UNDEFINED
    6. Merging chunks of identical event candidates that are close to each other
    7. If binocular data is available, candidates from both eyes are merged into a single list of candidates based on
    pre-defined logic (e.g. both eyes must detect a candidate for it to be considered a binocular candidate).
    """

    _MISSING_VALUE: float = np.nan  # value that represents missing data in the gaze data
    _MINIMUM_TIME_WITHIN_EVENT: float = 5  # min duration of single event (in milliseconds)
    _MINIMUM_TIME_BETWEEN_IDENTICAL_EVENTS: float = 5  # min duration between identical events (in milliseconds)

    def __init__(self, missing_value: float = _MISSING_VALUE):
        self._missing_value = missing_value
        self._sr = np.nan  # sampling rate in Hz

    @final
    def detect_candidates_monocular(self, t: np.ndarray, x: np.ndarray, y: np.ndarray,
                                    expand_blink_by_ms: int = 0) -> List[GazeEventTypeEnum]:
        """
        Detects event-candidates in the given gaze data from a single eye. Detection steps:
        1. Verify that t, x and y are valid inputs
        2. Detect blink candidates when there is missing gaze data
        3. Find event candidates based on each Detector's implementation of _identify_event_candidates()
        4. Fill short chunks of event candidates with GazeEventTypeEnum.UNDEFINED
        5. Merge chunks of identical event candidates that are close to each other

        :param t: timestamps of gaze data from a single eye
        :param x: x-coordinates of gaze data from a single eye
        :param y: y-coordinates of gaze data from a single eye
        :param expand_blink_by_ms: number of milliseconds to expand the blink candidates by. For example, if
            `expand_blink_by_ms=20`, then 20ms-worth-samples before and after each blink candidate will also be marked
            as blink candidates. This is useful for correcting blinks that are detected too late or too early.

        :return: list of GazeEventTypeEnum values, where each value indicates the type of event that is detected at the
            corresponding index in the given gaze data
        """
        candidates = np.full_like(x, GazeEventTypeEnum.UNDEFINED)
        try:
            t, x, y = self._verify_inputs(t, x, y)
            self._sr = self._calculate_sampling_rate(t)
            x, y, candidates = self._identify_blink_candidates(x, y, candidates, expand_blink_by_ms)
            candidates = self._identify_gaze_event_candidates(x, y, candidates)
            candidates = self._set_short_chunks_as_undefined(candidates)
            candidates = self._merge_proximal_chunks_of_identical_values(candidates)
        except ValueError as e:
            trace = traceback.format_exc()
            print(f"Failed to detect gaze-event candidates:\t{e.__class__.__name__}\n\t{trace}")
        return list(candidates)

    @final
    def detect_candidates_binocular(self,
                                    t: np.ndarray,
                                    x_l: np.ndarray, y_l: np.ndarray,
                                    x_r: np.ndarray, y_r: np.ndarray,
                                    expand_blink_by_ms: int = 0,
                                    detect_by: str = 'both') -> List[GazeEventTypeEnum]:
        """
        Detects event-candidates in the given gaze data from both eyes. First, candidates are detected separately for
        each eye using detect_candidates_monocular(). Then, the candidates from both eyes are merged into a single list
        based on the value of `detect_by`.

        :param t: timestamps of gaze data
        :param x_l: x-coordinates of gaze data from the left eye
        :param y_l: y-coordinates of gaze data from the left eye
        :param x_r: x-coordinates of gaze data from the right eye
        :param y_r: y-coordinates of gaze data from the right eye
        :param expand_blink_by_ms: number of milliseconds to expand the blink candidates by. For example, if
            `expand_blink_by_ms=20`, then 20ms-worth-samples before and after each blink candidate will also be marked
            as blink candidates. This is useful for correcting blinks that are detected too late or too early.
        :param detect_by: how to merge candidates from both eyes. Valid values are:
            - 'left': use candidates from the left eye only
            - 'right': use candidates from the right eye only
            - 'both'/'and': use candidates that are detected by both eyes simultaneously
            - 'either'/'or': use candidates that are detected by either eye (or both)

        :return: list of GazeEventTypeEnum values, where each value indicates the type of event that is detected at the
            corresponding index in the given gaze data
        """
        left_candidates = self.detect_candidates_monocular(x=x_l, y=y_l, expand_blink_by_ms=expand_blink_by_ms)
        right_candidates = self.detect_candidates_monocular(x=x_r, y=y_r, expand_blink_by_ms=expand_blink_by_ms)

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

    def _verify_inputs(self, t: np.ndarray, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not arr_utils.is_one_dimensional(t):
            raise ValueError("`t` must be one-dimensional")
        if not arr_utils.is_one_dimensional(x):
            raise ValueError("`x` must be one-dimensional")
        if not arr_utils.is_one_dimensional(y):
            raise ValueError("`y` must be one-dimensional")
        t = t.reshape(-1)
        x = x.reshape(-1)
        y = y.reshape(-1)
        if len(t) != len(x) or len(t) != len(y) or len(x) != len(y):
            raise ValueError("t, x and y must have the same length")
        return t, x, y

    @final
    def _identify_blink_candidates(self, x: np.ndarray, y: np.ndarray,
                                   candidates: np.ndarray,
                                   expand_blink_by_ms: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Identifies blink candidates in the given gaze data:
        1. Marks samples with missing gaze data (i.e. x or y is NaN) as blink candidates
        2. Ignores blink candidates that are too short (i.e. less than `self._minimum_samples_within_event` consecutive
        samples).
        3. Merges blink candidates that are close to each other (i.e. less than `self._minimum_samples_between_identical_events`
        samples apart).
        4. Expands blink candidates by `expand_blink_by_ms` milliseconds, i.e. marks samples that are `expand_blink_by_ms`
        milliseconds before and after each blink candidate as blink candidates as well.
        5. Modifies the gaze data by setting the x and y coordinates of blink candidates to NaN.

        :param x: x-coordinates of gaze data from a single eye
        :param y: y-coordinates of gaze data from a single eye
        :param candidates: list of event candidates, where each value indicates the type of event that is detected at
            the corresponding index in the given gaze data
        :param expand_blink_by_ms: number of milliseconds to expand the blink candidates by. For example, if
            `expand_blink_by_ms=20`, then 20ms-worth-samples before and after each blink candidate will also be marked
            as blink candidates. This is useful for correcting blinks that are detected too late or too early.

        :return: modified x and y coordinates, and a list of event candidates including where the blinks were detected
        """
        if expand_blink_by_ms < 0:
            raise ValueError("Argument `expand_blink_by_ms` must be non-negative")

        candidates_copy = np.asarray(candidates, dtype=GazeEventTypeEnum).copy()
        x_missing = np.array([self._is_missing_value(val) for val in x])
        y_missing = np.array([self._is_missing_value(val) for val in y])
        candidates_copy[x_missing | y_missing] = GazeEventTypeEnum.BLINK
        candidates_copy = self._set_short_chunks_as_undefined(candidates_copy)
        candidates_copy = self._merge_proximal_chunks_of_identical_values(candidates_copy)

        # wherever there's a blink candidate, expand it by `expand_blink_by_ms` milliseconds
        if expand_blink_by_ms > 0:
            expand_blink_by_samples = round(expand_blink_by_ms * self._sr / 1000)
            blink_indices = np.where(candidates_copy == GazeEventTypeEnum.BLINK)[0]
            for blink_idx in blink_indices:
                blink_start = max(0, blink_idx - expand_blink_by_samples)
                blink_end = min(len(candidates_copy), blink_idx + expand_blink_by_samples)
                candidates_copy[blink_start:blink_end] = GazeEventTypeEnum.BLINK

        x_copy = x.copy()
        y_copy = y.copy()
        x_copy[candidates_copy == GazeEventTypeEnum.BLINK] = np.nan
        y_copy[candidates_copy == GazeEventTypeEnum.BLINK] = np.nan
        return x_copy, y_copy, candidates_copy

    @abstractmethod
    def _identify_gaze_event_candidates(self, x: np.ndarray, y: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """
        Identifies gaze-event (fixations, saccades, etc.) candidates in the given gaze data from a single eye
        """
        raise NotImplementedError

    @final
    def _set_short_chunks_as_undefined(self, arr) -> np.ndarray:
        """
        If a "chunk" of identical values is shorter than `self._minimum_samples_within_event`, we fill the indices of
        said chunk with value GazeEventTypeEnum.UNDEFINED.
        """
        arr_copy = np.asarray(arr).copy()
        chunk_indices = arr_utils.get_chunk_indices(arr)
        for chunk_idx in chunk_indices:
            if len(chunk_idx) < self._minimum_samples_within_event():
                arr_copy[chunk_idx] = GazeEventTypeEnum.UNDEFINED
        return arr_copy

    @final
    def _merge_proximal_chunks_of_identical_values(self, arr,
                                                   allow_short_chunks_of: Set = None) -> np.ndarray:
        """
        If two "chunks" of identical values are separated by a short "chunk" of other values, merges the two chunks into
        one chunk by filling the middle chunk with the value of the left chunk.
        Chunks with values specified in `allow_short_chunks_of` are not merged.
        """
        if allow_short_chunks_of is None or len(allow_short_chunks_of) == 0:
            allow_short_chunks_of = set()

        arr_copy = np.asarray(arr).copy()
        chunk_indices = arr_utils.get_chunk_indices(arr)
        for i, middle_chunk in enumerate(chunk_indices):
            if i == 0 or i == len(chunk_indices) - 1:
                # ignore the first and last chunk
                continue
            if len(middle_chunk) >= self._minimum_samples_between_identical_events():
                # ignore chunks that are long enough
                continue
            middle_chunk_value = arr_copy[middle_chunk[0]]
            if middle_chunk_value in allow_short_chunks_of:
                # ignore chunks of the specified types
                continue
            left_chunk_value = arr_copy[chunk_indices[i - 1][0]]
            right_chunk_value = arr_copy[chunk_indices[i + 1][0]]
            if left_chunk_value != right_chunk_value:
                # ignore middle chunks if the left and right chunks are not identical
                continue

            # reached here if the middle chunk is short, its value is not allowed to be short, and left and right chunks
            # are identical. merge the left and right chunks by filling `middle_chunk` with the value of `left_chunk`.
            arr_copy[middle_chunk] = left_chunk_value
        return arr_copy

    @final
    def _minimum_samples_within_event(self) -> int:
        """
        Calculates the minimum number of samples within a single event
        """
        return round(self._MINIMUM_TIME_WITHIN_EVENT * self._sr / 1000)

    @final
    def _minimum_samples_between_identical_events(self) -> int:
        """
        Calculates the minimum number of samples between identical events
        """
        return round(self._MINIMUM_TIME_BETWEEN_IDENTICAL_EVENTS * self._sr / 1000)

    @staticmethod
    @final
    def _calculate_sampling_rate(ms: np.ndarray) -> float:
        """
        Calculates the sampling rate of the given timestamps in Hz.
        :param ms: timestamps in milliseconds (floating-point, not integer)
        """
        if len(ms) < 2:
            raise ValueError("timestamps must be of length at least 2")
        sr = cnst.MILLISECONDS_PER_SECOND / np.median(np.diff(ms))
        if not np.isfinite(sr):
            raise RuntimeError("Error calculating sampling rate")
        return sr

    def _is_missing_value(self, value: float) -> bool:
        if np.isnan(self._missing_value):
            return np.isnan(value)
        return value == self._missing_value

import numpy as np
from Config.GazeEventTypeEnum import GazeEventTypeEnum
from Detectors.BaseDetector import BaseDetector
from Utils import visual_angle_utils


class IDTDetector(BaseDetector):
    # a threshold value that determines the maximum allowable dispersion for a fixation.
    __DEFAULT_DISPERSION_THRESHOLD_ANGLE = 0.5  # degrees
    __DEFAULT_WINDOW_DURATION = 100  # ms- size of the duration time window = minimum allowed fixation duration

    def __init__(self, viewer_distance: float, pixel_size: float,
                 missing_value: float = BaseDetector._MISSING_VALUE,
                 dispersion_threshold: float = __DEFAULT_DISPERSION_THRESHOLD_ANGLE,
                 window_duration: int = __DEFAULT_WINDOW_DURATION):
        super().__init__(missing_value)
        if viewer_distance <= 0 or pixel_size <= 0 or dispersion_threshold <= 0 or window_duration <= 0:
            raise ValueError("all parameters must be positive")
        # visual angle to pixels
        self._dispersion_threshold_pixels = visual_angle_utils.visual_angle_to_pixels(dispersion_threshold,
                                                                                      viewer_distance, pixel_size)
        self._window_duration = window_duration  # in ms

    def _identify_gaze_event_candidates(self, x: np.ndarray, y: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        num_samples = len(x)
        candidates_copy = np.asarray(candidates, dtype=GazeEventTypeEnum).copy()

        # initialize a window in the minimum size for fixation
        window_size = round(self._window_duration * self._sr / 1000)
        window_start_idx = 0
        window_end_idx = window_size - 1
        # if the window is not in minimum size for fixation- the samples are saccades
        if window_end_idx >= num_samples:
            raise ValueError("window size cannot be larger than number of samples")

        fixation_flag = False  # if the window is fixation potential = True
        while window_end_idx < num_samples:
            dispersion = IDTDetector._calculate_dispersion(x, y, window_start_idx, window_end_idx)
            # as long as the dispersion doesn't exceed the threshold- it is a fixation,
            # and expand the window to the right
            if dispersion < self._dispersion_threshold_pixels:
                fixation_flag = True
                window_end_idx += 1
            # when exceeding the dispersion threshold in a fixation window: label all samples in the window as fixation
            # and start new window in the end of the old one
            elif fixation_flag:
                candidates_copy[window_start_idx: window_end_idx] = GazeEventTypeEnum.FIXATION
                window_start_idx = window_end_idx
                window_end_idx = window_start_idx + window_size - 1
                fixation_flag = False
            # when exceeding the dispersion threshold in a new window: label current sample as saccade
            # and start new window in the next sample
            else:
                candidates_copy[window_start_idx] = GazeEventTypeEnum.SACCADE
                window_start_idx += 1
                window_end_idx += 1

        # in case the last window is a fixation window
        if fixation_flag:
            candidates_copy[window_start_idx: window_end_idx] = GazeEventTypeEnum.FIXATION
        # in case last window is not a fixation window- all remaining samples will be saccades
        else:
            candidates_copy[window_start_idx: window_end_idx] = GazeEventTypeEnum.SACCADE

        return candidates_copy

    @staticmethod
    def _calculate_dispersion(x, y, window_start: int, window_end: int):
        horizontal_window = x[window_start: window_end + 1]
        vertical_window = y[window_start: window_end + 1]

        # dispersion = sum of the largest horizontal and vertical distances between samples in the window
        dispersion = ((max(horizontal_window) - min(horizontal_window)) +
                      (max(vertical_window) - min(vertical_window)))
        return dispersion

from typing import List
import numpy as np
from GazeEvents.GazeEventTypeEnum import GazeEventTypeEnum
from Detectors.BaseDetector import BaseDetector


class IDTDetector(BaseDetector):
    # a threshold value that determines the maximum allowable dispersion for a fixation.
    __DEFAULT_DISPERSION_THRESHOLD = 0.5  # convert from visual angle to pixels
    __DEFAULT_WINDOW_DURATION = 100  # ms- size of the duration time window = minimum allowed fixation duration

    def __init__(self, sr: float, viewer_distance: float,
                 dispersion_threshold: float = __DEFAULT_DISPERSION_THRESHOLD,
                 window_duration: int = __DEFAULT_WINDOW_DURATION):
        super().__init__(sr=sr)
        self._dispersion_threshold = dispersion_threshold
        self._window_duration = window_duration
        self._window_dim = int((sr / 1000) * window_duration)  # Hertz to ms
        self._viewer_distance = viewer_distance

    def _identify_gaze_event_candidates(self, x: np.ndarray, y: np.ndarray,
                                        candidates: List[GazeEventTypeEnum]) -> List[GazeEventTypeEnum]:
        num_samples = len(x)
        candidates = np.zeros(num_samples, dtype=GazeEventTypeEnum)

        # initialize a window in the minimum size for fixation
        window_start_idx = 0
        window_end_idx = self._window_dim - 1
        # if the window is not in minimum size for fixation- the samples are saccades
        if window_end_idx >= num_samples:
            candidates[window_start_idx: num_samples] = GazeEventTypeEnum.SACCADE
            return list(candidates)

        fixation_flag = False  # if the window is fixation potential = True
        while window_end_idx < num_samples:
            dispersion = IDTDetector._calculate_dispersion(x, y, window_start_idx, window_end_idx)
            # as long as the dispersion doesn't exceed the threshold- it is a fixation,
            # and expand the window to the right
            if dispersion < self._dispersion_threshold:
                fixation_flag = True
                window_end_idx += 1
            # when exceeding the dispersion threshold in a fixation window: label all samples in the window as fixation
            # and start new window in the end of the old one
            elif fixation_flag:
                candidates[window_start_idx: window_end_idx] = GazeEventTypeEnum.FIXATION
                window_start_idx = window_end_idx
                window_end_idx += self._window_dim - 1
                fixation_flag = False
            # when exceeding the dispersion threshold in a new window: label current sample as saccade
            # and start new window in the next sample
            else:
                candidates[window_start_idx] = GazeEventTypeEnum.SACCADE
                window_start_idx += 1
                window_end_idx += 1

        # in case the last window is a fixation window
        if fixation_flag:
            candidates[window_start_idx: window_end_idx + 1] = GazeEventTypeEnum.FIXATION
        # in case last window is not a fixation window- all remaining samples will be saccades
        else:
            candidates[window_start_idx: window_end_idx + 1] = GazeEventTypeEnum.SACCADE

        return list(candidates)

    @staticmethod
    def _calculate_dispersion(x, y, window_start: int, window_end: int):
        horizontal_window = x[window_start: window_end + 1]
        vertical_window = y[window_start: window_end + 1]

        # dispersion=average of the largest horizontal and vertical distances between samples in the window
        dispersion = ((max(horizontal_window) - min(horizontal_window)) +
                      (max(vertical_window) - min(vertical_window)))
        return dispersion


# if __name__ == "__main__":
#     data = np.array([[1, 2], [1, 2], [1, 3], [5, 7], [5, 7]])  # Replace with your eye movement data
#     Xs = data[:, 0]
#     Ys = data[:, 1]
#     detector = IDTDetector(500.0, 2, 5, 5)
#     print(detector._identify_gaze_event_candidates(Xs, Ys, []))

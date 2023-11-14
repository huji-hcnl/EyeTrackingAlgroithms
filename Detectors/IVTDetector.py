import numpy as np
from Config.GazeEventTypeEnum import GazeEventTypeEnum
from Detectors.BaseDetector import BaseDetector


class IVTDetector(BaseDetector):
    __DEFAULT_VELOCITY_THRESHOLD = 0.5

    def __init__(self,
                 velocity_threshold: float = __DEFAULT_VELOCITY_THRESHOLD):
        super().__init__()
        if velocity_threshold <= 0:
            raise ValueError("all parameters must be positive")
        self._velocity_threshold = velocity_threshold

    def _identify_gaze_event_candidates(self, x: np.ndarray, y: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        # calculate velocity by difference between coordinates
        candidates = np.array(candidates)

        diff_x = np.diff(x)
        diff_y = np.diff(y)

        # we assume that the frequency is 500Hz so there is 2ms gap between every two samples
        velocity = np.sqrt(np.power(diff_x, 2) + np.power(diff_y, 2)) / 2

        # velocities below threshold = fixation (label 1), above = saccade (label 2)
        candidates[velocity < self._velocity_threshold] = GazeEventTypeEnum.FIXATION
        candidates[velocity >= self._velocity_threshold] = GazeEventTypeEnum.SACCADE

        return candidates

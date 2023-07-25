import numpy as np


class IVTDetector:
    __DEFAULT_VELOCITY_THRESHOLD = 0.5

    def __init__(self,
                 velocity_threshold: float = __DEFAULT_VELOCITY_THRESHOLD):
        self._velocity_threshold = velocity_threshold

    def detect(self, x_coords, y_coords) -> np.ndarray:
        # calculate velocity by difference between coordinates
        diff_x = np.diff(x_coords)
        diff_y = np.diff(y_coords)

        # we assume that the frequency is 500Hz so there is 2ms gap between every two samples
        velocity = np.sqrt(np.power(diff_x, 2) + np.power(diff_y, 2)) / 2

        # velocities below threshold = fixation (label 1), above = saccade (label 2)
        labels = np.ndarray(shape=velocity.shape)
        labels[velocity < self._velocity_threshold] = 1
        labels[velocity >= self._velocity_threshold] = 2

        return labels

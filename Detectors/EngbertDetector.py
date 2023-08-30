import numpy as np
from typing import List, Tuple

from Detectors.BaseDetector import BaseDetector
from GazeEvents.GazeEventTypeEnum import GazeEventTypeEnum


class EngbertDetector(BaseDetector):
    """
    This class implements the algorithm described by Engbert, Kliegl, and Mergenthaler in
        "Microsaccades uncover the orientation of covert attention" (2003)
        "Microsaccades are triggered by low retinal image slip" (2006)

    Implementation is based on the following repositories:
        - https://github.com/Yuvishap/Gaze-Project/blob/master/Gaze/src/pre_processing/business/EngbertFixationsLogic.py
        - https://github.com/odedwer/EyelinkProcessor/blob/master/SaccadeDetectors.py

    General algorithm:
    1. Calculate the velocity of the gaze data in both axes
    2. Calculate the median-standard-deviation of the velocity in both axes
    3. Calculate the noise threshold as the multiple of the median-standard-deviation with the constant lambda_noise_threshold
    4. Identify saccade candidates as samples with velocity greater than the noise threshold
    """
    _LAMBDA_NOISE_THRESHOLD = 5  # default value used in the original paper
    _DERIVATION_WINDOW_SIZE = 2  # default value used in the original paper

    def __init__(self, sr: float,
                 lambda_noise_threshold: float = _LAMBDA_NOISE_THRESHOLD,
                 derivation_window_size: int = _DERIVATION_WINDOW_SIZE):
        super().__init__(sr)
        if lambda_noise_threshold <= 0:
            raise ValueError("lambda_noise_threshold must be positive")
        if derivation_window_size <= 0:
            raise ValueError("derivation_window_size must be positive")
        self._lambda_noise_threshold = lambda_noise_threshold
        self._derivation_window_size = round(derivation_window_size)

    def _identify_gaze_event_candidates(self, x: np.ndarray, y: np.ndarray, candidates: np.ndarray) -> List[GazeEventTypeEnum]:
        x_velocity = self._calculate_axial_velocity(x)
        x_std = self._median_standard_deviation(x_velocity)
        x_noise_threshold = self._lambda_noise_threshold * x_std

        y_velocity = self._calculate_axial_velocity(y)
        y_std = self._median_standard_deviation(y_velocity)
        y_noise_threshold = self._lambda_noise_threshold * y_std

        ellipse = (x_velocity / x_noise_threshold) ** 2 + (y_velocity / y_noise_threshold) ** 2
        candidates_copy = np.asarray(candidates, dtype=GazeEventTypeEnum).copy()
        candidates_copy[ellipse < 1] = GazeEventTypeEnum.FIXATION
        candidates_copy[ellipse >= 1] = GazeEventTypeEnum.SACCADE
        return list(candidates_copy)

    def _verify_inputs(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x, y = super()._verify_inputs(x, y)
        if len(x) < 2 * self._derivation_window_size:
            raise ValueError(
                f"x and y must be of length at least 2 * derivation_window_size (={2 * self._derivation_window_size})")
        return x, y

    def _calculate_axial_velocity(self, arr) -> np.ndarray:
        """
        Calculates the velocity along a single axis, based on the algorithm described in the original paper:
        1. Sum values in a window of size window_size, *before* the current sample:
            sum_before = arr(t-1) + arr(t-2) + ... + arr(t-ws)
        2. Sum values in a window of size window_size, *after* the current sample
            sum_after = arr(t+1) + arr(t+2) + ... + arr(t+ws)
        3. Calculate the difference between the two sums
            diff = sum_after - sum_before
        4. Divide by the time-difference, calculated as `sampling_rate` / (2 * `window_size`)
            velocity = diff * (sampling_rate / (2 * (window_size + 1))
        5. For the first and last `window_size` samples, the velocity is np.nan
        """
        arr_copy = np.copy(arr)
        ws = self._derivation_window_size
        velocities = np.full_like(arr_copy, np.nan)
        for t in range(ws, len(arr_copy) - ws):
            sum_before = np.sum(arr_copy[t - ws:t])
            sum_after = np.sum(arr_copy[t + 1:t + ws + 1])
            diff = sum_after - sum_before
            velocities[t] = diff * (self._sr / (2 * (ws + 1)))
        return velocities

    @staticmethod
    def _median_standard_deviation(arr) -> float:
        """
        Calculates the median standard deviation of the given array
        """
        squared_median = np.power(np.nanmedian(arr), 2)
        median_of_squares = np.nanmedian(np.power(arr, 2))
        sd = np.sqrt(median_of_squares - squared_median)
        return float(np.nanmax([sd, 1e-6]))




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
        - https://github.com/esdalmaijer/PyGazeAnalyser/blob/master/pygazeanalyser/detectors.py

    General algorithm:
    1. Calculate the velocity of the gaze data in both axes
    2. Calculate the median-standard-deviation of the velocity in both axes
    3. Calculate the noise threshold as the multiple of the median-standard-deviation with the constant lambda_noise_threshold
    4. Identify saccade candidates as samples with velocity greater than the noise threshold
    """
    _LAMBDA_NOISE_THRESHOLD = 5  # default value used in the original paper
    _DERIVATION_WINDOW_SIZE = 3  # default value used in the original paper

    def __init__(self, sr: float,
                 lambda_noise_threshold: float = _LAMBDA_NOISE_THRESHOLD,
                 derivation_window_size: int = _DERIVATION_WINDOW_SIZE):
        super().__init__(sr)
        if lambda_noise_threshold <= 0:
            raise ValueError("lambda_noise_threshold must be positive")
        if derivation_window_size <= 0:
            raise ValueError("derivation_window_size must be positive")
        self._lambda_noise_threshold = lambda_noise_threshold
        self._derivation_window_size = derivation_window_size

    def _identify_event_candidates(self, x: np.ndarray, y: np.ndarray) -> List[GazeEventTypeEnum]:
        raise NotImplementedError

    def _verify_inputs(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x, y = super()._verify_inputs(x, y)
        if len(x) < 2 * self._derivation_window_size:
            raise ValueError(
                f"x and y must be of length at least 2 * derivation_window_size (={2 * self._derivation_window_size})")
        return x, y


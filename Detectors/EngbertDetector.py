import numpy as np
from typing import List

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

    def _identify_event_candidates(self, x: np.ndarray, y: np.ndarray) -> List[GazeEventTypeEnum]:
        raise NotImplementedError


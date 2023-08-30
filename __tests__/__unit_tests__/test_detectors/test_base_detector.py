import unittest
import numpy as np

from Detectors.EngbertDetector import EngbertDetector
from GazeEvents.GazeEventTypeEnum import GazeEventTypeEnum


class TestBaseDetector(unittest.TestCase):
    _SR = 500
    _LAMBDA = 5
    _WS = 2

    DETECTOR = EngbertDetector(sr=500, lambda_noise_threshold=_LAMBDA, derivation_window_size=_WS)

    def test_set_short_chunks_as_undefined(self):
        arr = np.array([])
        self.assertTrue(np.array_equal(self.DETECTOR._set_short_chunks_as_undefined(arr), arr))

        arr = np.array([GazeEventTypeEnum.FIXATION] * 3 +
                       [GazeEventTypeEnum.SACCADE] * 1 +
                       [GazeEventTypeEnum.FIXATION] * 2)
        expected = np.array([GazeEventTypeEnum.FIXATION] * 3 +
                            [GazeEventTypeEnum.UNDEFINED] * 1 +
                            [GazeEventTypeEnum.FIXATION] * 2)
        res = self.DETECTOR._set_short_chunks_as_undefined(arr)
        self.assertTrue(np.array_equal(res, expected))

    def test_merge_proximal_chunks_of_identical_values(self):
        arr = np.array([])
        self.assertTrue(np.array_equal(self.DETECTOR._merge_proximal_chunks_of_identical_values(arr), arr))

        arr = np.array(([GazeEventTypeEnum.FIXATION] * 3 +
                        [GazeEventTypeEnum.UNDEFINED] * 1 +
                        [GazeEventTypeEnum.FIXATION] * 2))
        expected = np.array(([GazeEventTypeEnum.FIXATION] * 6))
        res = self.DETECTOR._merge_proximal_chunks_of_identical_values(arr)
        self.assertTrue(np.array_equal(res, expected))

        arr = np.array(([GazeEventTypeEnum.FIXATION] * 3 +
                        [GazeEventTypeEnum.SACCADE] * 1 +
                        [GazeEventTypeEnum.FIXATION] * 2))
        expected = arr
        res = self.DETECTOR._merge_proximal_chunks_of_identical_values(arr, allow_short_chunks_of={
            GazeEventTypeEnum.SACCADE})
        self.assertTrue(np.array_equal(res, expected))

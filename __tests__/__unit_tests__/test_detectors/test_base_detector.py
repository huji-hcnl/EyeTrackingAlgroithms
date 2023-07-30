import unittest

from Detectors.EngbertDetector import EngbertDetector
from GazeEvents.GazeEventTypeEnum import GazeEventTypeEnum


class TestBaseDetector(unittest.TestCase):
    _SR = 500
    _LAMBDA = 5
    _WS = 2

    DETECTOR = EngbertDetector(sr=500, lambda_noise_threshold=_LAMBDA, derivation_window_size=_WS)

    def test_fill_short_chunks_with_undefined(self):
        arr = []
        self.assertEqual(self.DETECTOR._set_short_chunks_as_undefined(arr), [])

        arr = ([GazeEventTypeEnum.FIXATION] * 3 +
               [GazeEventTypeEnum.SACCADE] * 1 +
               [GazeEventTypeEnum.FIXATION] * 2)
        expected = ([GazeEventTypeEnum.FIXATION] * 3 +
                    [GazeEventTypeEnum.UNDEFINED] * 1 +
                    [GazeEventTypeEnum.FIXATION] * 2)
        self.assertEqual(self.DETECTOR._set_short_chunks_as_undefined(arr), expected)

    def test_merge_proximal_chunks_of_identical_values(self):
        arr = []
        self.assertEqual(self.DETECTOR._merge_proximal_chunks_of_identical_values(arr), [])

        arr = ([GazeEventTypeEnum.FIXATION] * 3 +
               [GazeEventTypeEnum.UNDEFINED] * 1 +
               [GazeEventTypeEnum.FIXATION] * 2)
        expected = ([GazeEventTypeEnum.FIXATION] * 6)
        self.assertEqual(self.DETECTOR._merge_proximal_chunks_of_identical_values(arr), expected)

        arr = ([GazeEventTypeEnum.FIXATION] * 3 +
               [GazeEventTypeEnum.SACCADE] * 1 +
               [GazeEventTypeEnum.FIXATION] * 2)
        expected = arr
        self.assertEqual(self.DETECTOR._merge_proximal_chunks_of_identical_values(arr,
                                                                                  allow_short_chunks_of={
                                                                                      GazeEventTypeEnum.SACCADE}),
                         expected)

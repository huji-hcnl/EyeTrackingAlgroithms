import unittest
import numpy as np

import constants as cnst
from Detectors.EngbertDetector import EngbertDetector


class TestEngbertDetector(unittest.TestCase):
    _SR = 500
    _LAMBDA = 5
    _WS = 2

    def test_init(self):
        with self.assertRaises(ValueError):
            _ = EngbertDetector(lambda_noise_threshold=-1*self._LAMBDA, derivation_window_size=self._WS)
        with self.assertRaises(ValueError):
            _ = EngbertDetector(lambda_noise_threshold=self._LAMBDA, derivation_window_size=-1*self._WS)

    def test_calculate_axial_velocity(self):
        detector = self.__create_detector(set_sr=True)

        arr = np.zeros(10)
        expected = np.array([np.nan] * 2 + [0] * 6 + [np.nan] * 2)
        self.assertTrue(np.array_equal(detector._calculate_axial_velocity(arr), expected, equal_nan=True))

        arr = np.ones(10)
        self.assertTrue(np.array_equal(detector._calculate_axial_velocity(arr), expected, equal_nan=True))

        arr = np.concatenate([np.zeros(5), np.ones(5)])
        expected = np.array([np.nan] * 2 + [0, 1, 2, 2, 1, 0] + [np.nan] * 2) * self._SR / ((self._WS + 1) * 2)
        self.assertTrue(np.array_equal(detector._calculate_axial_velocity(arr), expected, equal_nan=True))

        arr = np.arange(10).astype(float)
        expected = np.array([np.nan] * 2 + [1] * 6 + [np.nan] * 2) * self._SR
        self.assertTrue(np.array_equal(detector._calculate_axial_velocity(arr), expected, equal_nan=True))

        arr[5] = np.nan
        expected = np.array([np.nan, np.nan, 1, np.nan, np.nan, 1, np.nan, np.nan, np.nan, np.nan]) * self._SR
        self.assertTrue(np.array_equal(detector._calculate_axial_velocity(arr), expected, equal_nan=True))

        arr = np.arange(0, 10, 2).astype(float)
        arr = np.concatenate([arr, arr[::-1]])
        expected = np.array([np.nan] * 2 + [12, 10, 4, -4, -10, -12] + [np.nan] * 2) * self._SR / ((self._WS + 1) * 2)
        self.assertTrue(np.allclose(detector._calculate_axial_velocity(arr), expected, equal_nan=True))

    def test_median_standard_deviation(self):
        detector = self.__create_detector(set_sr=True)

        arr = np.zeros(10)
        self.assertEqual(detector._median_standard_deviation(arr), cnst.EPSILON)

        arr = np.arange(11).astype(float)
        self.assertEqual(detector._median_standard_deviation(arr), cnst.EPSILON)

        arr[0] = np.nan
        self.assertEqual(detector._median_standard_deviation(arr), 0.5)

        arr = np.arange(11).astype(float)
        arr[5] = np.nan
        self.assertEqual(detector._median_standard_deviation(arr), 1)

    def __create_detector(self, set_sr=True):
        detector = EngbertDetector(lambda_noise_threshold=self._LAMBDA, derivation_window_size=self._WS)
        if set_sr:
            detector._sr = self._SR
        return detector

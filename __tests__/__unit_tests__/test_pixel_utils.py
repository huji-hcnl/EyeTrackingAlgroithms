import unittest
import numpy as np

from Utils import pixel_utils as pixel_utils


class TestUtils(unittest.TestCase):

    def test_calculate_euclidean_distances(self):
        sqrt2 = np.sqrt(2)
        x_coords = np.arange(0, 5)
        y_coords1 = np.arange(0, 5)
        self.assertTrue(np.array_equal(np.array([sqrt2, sqrt2, sqrt2, sqrt2]),
                                       pixel_utils.calculate_euclidean_distances(x_coords, y_coords1),
                                       equal_nan=True))
        y_coords2 = np.zeros_like(x_coords)
        self.assertTrue(np.array_equal(np.array([1, 1, 1, 1]),
                                       pixel_utils.calculate_euclidean_distances(x_coords, y_coords2),
                                       equal_nan=True))
        y_coords3 = y_coords1.copy().astype(float)
        y_coords3[2] = np.nan
        self.assertTrue(np.array_equal(np.array([sqrt2, np.nan, np.nan, sqrt2]),
                                       pixel_utils.calculate_euclidean_distances(x_coords, y_coords3),
                                       equal_nan=True))
        y_coords4 = y_coords1[:-1].copy()
        self.assertRaises(AssertionError, pixel_utils.calculate_euclidean_distances, x_coords, y_coords4)

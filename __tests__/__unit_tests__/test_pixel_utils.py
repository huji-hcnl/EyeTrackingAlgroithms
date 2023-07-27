import unittest
import numpy as np

from Utils import pixel_utils as pixel_utils


class TestPixelUtils(unittest.TestCase):

    def test_calculate_euclidean_distances(self):
        sqrt2 = np.sqrt(2)
        x_coords = np.arange(0, 5)
        y_coords1 = np.arange(0, 5)
        self.assertTrue(np.array_equal(np.array([np.nan, sqrt2, sqrt2, sqrt2, sqrt2]),
                                       pixel_utils.calculate_euclidean_distances(x_coords, y_coords1),
                                       equal_nan=True))
        y_coords2 = np.zeros_like(x_coords)
        self.assertTrue(np.array_equal(np.array([np.nan, 1, 1, 1, 1]),
                                       pixel_utils.calculate_euclidean_distances(x_coords, y_coords2),
                                       equal_nan=True))
        y_coords3 = y_coords1.copy().astype(float)
        y_coords3[2] = np.nan
        self.assertTrue(np.array_equal(np.array([np.nan, sqrt2, np.nan, np.nan, sqrt2]),
                                       pixel_utils.calculate_euclidean_distances(x_coords, y_coords3),
                                       equal_nan=True))
        y_coords4 = y_coords1[:-1].copy()
        self.assertRaises(AssertionError, pixel_utils.calculate_euclidean_distances, x_coords, y_coords4)

    def test_calculate_velocities(self):
        sqrt2 = np.sqrt(2)
        x_coords = np.arange(0, 5)
        y_coords1 = np.arange(0, 5)
        t_coords1 = np.arange(5)
        self.assertTrue(np.array_equal(np.array([np.nan, sqrt2, sqrt2, sqrt2, sqrt2]),
                                       pixel_utils.calculate_velocities(x_coords, y_coords1, t_coords1),
                                       equal_nan=True))
        y_coords2 = np.zeros_like(x_coords)
        self.assertTrue(np.array_equal(np.array([np.nan, 1, 1, 1, 1]),
                                       pixel_utils.calculate_velocities(x_coords, y_coords2, t_coords1),
                                       equal_nan=True))
        t_coords2 = t_coords1 * 2
        self.assertTrue(np.array_equal(np.array([np.nan, 0.5, 0.5, 0.5, 0.5]),
                                       pixel_utils.calculate_velocities(x_coords, y_coords2, t_coords2),
                                       equal_nan=True))
        y_coords3 = y_coords1.copy().astype(float)
        y_coords3[2] = np.nan
        self.assertTrue(np.array_equal(np.array([np.nan, sqrt2, np.nan, np.nan, sqrt2]),
                                       pixel_utils.calculate_velocities(x_coords, y_coords3, t_coords1),
                                       equal_nan=True))
        t_coords3 = t_coords1[:-1].copy()
        self.assertRaises(AssertionError, pixel_utils.calculate_velocities, x_coords, y_coords1, t_coords3)

    def test_calculate_azimuth(self):
        # angles are counter-clockwise from the positive x-axis, with y-axis pointing down
        self.assertEqual(0, pixel_utils.calculate_azimuth(p1=(0, 0), p2=(0, 0), use_radians=False))
        self.assertEqual(45, pixel_utils.calculate_azimuth(p1=(0, 0), p2=(1, -1), use_radians=False))
        self.assertEqual(315, pixel_utils.calculate_azimuth(p1=(0, 0), p2=(1, 1), use_radians=False))
        self.assertEqual(90, pixel_utils.calculate_azimuth(p1=(1, 1), p2=(1, -1), use_radians=False))
        self.assertEqual(180, pixel_utils.calculate_azimuth(p1=(1, 1), p2=(1, -1),
                                                            use_radians=False, zero_direction='S'))

        self.assertEqual(np.pi * 3 / 4, pixel_utils.calculate_azimuth(p1=(1, 1), p2=(-2, -2), use_radians=True))
        self.assertEqual(np.pi, pixel_utils.calculate_azimuth(p1=(1, 0), p2=(-1, 0), use_radians=True))
        self.assertEqual(np.pi * 5 / 4, pixel_utils.calculate_azimuth(p1=(1, 0), p2=(0, 1), use_radians=True))
        self.assertEqual(np.pi * 3 / 4, pixel_utils.calculate_azimuth(p1=(1, 0), p2=(0, 1),
                                                                      use_radians=True, zero_direction='N'))

    def test_calculate_visual_angle(self):
        # implausible values
        d = 1
        ps = 1
        self.assertEqual(45, pixel_utils.calculate_visual_angle(p1=(0, 0), p2=(0, 1),
                                                                distance_from_screen=d,
                                                                pixel_size=ps,
                                                                use_radians=False))
        self.assertEqual(45, pixel_utils.calculate_visual_angle(p1=(0, 0), p2=(1, 0),
                                                                distance_from_screen=d,
                                                                pixel_size=ps,
                                                                use_radians=False))
        self.assertAlmostEqual(70.528779366, pixel_utils.calculate_visual_angle(p1=(0, 0), p2=(2, 2),
                                                                                distance_from_screen=d,
                                                                                pixel_size=ps,
                                                                                use_radians=False))

        # plausible values
        d = 65
        ps = 0.028069  # length of pixel's diagonal (in cm) for the Tobii Screen Monitor
        self.assertAlmostEqual(0.0247421, pixel_utils.calculate_visual_angle(p1=(0, 1), p2=(1, 1),
                                                                             distance_from_screen=d,
                                                                             pixel_size=ps,
                                                                             use_radians=False))
        self.assertAlmostEqual(0.0247421, pixel_utils.calculate_visual_angle(p1=(1, 2), p2=(1, 1),
                                                                             distance_from_screen=d,
                                                                             pixel_size=ps,
                                                                             use_radians=False))
        self.assertAlmostEqual(0.0349906, pixel_utils.calculate_visual_angle(p1=(0, 0), p2=(1, 1),
                                                                             distance_from_screen=d,
                                                                             pixel_size=ps,
                                                                             use_radians=False))

import unittest
import numpy as np

import Utils.visual_angle_utils as visang_utils


class TestVisualAngleUtils(unittest.TestCase):
    D = 1   # distance from screen to eye
    PS = 1  # pixel size

    def test_visual_angle_between_pixels(self):
        # implausible values
        d = 1
        ps = 1
        self.assertEqual(45, visang_utils.visual_angle_between_pixels(p1=(0, 0), p2=(0, 1), distance_from_screen=self.D,
                                                                      pixel_size=self.PS, use_radians=False))
        self.assertEqual(45, visang_utils.visual_angle_between_pixels(p1=(0, 0), p2=(1, 0), distance_from_screen=self.D,
                                                                      pixel_size=self.PS, use_radians=False))
        self.assertAlmostEqual(70.528779366, visang_utils.visual_angle_between_pixels(p1=(0, 0), p2=(2, 2),
                                                                                      distance_from_screen=self.D,
                                                                                      pixel_size=self.PS,
                                                                                      use_radians=False))

    def test_pixels_to_visual_angles(self):
        xs1 = np.zeros(5)
        ys = np.arange(5)
        self.assertTrue(np.array_equal(np.array([np.nan, 0, 0, 0, 0]),
                                       visang_utils.pixels_to_visual_angles(xs1, xs1, self.D, self.PS),
                                       equal_nan=True))
        self.assertTrue(np.array_equal(np.array([np.nan, 45, 45, 45, 45]),
                                       visang_utils.pixels_to_visual_angles(xs1, ys, self.D, self.PS),
                                       equal_nan=True))
        self.assertTrue(np.array_equal(np.array([np.nan, 45, 45, 45, 45]),
                                       visang_utils.pixels_to_visual_angles(xs1, -ys, self.D, self.PS),
                                       equal_nan=True))
        xs2 = np.arange(5)
        exp = np.arctan(np.sqrt(2))
        self.assertTrue(np.array_equal(np.array([np.nan, exp, exp, exp, exp]),
                                       visang_utils.pixels_to_visual_angles(xs2, ys, self.D, self.PS, use_radians=True),
                                       equal_nan=True))
        xs3 = xs1.copy()
        xs3[2] = np.nan
        self.assertTrue(np.array_equal(np.array([np.nan, 45, np.nan, np.nan, 45]),
                                       visang_utils.pixels_to_visual_angles(xs3, ys, self.D, self.PS),
                                       equal_nan=True))
        xs4 = xs1[:-1].copy()
        self.assertRaises(AssertionError, visang_utils.pixels_to_visual_angles, xs4, ys, self.D, self.PS)

    def test_pixels_to_angular_velocities(self):
        xs = np.zeros(5)
        ys = np.arange(5)
        ts = np.arange(5)
        self.assertTrue(np.array_equal(np.array([np.nan, 0, 0, 0, 0]),
                                       visang_utils.pixels_to_angular_velocities(xs, xs, ts, self.D, self.PS),
                                       equal_nan=True))
        self.assertTrue(np.array_equal(np.array([np.nan, 45, 45, 45, 45]),
                                       visang_utils.pixels_to_angular_velocities(xs, ys, ts, self.D, self.PS),
                                       equal_nan=True))
        self.assertTrue(np.array_equal(np.array([np.nan, 45, 45, 45, 45]),
                                       visang_utils.pixels_to_angular_velocities(xs, -ys, ts, self.D, self.PS),
                                       equal_nan=True))
        print(visang_utils.pixels_to_angular_velocities(xs, ys, ts * 2, self.D, self.PS))
        self.assertTrue(np.array_equal(np.array([np.nan, 22.5, 22.5, 22.5, 22.5]),
                                       visang_utils.pixels_to_angular_velocities(xs, ys, ts * 2, self.D, self.PS),
                                       equal_nan=True))

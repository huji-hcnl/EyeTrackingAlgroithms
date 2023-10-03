import unittest
import numpy as np

import constants as cnst
import Utils.visual_angle_utils as visang_utils


class TestVisualAngleUtils(unittest.TestCase):
    D = 1   # distance from screen to eye
    PS = 1  # pixel size

    def test_visual_angle_between_pixels(self):
        # implausible values
        self.assertEqual(45, visang_utils.visual_angle_between_pixels(p1=(0, 0), p2=(0, 1), distance_from_screen=self.D,
                                                                      pixel_size=self.PS, use_radians=False))
        self.assertEqual(45, visang_utils.visual_angle_between_pixels(p1=(0, 0), p2=(1, 0), distance_from_screen=self.D,
                                                                      pixel_size=self.PS, use_radians=False))
        self.assertAlmostEqual(70.528779366, visang_utils.visual_angle_between_pixels(p1=(0, 0), p2=(2, 2),
                                                                                      distance_from_screen=self.D,
                                                                                      pixel_size=self.PS,
                                                                                      use_radians=False))

    def test_pixels_to_visual_angle(self):
        self.assertEqual(0, visang_utils.pixels_to_visual_angle(num_px=0, d=self.D, pixel_size=self.PS))
        self.assertEqual(45, visang_utils.pixels_to_visual_angle(num_px=1, d=self.D, pixel_size=self.PS))
        self.assertEqual(np.pi / 4, visang_utils.pixels_to_visual_angle(num_px=1, d=self.D, pixel_size=self.PS,
                                                                        use_radians=True))
        self.assertTrue(np.isnan(visang_utils.pixels_to_visual_angle(num_px=np.inf, d=self.D, pixel_size=self.PS)))
        self.assertRaises(ValueError, visang_utils.pixels_to_visual_angle, num_px=-1, d=self.D, pixel_size=self.PS)
        self.assertRaises(ValueError, visang_utils.pixels_to_visual_angle, num_px=1, d=-1, pixel_size=self.PS)
        self.assertRaises(ValueError, visang_utils.pixels_to_visual_angle, num_px=1, d=self.D, pixel_size=-1)

    def test_pixels_array_to_vis_angle_array(self):
        xs1 = np.zeros(5)
        ys = np.arange(5)
        self.assertTrue(np.array_equal(np.array([0, 0, 0, 0, 0]),
                                       visang_utils.pixels_array_to_vis_angle_array(xs1, xs1, self.D, self.PS),
                                       equal_nan=True))
        self.assertTrue(np.array_equal(np.array([0, 45, 45, 45, 45]),
                                       visang_utils.pixels_array_to_vis_angle_array(xs1, ys, self.D, self.PS),
                                       equal_nan=True))
        self.assertTrue(np.array_equal(np.array([0, 45, 45, 45, 45]),
                                       visang_utils.pixels_array_to_vis_angle_array(xs1, -ys, self.D, self.PS),
                                       equal_nan=True))
        xs2 = np.arange(5)
        exp = np.arctan(np.sqrt(2))
        self.assertTrue(np.array_equal(np.array([0, exp, exp, exp, exp]),
                                       visang_utils.pixels_array_to_vis_angle_array(xs2, ys, self.D, self.PS,
                                                                                    use_radians=True),
                                       equal_nan=True))
        xs3 = xs1.copy()
        xs3[2] = np.nan
        self.assertTrue(np.array_equal(np.array([0, 45, np.nan, np.nan, 45]),
                                       visang_utils.pixels_array_to_vis_angle_array(xs3, ys, self.D, self.PS),
                                       equal_nan=True))
        xs4 = xs1[:-1].copy()
        self.assertRaises(AssertionError, visang_utils.pixels_array_to_vis_angle_array, xs4, ys, self.D, self.PS)

    def test_pixels_array_to_vis_angle_velocity_array(self):
        xs = np.zeros(5)
        ys = np.arange(5)
        ts = np.arange(5)
        self.assertTrue(np.array_equal(np.array([np.nan, 0, 0, 0, 0]),
                                       visang_utils.pixels_array_to_vis_angle_velocity_array(xs, xs, ts, self.D,
                                                                                             self.PS),
                                       equal_nan=True))
        self.assertTrue(np.array_equal(np.array([np.nan, 45, 45, 45, 45]) * cnst.MILLISECONDS_PER_SECOND,
                                       visang_utils.pixels_array_to_vis_angle_velocity_array(xs, ys, ts, self.D,
                                                                                             self.PS),
                                       equal_nan=True))
        self.assertTrue(np.array_equal(np.array([np.nan, 45, 45, 45, 45]) * cnst.MILLISECONDS_PER_SECOND,
                                       visang_utils.pixels_array_to_vis_angle_velocity_array(xs, -ys, ts, self.D,
                                                                                             self.PS),
                                       equal_nan=True))
        self.assertTrue(np.array_equal(np.array([np.nan, 22.5, 22.5, 22.5, 22.5]) * cnst.MILLISECONDS_PER_SECOND,
                                       visang_utils.pixels_array_to_vis_angle_velocity_array(xs, ys, ts * 2, self.D,
                                                                                             self.PS),
                                       equal_nan=True))

    def test_visual_angle_to_pixels(self):
        self.assertTrue(np.isnan(visang_utils.visual_angle_to_pixels(d=self.D, deg=np.inf, pixel_size=self.PS)))
        self.assertEqual(0, visang_utils.visual_angle_to_pixels(d=self.D, deg=0, pixel_size=self.PS))
        self.assertAlmostEqual(2.0, visang_utils.visual_angle_to_pixels(d=self.D, deg=90, pixel_size=self.PS))
        self.assertAlmostEqual(2.0, visang_utils.visual_angle_to_pixels(d=self.D, deg=-90, pixel_size=self.PS,
                                                                        keep_sign=False))
        self.assertAlmostEqual(-2.0, visang_utils.visual_angle_to_pixels(d=self.D, deg=-90, pixel_size=self.PS,
                                                                        keep_sign=True))
        self.assertAlmostEqual(0.0261814,
                               visang_utils.visual_angle_to_pixels(d=self.D, deg=1.5, pixel_size=self.PS))


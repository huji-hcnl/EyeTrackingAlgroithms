import unittest
from Detectors.NHDetector import NHDetector
from test_base_detector import TestBaseDetector


class TestNHDetector(TestBaseDetector):
    def setUp(self) -> None:
        super().setUp()

    def test_init(self):
        with self.assertRaises(ValueError):
            NHDetector(sr=-1 * self.sr, pixel_size=self.pixel_size, view_dist=self.view_dist,
                       timestamps=self.timestamps)

    def test_algorithm(self):
        detect_obj = NHDetector(sr=self.sr, pixel_size=self.pixel_size, view_dist=self.view_dist,
                                timestamps=self.timestamps)
        candidates = detect_obj.detect_candidates_monocular(self.timestamps, self.x_coords, self.y_coords)

        self.evaluate_performance(candidates)


if __name__ == '__main__':
    unittest.main()

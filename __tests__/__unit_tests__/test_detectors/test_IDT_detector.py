import unittest
from Detectors.IDTDetector import IDTDetector
from test_base_detector import TestBaseDetector


class TestIDTDetector(TestBaseDetector):
    def setUp(self) -> None:
        super().setUp()

    def test_init(self):
        with self.assertRaises(ValueError):
            IDTDetector(viewer_distance=self.view_dist, pixel_size=self.pixel_size)

    def test_algorithm(self):
        detect_obj = IDTDetector(viewer_distance=self.view_dist, pixel_size=self.pixel_size)
        candidates = detect_obj.detect_candidates_monocular(self.timestamps, self.x_coords, self.y_coords)

        self.evaluate_performance(candidates)


if __name__ == '__main__':
    unittest.main()

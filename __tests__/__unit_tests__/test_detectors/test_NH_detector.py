import unittest
from Detectors.NHDetector import NHDetector
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import constants as cnst
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score
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
        # Implement NH-specific test logic
        candidates = detect_obj.detect_candidates_monocular(self.timestamps, self.x_coords, self.y_coords)
        confusion_mat = confusion_matrix(self.labels, candidates)
        row_labels = ["undefined_gt", "fixation_gt", "saccade_gt", "pso_gt", "smooth_pursuit_gt", "blink_gt"]
        column_labels = ["undefined_alg", "fixation_alg", "saccade_alg", "pso_alg", "smooth_pursuit_alg", "blink_alg"]

        confusion_df = pd.DataFrame(confusion_mat, index=row_labels, columns=column_labels)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(confusion_df)

        # calculate rmsd for each type of stimulus
        self.data['candidates'] = candidates
        # # Group by the DataFrame by the "stimulus" column
        grouped = self.data.groupby([cnst.EVENT_TYPE, 'stimulus'])
        mse = grouped.apply(lambda x: mean_squared_error(x['candidates'], x[cnst.EVENT_TYPE]))
        print(mse)


if __name__ == '__main__':
    unittest.main()

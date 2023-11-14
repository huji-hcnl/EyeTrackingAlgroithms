import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score

from Detectors.EngbertDetector import EngbertDetector
from Config.GazeEventTypeEnum import GazeEventTypeEnum
from DataSetLoaders.Lund2013DataSetLoader import Lund2013DataSetLoader
import constants as cnst


class TestBaseDetector(unittest.TestCase):
    _SR = 500
    _LAMBDA = 5
    _WS = 2

    DETECTOR = EngbertDetector(lambda_noise_threshold=_LAMBDA, derivation_window_size=_WS)
    DETECTOR._sr = _SR

    def setUp(self) -> None:
        # data = LoadAnderssonData.load_from_url()
        self.data = Lund2013DataSetLoader.download()

        self.pixel_size = self.data["pixel_size_cm"][0]
        self.view_dist = self.data["viewer_distance_cm"][0]
        self.labels = np.array(self.data[cnst.EVENT_TYPE])
        self.x_coords = np.array(self.data[cnst.RIGHT_X])
        self.y_coords = np.array(self.data[cnst.RIGHT_Y])
        self.timestamps = np.array(self.data[cnst.MILLISECONDS])
        self.sr = TestBaseDetector._SR

    def test_algorithm(self):
        # Implement for each algorithm individually
        pass

    def evaluate_performance(self, candidates):
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

        grouped_2 = self.data.groupby(cnst.EVENT_TYPE)
        accuracy = grouped_2.apply(lambda x: accuracy_score(x['candidates'], x[cnst.EVENT_TYPE]))
        print("accuracy: ", accuracy)

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

import unittest
from Config.GazeEventTypeEnum import GazeEventTypeEnum
from Detectors.NHDetector import NHDetector
import numpy as np
import LoadAnderssonData
from Detectors.IDTDetector import IDTDetector
from Detectors.IVTDetector import IVTDetector
from sklearn.metrics import confusion_matrix
import pandas as pd
from DataSetLoaders.Lund2013DataSetLoader import Lund2013DataSetLoader
import constants as cnst
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score




class TestNHDetector(unittest.TestCase):

    def test_NH(self):
        enum_dict = {GazeEventTypeEnum.FIXATION: 1, GazeEventTypeEnum.SACCADE: 2, GazeEventTypeEnum.PSO: 3,
                     GazeEventTypeEnum.SMOOTH_PURSUIT: 4, GazeEventTypeEnum.BLINK: 5, GazeEventTypeEnum.UNDEFINED: 0}

        # data = LoadAnderssonData.load_from_url()
        data = Lund2013DataSetLoader.download()

        pixel_size = 0.037824
        labels = np.array(data[cnst.EVENT_TYPE])
        x_coords = np.array(data[cnst.RIGHT_X])
        y_coords = np.array(data[cnst.RIGHT_Y])
        timestamps = np.array(data[cnst.MILLISECONDS])
        detect_obj = NHDetector(sr=500, pixel_size=pixel_size, view_dist=0.6700, timestamps=list(timestamps))
        candidates = detect_obj.detect_candidates_monocular(timestamps, x_coords, y_coords)

        int_labels = [enum_dict[label] for label in labels]
        int_candidates = [enum_dict[candidate] for candidate in candidates]

        confusion_mat = confusion_matrix(int_labels, int_candidates)
        row_labels = ["undefined_gt", "fixation_gt", "saccade_gt", "pso_gt", "smooth_pursuit_gt", "blink_gt"]
        column_labels = ["undefined_alg", "fixation_alg", "saccade_alg", "pso_alg", "smooth_pursuit_alg", "blink_alg"]

        confusion_df = pd.DataFrame(confusion_mat, index=row_labels, columns=column_labels)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        print(confusion_df)

        # calculate rmsd for each type of stimulus
        data['candidates'] = candidates

        # Group by the DataFrame by the "stimulus" column
        grouped = data.groupby([cnst.EVENT_TYPE, 'stimulus'])
        # grouped = data.groupby(cnst.EVENT_TYPE)
        mse = grouped.apply(lambda x: mean_squared_error(x['candidates'], x[cnst.EVENT_TYPE]))
        print(mse)




    # def test_idt(self):
    #     data = LoadAnderssonData.load_from_url()
    #     pixel_size = 0.037824
    #     # part_data = data[0:8304]
    #     candidates = [GazeEventTypeEnum.UNDEFINED] * len(data["label"])
    #     labels = np.array(data["label"])
    #     x_coords = np.array(data["right_eye_x"])
    #     y_coords = np.array(data["right_eye_y"])
    #     detect_obj = IDTDetector(pixel_size=pixel_size, viewer_distance=0.6700, sr=500)
    #     candidates = detect_obj._identify_gaze_event_candidates(np.array(x_coords), np.array(y_coords),
    #                                                             np.array(candidates))
    #     enum_dict = {GazeEventTypeEnum.FIXATION: 1, GazeEventTypeEnum.SACCADE: 2, GazeEventTypeEnum.PSO: 3,
    #                  GazeEventTypeEnum.SMOOTH_PURSUIT: 4, GazeEventTypeEnum.BLINK: 5, GazeEventTypeEnum.UNDEFINED: 0}
    #     int_labels = [enum_dict[label] for label in labels]
    #     int_candidates = [enum_dict[candidate] for candidate in candidates]
    #     confusion_mat = confusion_matrix(int_labels, int_candidates, labels=[1, 2])
    #     print(confusion_mat)
    #
    # def test_ivt(self):
    #     data = LoadAnderssonData.load_from_url()
    #     candidates = [GazeEventTypeEnum.UNDEFINED] * len(data["label"])
    #     labels = np.array(data["label"])
    #     x_coords = np.array(data["right_eye_x"])
    #     y_coords = np.array(data["right_eye_y"])
    #     detect_obj = IVTDetector()
    #     candidates = detect_obj._identify_gaze_event_candidates(np.array(x_coords), np.array(y_coords),
    #                                                             np.array(candidates))
    #     enum_dict = {GazeEventTypeEnum.FIXATION: 1, GazeEventTypeEnum.SACCADE: 2, GazeEventTypeEnum.PSO: 3,
    #                  GazeEventTypeEnum.SMOOTH_PURSUIT: 4, GazeEventTypeEnum.BLINK: 5,
    #                  GazeEventTypeEnum.UNDEFINED: 0}
    #     int_labels = [enum_dict[label] for label in labels]
    #     int_candidates = [enum_dict[candidate] for candidate in candidates]
    #     confusion_mat = confusion_matrix(int_labels, int_candidates, labels=[1, 2])
    #     print(confusion_mat)

if __name__ == '__main__':
    unittest.main()

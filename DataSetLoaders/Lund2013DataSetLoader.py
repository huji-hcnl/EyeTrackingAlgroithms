import os
import io
import zipfile as zp
import numpy as np
import pandas as pd
import requests as req
from scipy.io import loadmat
from typing import Tuple, Dict

import constants as cnst
from DataSetLoaders.BaseDataSetLoader import BaseDataSetLoader
from Config.ScreenMonitor import ScreenMonitor
from Config.GazeEventTypeEnum import get_event_type


class Lund2013DataSetLoader(BaseDataSetLoader):
    """
    Loads the dataset presented in the article:
    Andersson, R., Larsson, L., Holmqvist, K., Stridh, M., & Nyström, M. (2017): One algorithm to rule them all? An
    evaluation and discussion of ten eye movement event-detection algorithms. Behavior Research Methods, 49(2), 616-637.

    Note that there was an error in the original dataset, which was corrected in a later article:
    Zemblys, R., Niehorster, D.C. & Holmqvist, K. gazeNet: End-to-end eye-movement event detection with deep neural
    networks. Behav Res 51, 840–864 (2019).

    This loader is based on a previous implementation, see article:
    Startsev, M., Zemblys, R. Evaluating Eye Movement Event Detection: A Review of the State of the Art. Behav Res 55, 1653–1714 (2023)
    See their implementation: https://github.com/r-zemblys/EM-event-detection-evaluation/blob/main/misc/data_parsers/lund2013.py
    """

    _URL = 'https://github.com/richardandersson/EyeMovementDetectorEvaluation/archive/refs/heads/master.zip'

    _ARTICLES = [
        "Andersson, R., Larsson, L., Holmqvist, K., Stridh, M., & Nyström, M. (2017): One algorithm to rule them " +
        "all? An evaluation and discussion of ten eye movement event-detection algorithms. Behavior Research Methods, " +
        "49(2), 616-637.",

        "Zemblys, R., Niehorster, D. C., Komogortsev, O., & Holmqvist, K. (2018). Using machine learning to detect " +
        "events in eye-tracking data. Behavior Research Methods, 50(1), 160–181."
    ]

    __STIMULUS_NAME = f"{cnst.STIMULUS}_name"
    __PIXEL_SIZE_CM = "pixel_size_cm"
    __VIEWER_DISTANCE_CM = "viewer_distance_cm"

    @classmethod
    def column_order(cls) -> Dict[str, float]:
        order = BaseDataSetLoader.column_order()
        order.update({cls.__STIMULUS_NAME: 6.1, cls.__PIXEL_SIZE_CM: 6.2, cls.__VIEWER_DISTANCE_CM: 6.3})
        return order

    @classmethod
    def _parse_response(cls, response: req.Response) -> pd.DataFrame:
        zip_file = zp.ZipFile(io.BytesIO(response.content))

        # list all files in the zip archive that are relevant to this dataset
        # replaces erroneously labelled files with the corrected ones (see readme.md for more info)
        prefix = 'EyeMovementDetectorEvaluation-master/annotated_data/originally uploaded data/'
        erroneous_files = ['UH29_img_Europe_labelled_MN.mat']
        is_valid_file = lambda f: f.startswith(prefix) and f.endswith('.mat') and f not in erroneous_files
        file_names = [f for f in zip_file.namelist() if is_valid_file(f)]
        file_names.append('EyeMovementDetectorEvaluation-master/annotated_data/fix_by_Zemblys2018/UH29_img_Europe_labelled_FIX_MN.mat')

        # read all files into a list of dataframes
        dataframes = {}
        for f in file_names:
            file = zip_file.open(f)
            gaze_data = Lund2013DataSetLoader.__read_gaze_data(file)
            subject_id, stimulus_type, stimulus_name, rater = Lund2013DataSetLoader.__extract_metadata(file)
            stimulus_name = stimulus_name.removesuffix("_labelled")
            gaze_data.rename(columns={cnst.EVENT_TYPE: rater}, inplace=True)

            # write the DF to a dict based on the subject id, stimulus type, stimulus name, or add to existing DF
            existing_df = dataframes.get((subject_id, stimulus_type, stimulus_name), None)
            if existing_df is None:
                gaze_data[cnst.SUBJECT_ID] = subject_id
                gaze_data[cnst.STIMULUS] = stimulus_type
                gaze_data[cls.__STIMULUS_NAME] = stimulus_name
                dataframes[(subject_id, stimulus_type, stimulus_name)] = gaze_data
            else:
                existing_df[rater] = gaze_data[rater]
                dataframes[(subject_id, stimulus_type, stimulus_name)] = existing_df
        merged_df = pd.concat(dataframes.values(), ignore_index=True, axis=0)
        return merged_df

    @classmethod
    def _clean_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        # replace missing samples with NaNs:
        # this dataset marks missing samples with (0, 0) coordinates, instead of NaNs.
        x_missing = df[cnst.RIGHT_X] == 0
        y_missing = df[cnst.RIGHT_Y] == 0
        missing_idxs = np.where(x_missing & y_missing)[0]
        col_idxs = df.columns.get_indexer([cnst.RIGHT_X, cnst.RIGHT_Y])
        df.iloc[missing_idxs, col_idxs] = np.nan

        # add a column for trial number:
        # trials are instances that share the same subject id, stimulus type and stimulus name.
        trial_counter = 1
        df[cnst.TRIAL] = np.nan
        for _, trial_df in df.groupby([cnst.SUBJECT_ID, cnst.STIMULUS, cls.__STIMULUS_NAME]):
            df.loc[trial_df.index, cnst.TRIAL] = trial_counter
            trial_counter += 1
        df[cnst.TRIAL] = df[cnst.TRIAL].astype(int)
        return df

    @staticmethod
    def __read_gaze_data(file) -> pd.DataFrame:
        mat = loadmat(file)
        eyetracking_data = mat["ETdata"]
        eyetracking_data_dict = {name: eyetracking_data[name][0, 0] for name in eyetracking_data.dtype.names}

        # extract singleton values and convert from meters to cm:
        view_dist = eyetracking_data_dict['viewDist'][0, 0] * 100
        screen_width, screen_height = eyetracking_data_dict['screenDim'][0] * 100
        screen_res = eyetracking_data_dict['screenRes'][0]  # (1024, 768)
        pixel_size = ScreenMonitor.calculate_pixel_size(screen_width, screen_height, screen_res)

        # extract gaze data:
        sampling_rate = eyetracking_data_dict['sampFreq'][0, 0]
        samples_data = eyetracking_data_dict['pos']
        right_x, right_y = samples_data[:, 3:5].T  # only recording right eye
        timestamps = Lund2013DataSetLoader.__calculate_timestamps(samples_data[:, 0], sampling_rate)
        labels = [get_event_type(int(event_type), safe=True) for event_type in samples_data[:, 5]]

        # create dataframe:
        df = pd.DataFrame(data={cnst.MILLISECONDS: timestamps,
                                cnst.RIGHT_X: right_x, cnst.RIGHT_Y: right_y,
                                cnst.EVENT_TYPE: labels})
        df[Lund2013DataSetLoader.__VIEWER_DISTANCE_CM] = view_dist
        df[Lund2013DataSetLoader.__PIXEL_SIZE_CM] = pixel_size
        return df

    @staticmethod
    def __calculate_timestamps(timestamps, sampling_rate):
        """
        Returns an arrays of timestamps per each sample, in milliseconds.
        The original timestamps of the Anderson dataset are either an array of microseconds or an array of NaNs.
        """
        if np.isnan(timestamps).any():
            num_samples = len(timestamps)
            timestamps = np.arange(num_samples) * cnst.MILLISECONDS_PER_SECOND / sampling_rate
            return timestamps
        timestamps = timestamps - np.nanmin(timestamps)
        timestamps = timestamps / cnst.MICROSECONDS_PER_MILLISECOND
        return timestamps

    @staticmethod
    def __extract_metadata(file) -> Tuple[str, str, str, str]:
        file_name = os.path.basename(file.name)  # remove path
        if not file_name.endswith(".mat"):
            raise ValueError(f"Expected a `.mat` file, got: {file_name}")

        # file_name fmt: `<subject_id>_<stimulus_type>_<stimulus_name_1>_ ... _<stimulus_name_N>_labelled_<rater_name>`
        # moving-dot trials don't contain stimulus names
        file_name = file_name.replace(".mat", "")  # remove extension
        split_name = file_name.split("_")
        subject_id = split_name[0]  # subject id is always 1st in the file name
        stimulus_type = split_name[1]  # stimulus type is always 2nd in the file name
        rater = split_name[-1]  # rater is always last in the file name
        stimulus_name = "_".join(split_name[2:-2])  # stimulus name is everything in between stimulus type and rater
        if stimulus_type.startswith("trial"):
            stimulus_type = "moving dot"  # moving-dot stimulus is labelled as "trial1", "trial2", etc.
        return subject_id, stimulus_type, stimulus_name, rater

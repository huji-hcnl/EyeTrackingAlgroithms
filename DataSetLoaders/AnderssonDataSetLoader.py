import os
import numpy as np
import pandas as pd
import requests as req
from scipy.io import loadmat
from abc import ABC
from typing import List, Tuple

import constants as cnst
from DataSetLoaders.BaseDataSetLoader import BaseDataSetLoader


class AnderssonDataSetLoader(BaseDataSetLoader, ABC):
    """
    Loads the dataset presented in the article:
    Andersson, R., Larsson, L., Holmqvist, K., Stridh, M., & Nyström, M. (2017): One algorithm to rule them all? An
    evaluation and discussion of ten eye movement event-detection algorithms. Behavior Research Methods, 49(2), 616-637.

    Note that there was an error in the original dataset, which was corrected in a later article:
    Zemblys, R., Niehorster, D. C., Komogortsev, O., & Holmqvist, K. (2018). Using machine learning to detect events in
    eye-tracking data. Behavior Research Methods, 50(1), 160–181.
    See their implementation: https://github.com/r-zemblys/EM-event-detection-evaluation/blob/main/misc/data_parsers/lund2013.py
    """

    _URL: str = "http://www.kasprowski.pl/datasets/events.zip"
    _ARTICLE: str = "https://link.springer.com/article/10.3758/s13428-016-0738-9"

    __STIMULUS_NAME = f"{cnst.STIMULUS}_name"
    __RATER = "rater"
    __PIXEL_SIZE_CM = "pixel_size_cm"
    __VIEWER_DISTANCE_CM = "viewer_distance_cm"

    @classmethod
    def columns(cls) -> List[str]:
        return [cnst.SUBJECT_ID, cls.__VIEWER_DISTANCE_CM, cnst.STIMULUS, cls.__STIMULUS_NAME, cls.__PIXEL_SIZE_CM,
                cls.__RATER, cnst.TRIAL, cnst.MILLISECONDS, cnst.RIGHT_X, cnst.RIGHT_Y, cnst.EVENT_TYPE]

    @classmethod
    def _parse_response(cls, response: req.Response) -> pd.DataFrame:
        import io
        import zipfile
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))

        dataframes = []
        for filename in zip_file.namelist():
            if not filename.endswith(".mat"):
                continue
            mat_file = zip_file.open(filename)
            df = cls.__read_mat_file(mat_file)
            dataframes.append(df)
        # create a unified dataframe:
        df = pd.concat(dataframes, ignore_index=True, axis=0)

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

    @classmethod
    def __read_mat_file(cls, mat_file) -> pd.DataFrame:
        gaze_data = cls.__handle_mat_file_data(mat_file)
        subject_id, stimulus_type, stimulus_name, rater = cls.__handle_mat_file_name(mat_file.name)
        gaze_data[cnst.SUBJECT_ID] = subject_id
        gaze_data[cnst.STIMULUS] = stimulus_type
        gaze_data[cls.__STIMULUS_NAME] = stimulus_name
        gaze_data[cls.__RATER] = rater
        return gaze_data

    @staticmethod
    def __handle_mat_file_name(file_name: str) -> Tuple[str, str, str, str]:
        if not file_name.endswith(".mat"):
            raise ValueError(f"Expected a `.mat` file, got: {file_name}")

        file_name = os.path.basename(file_name)  # remove path
        file_name = file_name.replace(".mat", "")  # remove extension
        # file_name: `<subject_id>_<stimulus_type>_<stimulus_name_1>_ ... _<stimulus_name_N>_labelled_<rater_name>`
        # moving-dot trials for not contain stimulus names
        split_name = file_name.split("_")
        subject_id = split_name[0]                  # subject id is always 1st in the file name
        stimulus_type = split_name[1]               # stimulus type is always 2nd in the file name
        rater = split_name[-1]                      # rater is always last in the file name
        stimulus_name = "_".join(split_name[2:-2])  # stimulus name is everything in between stimulus type and rater
        if stimulus_type.startswith("trial"):
            stimulus_type = "moving dot"            # moving-dot stimulus is labelled as "trial1", "trial2", etc.
        return subject_id, stimulus_type, stimulus_name, rater

    @staticmethod
    def __handle_mat_file_data(mat_file) -> pd.DataFrame:
        mat = loadmat(mat_file)
        eyetracking_data = mat["ETdata"]
        eyetracking_data_dict = {name: eyetracking_data[name][0, 0] for name in eyetracking_data.dtype.names}

        # extract singleton values and convert from meters to cm:
        from Config.ScreenMonitor import ScreenMonitor
        view_dist = eyetracking_data_dict['viewDist'][0, 0] * 100
        screen_width, screen_height = eyetracking_data_dict['screenDim'][0] * 100
        screen_res = eyetracking_data_dict['screenRes'][0]  # (1024, 768)
        pixel_size = ScreenMonitor.calculate_pixel_size(screen_width, screen_height, screen_res)
        sampling_rate = eyetracking_data_dict['sampFreq'][0, 0]

        # extract gaze data:
        from Config.GazeEventTypeEnum import get_event_type
        samples_data = eyetracking_data_dict['pos']
        right_x, right_y = samples_data[:, 3:5].T
        timestamps = AnderssonDataSetLoader.__calculate_timestamps(samples_data[:, 0], sampling_rate)
        labels = [get_event_type(int(event_type), safe=True) for event_type in samples_data[:, 5]]

        # create dataframe:
        df = pd.DataFrame(data={cnst.MILLISECONDS: timestamps,
                                cnst.RIGHT_X: right_x, cnst.RIGHT_Y: right_y,
                                cnst.EVENT_TYPE: labels})
        df[AnderssonDataSetLoader.__VIEWER_DISTANCE_CM] = view_dist
        df[AnderssonDataSetLoader.__PIXEL_SIZE_CM] = pixel_size
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

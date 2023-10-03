import io
import zipfile as zp
import posixpath as psx
import json
import numpy as np
import pandas as pd
import requests as req
import itertools
from scipy.interpolate import interp1d
from typing import Tuple, Dict

import constants as cnst
import Utils.io_utils as ioutils
import Utils.visual_angle_utils as vis_utils
from DataSetLoaders.BaseDataSetLoader import BaseDataSetLoader
from Config.ScreenMonitor import ScreenMonitor
from Config.GazeEventTypeEnum import get_event_type


class IRFDataSetLoader(BaseDataSetLoader):
    """
    Loads the dataset from a replication study of the article:
    Using machine learning to detect events in eye-tracking data. Zemblys et al. (2018).
    See also about the repro study: https://github.com/r-zemblys/irf/blob/master/doc/IRF_replication_report.pdf

    Note: binocular data was recorded but only one pair of (x, y) coordinates is provided.
    For the sake of consistency, we will consider these right-eye coordinates.

    This loader is based on a previous implementation, see article:
    Startsev, M., Zemblys, R. Evaluating Eye Movement Event Detection: A Review of the State of the Art. Behav Res 55, 1653–1714 (2023)
    See their implementation: https://github.com/r-zemblys/EM-event-detection-evaluation/blob/main/misc/data_parsers/humanFixationClassification.py
    """

    _URL = r'https://github.com/r-zemblys/irf/archive/refs/heads/master.zip'

    _ARTICLES = [
        "Zemblys, Raimondas and Niehorster, Diederick C and Komogortsev, Oleg and Holmqvist, Kenneth. Using machine " +
        "learning to detect events in eye-tracking data. Behavior Research Methods, 50(1), 160–181 (2018)."
    ]

    __RATER_NAME = "RZ"
    __PIXEL_SIZE_CM = "pixel_size_cm"
    __VIEWER_DISTANCE_CM = "viewer_distance_cm"

    # Values used in the apparatus of the experiment.
    # see https://github.com/r-zemblys/irf/blob/master/etdata/lookAtPoint_EL/db_config.json
    __STIMULUS_VAL = "moving_dot"  # all subjects were shown the same 13-point moving dot stimulus
    __VIEWER_DISTANCE_CM_VAL = 56.5
    __SCREEN_MONITOR = ScreenMonitor(width=1920, height=1080, resolution=(1920, 1080), refresh_rate=60)

    @classmethod
    def column_order(cls) -> Dict[str, float]:
        order = BaseDataSetLoader.column_order()
        order.update({cls.__PIXEL_SIZE_CM: 6.2, cls.__VIEWER_DISTANCE_CM: 6.3})
        return order

    @classmethod
    def _parse_response(cls, response: req.Response) -> pd.DataFrame:
        zip_file = zp.ZipFile(io.BytesIO(response.content))

        # Get ET Data:
        prefix = 'irf-master/etdata/lookAtPoint_EL'
        gaze_file_names = [f for f in zip_file.namelist() if (f.startswith(psx.join(prefix, "lookAtPoint_EL_"))
                                                              and f.endswith('.npy'))]
        gaze_dfs = []
        for f in gaze_file_names:
            file = zip_file.open(f)
            gaze_data = pd.DataFrame(np.load(file))

            # convert gaze events from int to GazeEventTypeEnum
            gaze_data['evt'] = gaze_data['evt'].apply(lambda x: get_event_type(x))

            # extract subject id:
            _, file_name, _ = ioutils.split_path(f)
            subject_id = file_name.split('_')[-1]  # format: "lookAtPoint_EL_S<subject_num>"
            gaze_data[cnst.SUBJECT_ID] = subject_id
            gaze_dfs.append(gaze_data)
        merged_df = pd.concat(gaze_dfs, ignore_index=True, axis=0)

        # add meta data columns:
        merged_df[cnst.STIMULUS] = cls.__STIMULUS_VAL
        merged_df[cls.__VIEWER_DISTANCE_CM] = cls.__VIEWER_DISTANCE_CM_VAL
        merged_df[cls.__PIXEL_SIZE_CM] = cls.__SCREEN_MONITOR.pixel_size
        return merged_df

    @classmethod
    def _clean_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        # replace invalid samples with NaN:
        idxs_to_replace = df[~df['status']].index
        df.loc[idxs_to_replace, cnst.X] = np.nan
        df.loc[idxs_to_replace, cnst.Y] = np.nan

        # rename columns: replace `t` with `milliseconds`, `evt` with `rater_name`, and change the unambiguous `x` and
        # `y` with `right_x` and `right_y`. also, drop the `status` column that indicates whether the data is valid.
        df.rename(columns={"t": cnst.MILLISECONDS, "evt": cls.__RATER_NAME, "x": cnst.RIGHT_X, "y": cnst.RIGHT_Y},
                  inplace=True)
        df.drop(columns=["status"], inplace=True)

        # convert to milliseconds:
        df[cnst.MILLISECONDS] = df[cnst.MILLISECONDS] * 1000

        # convert x-y coordinates to pixels (use apparatus values):
        x, y = cls.__convert_coordinates(x=df[cnst.RIGHT_X], y=df[cnst.RIGHT_Y])
        df[cnst.RIGHT_X] = x
        df[cnst.RIGHT_Y] = y

        # add a column for trial number:
        # trials are instances that share the same subject id & stimulus.
        trial_counter = 1
        df[cnst.TRIAL] = np.nan
        for _, trial_df in df.groupby([cnst.SUBJECT_ID]):
            df.loc[trial_df.index, cnst.TRIAL] = trial_counter
            trial_counter += 1
        df[cnst.TRIAL] = df[cnst.TRIAL].astype(int)
        return df

    @classmethod
    def __convert_coordinates(cls, x: pd.Series, y: pd.Series) -> Tuple[pd.Series, pd.Series]:
        pixel_width = cls.__SCREEN_MONITOR.width / cls.__SCREEN_MONITOR.resolution[0]  # in cm
        x = x.apply(lambda deg: vis_utils.visual_angle_to_pixels(deg=deg, d=cls.__VIEWER_DISTANCE_CM_VAL,
                                                                 pixel_size=pixel_width, keep_sign=True))
        x += cls.__SCREEN_MONITOR.resolution[0] // 2  # move x=0 coordinate to the top of the screen

        pixel_height = cls.__SCREEN_MONITOR.height / cls.__SCREEN_MONITOR.resolution[1]  # in cm
        y = y.apply(lambda deg: vis_utils.visual_angle_to_pixels(deg=deg, d=cls.__VIEWER_DISTANCE_CM_VAL,
                                                                 pixel_size=pixel_height, keep_sign=True))
        y += cls.__SCREEN_MONITOR.resolution[1] // 2  # move y=0 coordinate to the top of the screen
        return x, y

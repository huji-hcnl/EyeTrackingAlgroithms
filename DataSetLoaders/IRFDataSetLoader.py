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
from DataSetLoaders.BaseDataSetLoader import BaseDataSetLoader
from Config.ScreenMonitor import ScreenMonitor
from Config.GazeEventTypeEnum import get_event_type


class IRFDataSetLoader(BaseDataSetLoader):
    """
    Loads the two datasets presented in articles:
    - (adults) Is human classification by experienced untrained observers a gold standard in fixation detection?
        Hooge et al. (2018)
    - (infants) An in-depth look at saccadic search in infancy.
        Hessels et al. (2016)

    This loader is based on a previous implementation, see article:
    Startsev, M., Zemblys, R. Evaluating Eye Movement Event Detection: A Review of the State of the Art. Behav Res 55, 1653–1714 (2023)
    See their implementation: https://github.com/r-zemblys/EM-event-detection-evaluation/blob/main/misc/data_parsers/humanFixationClassification.py
    """

    _URL = r'https://github.com/r-zemblys/irf/archive/refs/heads/master.zip'

    _ARTICLES = [
        "Zemblys, Raimondas and Niehorster, Diederick C and Komogortsev, Oleg and Holmqvist, Kenneth. Using machine " +
        "learning to detect events in eye-tracking data. Behavior Research Methods, 50(1), 160–181 (2018)."
    ]

    __PIXEL_SIZE_CM = "pixel_size_cm"
    __VIEWER_DISTANCE_CM = "viewer_distance_cm"
    __RATER_NAME = "RZ"

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

            # only use events 1, 2, 3 (fixations, saccades, PSOs)
            mask = gaze_data["evt"].isin([1, 2, 3])
            gaze_data.loc[~mask, "evt"] = 0
            gaze_data['evt'] = gaze_data['evt'].apply(lambda x: get_event_type(x))

            _, file_name, _ = ioutils.split_path(f)
            subject_id = file_name.split('_')[-1]  # format: "lookAtPoint_EL_S<subject_num>"
            gaze_data[cnst.SUBJECT_ID] = subject_id
            gaze_dfs.append(gaze_data)

        merged_df = pd.concat(gaze_dfs, ignore_index=True, axis=0)

        # add meta data columns:
        config_file = psx.join(prefix, "db_config.json")
        config = json.load(zip_file.open(config_file))['geom']
        stimulus = "moving_dot"  # all subjects were shown the same 13-point moving dot stimulus
        viewer_distance = config['eye_distance'] / 10  # convert to cm
        pixel_size = ScreenMonitor.calculate_pixel_size(width=config['screen_width'] / 10,
                                                        height=config['screen_height'] / 10,
                                                        resolution=(config['display_width_pix'],
                                                                    config['display_height_pix']))
        merged_df[cnst.STIMULUS] = stimulus
        merged_df[cls.__VIEWER_DISTANCE_CM] = viewer_distance
        merged_df[cls.__PIXEL_SIZE_CM] = pixel_size
        return merged_df

    @classmethod
    def _clean_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        # rename columns:
        df.rename(columns={"t": cnst.MILLISECONDS, "evt": cls.__RATER_NAME}, inplace=True)

        # convert to milliseconds:
        df[cnst.MILLISECONDS] = df[cnst.MILLISECONDS] * 1000

        # convert x-y coordinates to pixels:
        # TODO!

        # add a column for trial number:
        # trials are instances that share the same subject id & stimulus.
        trial_counter = 1
        df[cnst.TRIAL] = np.nan
        for _, trial_df in df.groupby([cnst.SUBJECT_ID]):
            df.loc[trial_df.index, cnst.TRIAL] = trial_counter
            trial_counter += 1
        df[cnst.TRIAL] = df[cnst.TRIAL].astype(int)
        return df

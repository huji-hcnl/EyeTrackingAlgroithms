import os
import io
import zipfile as zp
import posixpath as psx
import numpy as np
import pandas as pd
import requests as req
import itertools
from scipy.interpolate import interp1d
from typing import List, Tuple, Dict

import constants as cnst
import Utils.io_utils as ioutils
from DataSetLoaders.BaseDataSetLoader import BaseDataSetLoader


class HFCDataSetLoader(BaseDataSetLoader):
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

    _URL = r'https://github.com/dcnieho/humanFixationClassification/archive/refs/heads/master.zip'

    _ARTICLES = [
        "Hooge, I.T.C., Niehorster, D.C., Nyström, M., Andersson, R. & Hessels, R.S. (2018). Is human classification " +
        "by experienced untrained observers a gold standard in fixation detection?",
        "Hessels, R.S., Hooge, I.T.C., & Kemner, C. (2016). An in-depth look at saccadic search in infancy. " +
        "Journal of Vision, 16(8), 10."
    ]

    __SUBJECT_TYPE = "subject_type"

    @classmethod
    def _parse_response(cls, response: req.Response) -> pd.DataFrame:
        zip_file = zp.ZipFile(io.BytesIO(response.content))

        # Get ET Data:
        prefix = 'humanFixationClassification-master/data'
        gaze_file_names = [f for f in zip_file.namelist() if (f.startswith(psx.join(prefix, "ETdata"))
                                                              and f.endswith('.txt'))]
        gaze_dfs = {}
        for f in gaze_file_names:
            trial_name, gaze_data = cls.__read_gaze_data(zip_file.open(f))
            gaze_dfs[trial_name] = gaze_data

        # Get Annotations:
        coder_files = [f for f in zip_file.namelist() if (f.startswith(psx.join(prefix, "coderSettings"))
                                                          and f.endswith('.txt'))]
        annotation_dfs = {}
        for f in coder_files:
            rater_name, rater_data = cls.__read_annotations(zip_file.open(f))
            annotation_dfs[rater_name] = rater_data

        # merge annotations with ET data:
        merged_df = cls.__merge_gaze_data_with_annotations(gaze_dfs, annotation_dfs)
        return merged_df

    @classmethod
    def _clean_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        # rename columns:
        df.rename(columns={"time": cnst.MILLISECONDS, "x": cnst.LEFT_X, "y": cnst.LEFT_Y}, inplace=True)

        # add a column for trial number:
        # trials are instances that share the same subject id & stimulus.
        trial_counter = 1
        df[cnst.TRIAL] = np.nan
        for _, trial_df in df.groupby([cnst.SUBJECT_ID, cnst.STIMULUS]):
            df.loc[trial_df.index, cnst.TRIAL] = trial_counter
            trial_counter += 1
        df[cnst.TRIAL] = df[cnst.TRIAL].astype(int)
        return df

    @staticmethod
    def __read_gaze_data(file) -> Tuple[str, pd.DataFrame]:
        gaze_data = pd.read_csv(file, sep='\t', usecols=["time", "x", "y"])
        _, trial_name, _ = ioutils.split_path(file.name)
        subject_type, subject_id = trial_name.split('_')
        gaze_data[cnst.SUBJECT_ID] = subject_id
        gaze_data[HFCDataSetLoader.__SUBJECT_TYPE] = subject_type
        gaze_data[cnst.STIMULUS] = "free_viewing" if subject_type == "adult" else "search_task"
        return trial_name, gaze_data

    @staticmethod
    def __read_annotations(file) -> Tuple[str, pd.DataFrame]:
        rater_data = pd.read_csv(file, sep='\t')
        _, rater_name, _ = ioutils.split_path(file.name)
        return rater_name, rater_data

    @staticmethod
    def __merge_gaze_data_with_annotations(gaze_dfs: Dict[str, pd.DataFrame],
                                           annotation_dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        dataframes = []
        for trial_name in gaze_dfs.keys():
            data = gaze_dfs.get(trial_name, None)
            if data is None or len(data) == 0:
                continue
            l = len(data)
            for coder_name in annotation_dfs.keys():
                evnts = np.zeros(l, dtype=int)
                coder_annotations = annotation_dfs.get(coder_name).query("Trial==@trial_name")
                if len(coder_annotations):
                    # reached here if there are annotations for this trial
                    f = interp1d(data["time"], range(l), kind="nearest", bounds_error=False, fill_value="extrapolate")
                    fixation_samples = itertools.chain(
                        *[range(int(s), int(e + 1)) for s, e in zip(f(coder_annotations["FixStart"]),
                                                                    f(coder_annotations["FixEnd"]))])
                    evnts[list(fixation_samples)] = 1
                data[coder_name] = evnts
                dataframes.append(data)

        merged_df = pd.concat(dataframes, ignore_index=True, axis=0)
        return merged_df

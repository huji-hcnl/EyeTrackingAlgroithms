import os
import h5py as h5
import numpy as np
import pandas as pd
from typing import Set, Union, Optional

from DataParsers.BaseEyeTrackerParser import BaseEyeTrackerParser


class TobiiEyeTrackerHDF5Parser(BaseEyeTrackerParser):
    """
    Parses eye-tracking data based on the HDF5 format exported by Tobii eye-tracker and PsychoPy.
    See additional information here: https://psychopy.org/hardware/eyeTracking.html#what-about-the-data
    """

    def parse(self, input_path: str, *args) -> pd.DataFrame:
        raise NotImplementedError

    @classmethod
    def _read_raw_data(cls, input_path: str):
        """
        Reads the raw HDF5 file exported by Tobii eye-tracker and returns a pandas DataFrame.
        See aditional resources:
        - file structure (hdf5): https://psychopy.org/hardware/eyeTracking.html#what-about-the-data
        - data format: https://psychopy.org/api/iohub/device/eyetracker_interface/Tobii_Implementation_Notes.html
        - PsychoPy's IOHub code:
            * event constants: https://github.com/psychopy/versions/blob/master/psychopy/iohub/constants.py#L81
            * eye tracker constants: https://github.com/psychopy/versions/blob/master/psychopy/iohub/constants.py#L985
            * eye tracker binocular event: https://github.com/psychopy/versions/blob/master/psychopy/iohub/devices/eyetracker/eye_events.py#L330

        :param input_path: path to the HDF5 file
        :return: a DataFrame containing the raw data

        :raises FileNotFoundError: if the file does not exist

        """
        cls._raise_for_invalid_input_path(input_path)
        with h5.File(input_path, 'r') as f:
            dataset = f['data_collection']['events']['eyetracker']['BinocularEyeSampleEvent']
            colnames = dataset.dtype.names
            data_dict = {col: [] for col in colnames}
            for row in dataset:
                for i, col in enumerate(colnames):
                    data_dict[col].append(row[i])
        return pd.DataFrame(data_dict)

    @classmethod
    def FILE_EXTENSION(cls) -> str:
        # file extension of raw data files
        return '.hdf5'

    @classmethod
    def MISSING_VALUES(cls) -> Set[Union[int, float, str]]:
        return {np.nan, None}

    @classmethod
    def TRIAL_COLUMN(cls) -> Optional[str]:
        # column name for trial number
        return None

    @classmethod
    def SECONDS_COLUMN(cls) -> Optional[str]:
        # column name for time in seconds
        return "time"

    @classmethod
    def MILLISECONDS_COLUMN(cls) -> Optional[str]:
        # column name for time in milliseconds
        return None

    @classmethod
    def MICROSECONDS_COLUMN(cls) -> Optional[str]:
        # column name for time in microseconds
        return None

    @classmethod
    def LEFT_X_COLUMN(cls) -> Optional[str]:
        # column name for left eye x coordinate
        return "left_gaze_x"

    @classmethod
    def LEFT_Y_COLUMN(cls) -> Optional[str]:
        # column name for left eye y coordinate
        return "left_gaze_y"

    @classmethod
    def LEFT_PUPIL_COLUMN(cls) -> Optional[str]:
        # column name for left eye pupil diameter
        return "left_pupil_measure_1"

    @classmethod
    def RIGHT_X_COLUMN(cls) -> Optional[str]:
        # column name for right eye x coordinate
        return "right_gaze_x"

    @classmethod
    def RIGHT_Y_COLUMN(cls) -> Optional[str]:
        # column name for right eye y coordinate
        return "right_gaze_y"

    @classmethod
    def RIGHT_PUPIL_COLUMN(cls) -> Optional[str]:
        # column name for right eye pupil diameter
        return "right_pupil_measure_1"

    raise NotImplementedError







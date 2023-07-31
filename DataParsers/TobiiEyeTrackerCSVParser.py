import os
import pandas as pd
from typing import Set, Union, Tuple, Optional

from Config import experiment_config as cnfg
from DataParsers.BaseEyeTrackerParser import BaseEyeTrackerParser


class TobiiEyeTrackerCSVParser(BaseEyeTrackerParser):
    """
    Parses eye-tracking data based on the CSV format exported by Tobii eye-tracker and E-Prime.
    See information on the raw data format under "Tutorial 2 // Task 7" (page 56) in E-Prime's user manual for the Tobii
    eye-tracker (TET) package: https://pstnet.com/wp-content/uploads/2019/05/EET_User_Guide_3.2.pdf
    """

    def parse(self, input_path: str,
              screen_resolution: Tuple[float, float] = cnfg.SCREEN_MONITOR.resolution) -> pd.DataFrame:
        df = self._read_raw_data(input_path)
        columns_to_keep = self._get_common_columns() + self._additional_columns
        df.drop(columns=[col for col in df.columns if col not in columns_to_keep], inplace=True)
        df.replace(dict.fromkeys(self.MISSING_VALUES(), self._DEFAULT_MISSING_VALUE), inplace=True)

        # correct for screen resolution
        # note that coordinates may fall outside the screen, we don't clip them (see https://shorturl.at/hvBCY)
        screen_w, screen_h = max(screen_resolution), min(screen_resolution)
        df[self.LEFT_X_COLUMN()] = df[self.LEFT_X_COLUMN()] * screen_w
        df[self.LEFT_Y_COLUMN()] = df[self.LEFT_Y_COLUMN()] * screen_h
        df[self.RIGHT_X_COLUMN()] = df[self.RIGHT_X_COLUMN()] * screen_w
        df[self.RIGHT_Y_COLUMN()] = df[self.RIGHT_Y_COLUMN()] * screen_h

        # convert pupil size to float
        df[self.LEFT_PUPIL_COLUMN()] = df[self.LEFT_PUPIL_COLUMN()].astype(float)
        df[self.RIGHT_PUPIL_COLUMN()] = df[self.RIGHT_PUPIL_COLUMN()].astype(float)

        # reorder + rename columns to match the standard (except for the additional columns)
        df = df[columns_to_keep]
        df.rename(columns=lambda col: self._column_name_mapper(col), inplace=True)
        return df

    @classmethod
    def _read_raw_data(cls, input_path: str) -> pd.DataFrame:
        """
        Reads the raw data from the input CSV file.
        See information on the raw data format under "Tutorial 2 // Task 7" (page 56) in E-Prime's user manual for the
        Tobii eye-tracker (TET) package: https://pstnet.com/wp-content/uploads/2019/05/EET_User_Guide_3.2.pdf

        :param input_path: path to the input CSV file
        :return: a DataFrame containing the raw data

        :raises FileNotFoundError: if the input file does not exist
        :raise ValueError: if the input file is not a csv file
        """
        cls._raise_for_invalid_input_path(input_path)
        df = pd.read_csv(input_path, sep='\t', low_memory=False)
        return df

    @classmethod
    def FILE_EXTENSION(cls) -> str:
        # file extension of raw data files
        return '.csv'

    @classmethod
    def MISSING_VALUES(cls) -> Set[Union[int, float, str, None]]:
        return {-1, "-1", "-1.#IND0"}

    @classmethod
    def TRIAL_COLUMN(cls) -> Optional[str]:
        return 'RunningSample'

    @classmethod
    def SECONDS_COLUMN(cls) -> Optional[str]:
        return None

    @classmethod
    def MILLISECONDS_COLUMN(cls) -> Optional[str]:
        return 'RTTime'

    @classmethod
    def MICROSECONDS_COLUMN(cls) -> Optional[str]:
        return 'RTTimeMicro'

    @classmethod
    def LEFT_X_COLUMN(cls) -> Optional[str]:
        return 'GazePointPositionDisplayXLeftEye'

    @classmethod
    def LEFT_Y_COLUMN(cls) -> Optional[str]:
        return 'GazePointPositionDisplayYLeftEye'

    @classmethod
    def LEFT_PUPIL_COLUMN(cls) -> Optional[str]:
        return "PupilDiameterLeftEye"

    @classmethod
    def RIGHT_X_COLUMN(cls) -> Optional[str]:
        return 'GazePointPositionDisplayXRightEye'

    @classmethod
    def RIGHT_Y_COLUMN(cls) -> Optional[str]:
        return 'GazePointPositionDisplayYRightEye'

    @classmethod
    def RIGHT_PUPIL_COLUMN(cls) -> Optional[str]:
        return "PupilDiameterRightEye"

import os
import pandas as pd
from typing import List, Union, Tuple

from Config import experiment_config as cnfg
from DataParsers.BaseEyeTrackerParser import BaseEyeTrackerParser


class TobiiEyeTrackerCSVParser(BaseEyeTrackerParser):
    """
    Parses eye-tracking data based on the CSV format exported by Tobii eye-tracker and E-Prime.
    """

    def parse(self, input_path: str,
              screen_resolution: Tuple[float, float] = cnfg.SCREEN_MONITOR.resolution) -> pd.DataFrame:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f'File not found: {input_path}')
        df = pd.read_csv(input_path, sep='\t', low_memory=False)
        columns_to_keep = self.get_common_columns() + self.__additional_columns
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
    def MISSING_VALUES(cls) -> List[Union[int, float, str]]:
        return [-1, "-1", "-1.#IND0"]

    @classmethod
    def TRIAL_COLUMN(cls) -> str:
        return 'RunningSample'

    @classmethod
    def MILLISECONDS_COLUMN(cls) -> str:
        return 'RTTime'

    @classmethod
    def MICROSECONDS_COLUMN(cls) -> str:
        return 'RTTimeMicro'

    @classmethod
    def LEFT_X_COLUMN(cls) -> str:
        return 'GazePointPositionDisplayXLeftEye'

    @classmethod
    def LEFT_Y_COLUMN(cls) -> str:
        return 'GazePointPositionDisplayYLeftEye'

    @classmethod
    def LEFT_PUPIL_COLUMN(cls) -> str:
        return "PupilDiameterLeftEye"

    @classmethod
    def RIGHT_X_COLUMN(cls) -> str:
        return 'GazePointPositionDisplayXRightEye'

    @classmethod
    def RIGHT_Y_COLUMN(cls) -> str:
        return 'GazePointPositionDisplayYRightEye'

    @classmethod
    def RIGHT_PUPIL_COLUMN(cls) -> str:
        return "PupilDiameterRightEye"

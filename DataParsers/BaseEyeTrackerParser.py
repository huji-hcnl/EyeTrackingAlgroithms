import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Set, Tuple, final

import constants as cnst
from Config import experiment_config as cnfg


class BaseEyeTrackerParser(ABC):
    """
    Base class for all parsers handling raw data from eye trackers.
    These parsers take inputs from different eye trackers and map the data to a common format, using the method `parse`
    for parsing the data and `parse_and_split` for splitting the data into trials.
    """

    _DEFAULT_MISSING_VALUE = np.nan

    @final
    def __init__(self, experiment_specific_columns: Optional[List[str]] = None):
        if experiment_specific_columns is None:
            experiment_specific_columns = []
        self._experiment_specific_columns: List[str] = list(filter(lambda col: self.__is_valid_column_name(col),
                                                                   experiment_specific_columns))

    @final
    def parse(self, input_path: str,
              screen_resolution: Tuple[float, float] = cnfg.SCREEN_MONITOR.resolution) -> pd.DataFrame:
        """
        Reads and parses the raw data from the input file, following these steps:
        1. Reads the raw data from the input file (implemented separately for each parser)
        2. Removes all columns that are not relevant for the analysis
        3. Corrects the gaze coordinates for the screen resolution
        4. Performs additional parsing (implemented separately for each parser)
        5. Reorders and renames the columns to match the common format

        Returns a single Dataframe containing the parsed data.
        """
        df = self._read_raw_data(input_path)
        df = self._keep_relevant_data(df)
        df = self._correct_gaze_for_screen_resolution(df, screen_resolution)
        df = self._perform_additional_parsing(df)
        df = self._reorder_and_rename_columns(df)
        return df

    @final
    def parse_and_split_by_column(self, input_path: str, column_name: str) -> List[pd.DataFrame]:
        """
        Reads and parses the raw data from the input file, and splits the data into multiple DataFrames based on the
        unique values of the given column.

        :param input_path: path to the input file
        :param column_name: name of the column to split by
        :return: a list of DataFrames, each containing the data for a single value of the given column

        :raises ValueError: if the given column name is invalid
        :raises AttributeError: if the given column name is unfamiliar to the parser
        """
        if self.__is_valid_column_name(column_name):
            raise ValueError(f'Invalid column name: {column_name}')
        if column_name not in self.columns:
            raise AttributeError(f'{self.__class__.__name__} does not contain column {column_name}')
        df = self.parse(input_path)
        column_values = df[column_name].unique()
        return [df[df[column_name] == column_value] for column_value in column_values]

    @classmethod
    @abstractmethod
    def _read_raw_data(cls, input_path: str) -> pd.DataFrame:
        raise NotImplementedError

    @final
    def _keep_relevant_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes all columns from the DataFrame that are not relevant for the analysis, and replaces missing values with
        the default missing value.
        """
        new_df = df.drop(columns=[col for col in df.columns if col not in self.columns])
        new_df.replace(dict.fromkeys(self.MISSING_VALUES(), self._DEFAULT_MISSING_VALUE), inplace=True)
        return new_df

    @classmethod
    @final
    def _correct_gaze_for_screen_resolution(cls, df: pd.DataFrame,
                                            screen_resolution: Tuple[float, float]) -> pd.DataFrame:
        """
        Gaze data is measured in relative-coordinates, i.e. they are normalized to screen-resolution such that the
        top-left corner is (0, 0) and the bottom-right corner is (1, 1). This converts the relative-coordinates to
        absolute-coordinates, i.e. the top left corner is (0, 0) and the bottom right corner is (screen_width, screen_height).
        Note that coordinates may fall outside the screen, we don't clip them, see: https://developer.tobiipro.com/commonconcepts/coordinatesystems.html
        """
        new_df = df.copy()
        screen_w, screen_h = max(screen_resolution), min(screen_resolution)
        new_df[cls.LEFT_X_COLUMN()] = df[cls.LEFT_X_COLUMN()] * screen_w
        new_df[cls.LEFT_Y_COLUMN()] = df[cls.LEFT_Y_COLUMN()] * screen_h
        new_df[cls.RIGHT_X_COLUMN()] = df[cls.RIGHT_X_COLUMN()] * screen_w
        new_df[cls.RIGHT_Y_COLUMN()] = df[cls.RIGHT_Y_COLUMN()] * screen_h
        return new_df

    @classmethod
    @abstractmethod
    def _perform_additional_parsing(cls, df: pd.DataFrame) -> pd.DataFrame:
        # This method can be overridden by subclasses to perform additional parsing steps.
        raise NotImplementedError

    @final
    def _reorder_and_rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorders the columns of the DataFrame to the order specified by `columns` and renames the columns to match the
        standard naming scheme (except for the additional columns, which are not renamed).
        """
        new_df = df.copy()
        new_df = new_df[self.columns]
        df.rename(columns=lambda col: self._column_name_mapper(col), inplace=True)
        return new_df

    @final
    @property
    def columns(self) -> List[str]:
        return self._get_common_columns() + self._experiment_specific_columns

    @classmethod
    @abstractmethod
    def FILE_EXTENSION(cls) -> str:
        # file extension of raw data files
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def MISSING_VALUES(cls) -> Set[Union[int, float, str, None]]:
        # values of missing data in the raw data file
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def TRIAL_COLUMN(cls) -> Optional[str]:
        # column name for trial number
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def SECONDS_COLUMN(cls) -> Optional[str]:
        # column name for time in seconds
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def MILLISECONDS_COLUMN(cls) -> Optional[str]:
        # column name for time in milliseconds
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def MICROSECONDS_COLUMN(cls) -> Optional[str]:
        # column name for time in microseconds
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def LEFT_X_COLUMN(cls) -> Optional[str]:
        # column name for left eye x coordinate
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def LEFT_Y_COLUMN(cls) -> Optional[str]:
        # column name for left eye y coordinate
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def LEFT_PUPIL_COLUMN(cls) -> Optional[str]:
        # column name for left eye pupil diameter
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def RIGHT_X_COLUMN(cls) -> Optional[str]:
        # column name for right eye x coordinate
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def RIGHT_Y_COLUMN(cls) -> Optional[str]:
        # column name for right eye y coordinate
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def RIGHT_PUPIL_COLUMN(cls) -> Optional[str]:
        # column name for right eye pupil diameter
        raise NotImplementedError

    @classmethod
    @final
    def _raise_for_invalid_input_path(cls, input_path: str):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file '{input_path}' does not exist.")
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file '{input_path}' is not a file.")
        if not input_path.endswith(cls.FILE_EXTENSION()):
            raise ValueError(f"Input file '{input_path}' is not a '{cls.FILE_EXTENSION()}' file.")

    @classmethod
    def _get_common_columns(cls):
        columns = [cls.TRIAL_COLUMN(),
                   cls.SECONDS_COLUMN(), cls.MILLISECONDS_COLUMN(), cls.MICROSECONDS_COLUMN(),
                   cls.LEFT_X_COLUMN(), cls.LEFT_Y_COLUMN(), cls.LEFT_PUPIL_COLUMN(),
                   cls.RIGHT_X_COLUMN(), cls.RIGHT_Y_COLUMN(), cls.RIGHT_PUPIL_COLUMN()]
        return list(filter(lambda col: cls.__is_valid_column_name(col), columns))

    @classmethod
    def _column_name_mapper(cls, column_name: str) -> Optional[str]:
        # maps column names to constants
        if column_name == cls.TRIAL_COLUMN():
            return cnst.TRIAL
        if column_name == cls.MILLISECONDS_COLUMN():
            return cnst.MILLISECONDS
        if column_name == cls.MICROSECONDS_COLUMN():
            return cnst.MICROSECONDS
        if column_name == cls.LEFT_X_COLUMN():
            return cnst.LEFT_X
        if column_name == cls.LEFT_Y_COLUMN():
            return cnst.LEFT_Y
        if column_name == cls.LEFT_PUPIL_COLUMN():
            return cnst.LEFT_PUPIL
        if column_name == cls.RIGHT_X_COLUMN():
            return cnst.RIGHT_X
        if column_name == cls.RIGHT_Y_COLUMN():
            return cnst.RIGHT_Y
        if column_name == cls.RIGHT_PUPIL_COLUMN():
            return cnst.RIGHT_PUPIL
        return column_name

    @classmethod
    def __is_valid_column_name(cls, column_name: Optional[str]) -> bool:
        # checks if a column name is valid string
        if column_name is None:
            return False
        if not isinstance(column_name, str):
            return False
        if len(column_name) == 0:
            return False
        return True

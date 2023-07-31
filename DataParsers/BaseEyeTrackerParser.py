import os
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Optional, Union, Set, final

import constants as cnst


class BaseEyeTrackerParser(ABC):
    """
    Base class for all parsers handling raw data from eye trackers.
    These parsers take inputs from different eye trackers and map the data to a common format, using the method `parse`
    for parsing the data and `parse_and_split` for splitting the data into trials.
    """

    _DEFAULT_MISSING_VALUE = np.nan

    def __init__(self, additional_columns: Optional[List[str]] = None):
        if additional_columns is None:
            additional_columns = []
        self._additional_columns: List[str] = list(filter(lambda col: self.__is_valid_column_name(col),
                                                          additional_columns))

    @abstractmethod
    def parse(self, input_path: str, *args) -> pd.DataFrame:
        raise NotImplementedError

    @final
    def parse_and_split_by_trial(self, input_path: str) -> List[pd.DataFrame]:
        df = self.parse(input_path)
        trial_indices = df[cnst.TRIAL].unique()
        return [df[df[cnst.TRIAL] == trial_idx] for trial_idx in trial_indices]

    @classmethod
    def _get_common_columns(cls):
        columns = [cls.TRIAL_COLUMN(),
                   cls.SECONDS_COLUMN(), cls.MILLISECONDS_COLUMN(), cls.MICROSECONDS_COLUMN(),
                   cls.LEFT_X_COLUMN(), cls.LEFT_Y_COLUMN(), cls.LEFT_PUPIL_COLUMN(),
                   cls.RIGHT_X_COLUMN(), cls.RIGHT_Y_COLUMN(), cls.RIGHT_PUPIL_COLUMN()]
        return list(filter(lambda col: cls.__is_valid_column_name(col), columns))

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

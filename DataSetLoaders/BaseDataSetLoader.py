import pandas as pd
import requests as req

from abc import ABC, abstractmethod
from typing import final, List


class BaseDataSetLoader(ABC):

    _URL: str = None
    _ARTICLES: List[str] = None

    @classmethod
    @final
    def download(cls) -> pd.DataFrame:
        """ Downloads the dataset, parses it and returns a DataFrame with cleaned data """
        response = cls._download_raw_dataset()
        df = cls._parse_response(response)
        df = cls._clean_data(df)
        # df = df[cls.columns()]  # reorder columns
        return df

    @classmethod
    @final
    def save_to_pickle(cls, df: pd.DataFrame, path_file: str = None) -> None:
        if path_file is None:
            path_file = f"{cls.__name__}.pkl"
        if not path_file.endswith(".pkl"):
            path_file += ".pkl"
        df.to_pickle(path_file)

    @classmethod
    @final
    def articles(cls) -> List[str]:
        """ List of articles that are connected to the creation of this dataset """
        if not cls._ARTICLES:
            raise AttributeError(f"Class {cls.__name__} must implement class attribute `_ARTICLES`")
        return cls._ARTICLES

    @classmethod
    @abstractmethod
    def columns(cls) -> List[str]:
        """ Specifies the columns of the DataFrame that is returned by the `load` method """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _parse_response(cls, response: req.Response) -> pd.DataFrame:
        """ Parses the downloaded response and returns a DataFrame containing the raw dataset """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _clean_data(cls, df: pd.DataFrame) -> pd.DataFrame:
        """ Cleans the raw dataset and returns a DataFrame containing the cleaned dataset """
        raise NotImplementedError

    @classmethod
    @final
    def _download_raw_dataset(cls):
        if not cls._URL:
            raise AttributeError(f"Class {cls.__name__} must implement class attribute `_URL`")
        response = req.get(cls._URL)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download dataset from {cls._URL}")
        return response


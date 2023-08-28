import os
import pandas as pd
import requests as req
from abc import ABCMeta, abstractmethod
from typing import List


class BaseDataSetLoader(ABCMeta):
    _URL: str = None
    _ARTICLE: str = None

    @classmethod
    def from_remote(cls) -> pd.DataFrame:
        """ Loads the dataset from a remote URL, parses it and returns a DataFrame. """
        response = req.get(cls._URL)
        if response.status_code != 200:
            raise RuntimeError(f"Failed to download dataset from {cls._URL}")
        df = cls._parse_response(response)
        df = cls._replace_missing_values(df)
        df = df[cls.columns()]  # reorder columns
        return df

    @classmethod
    def save_to_pickle(cls, df: pd.DataFrame, path_file: str = None) -> None:
        if path_file is None:
            path_file = f"{cls.__name__}.pkl"
        if not path_file.endswith(".pkl"):
            path_file += ".pkl"
        df.to_pickle(path_file)

    @classmethod
    def url(cls) -> str:
        return cls._URL

    @classmethod
    def article(cls) -> str:
        return cls._ARTICLE

    @classmethod
    @abstractmethod
    def columns(cls) -> List[str]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _parse_response(cls, response: req.Response) -> pd.DataFrame:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def _replace_missing_values(cls, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs):
        if cls._URL is None:
            raise NotImplementedError(f"Class {cls.__name__} must implement class attribute `_URL`")
        if cls._ARTICLE is None:
            raise NotImplementedError(f"Class {cls.__name__} must implement class attribute `_ARTICLE`")


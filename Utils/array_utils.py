import warnings as w
import numpy as np
import pandas as pd
from typing import List, Tuple

import constants as cnst


def is_one_dimensional(arr) -> bool:
    """ Returns true if the array's shape is (n,) or (1, n) or (n, 1) """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return True
    if arr.ndim == 2 and min(arr.shape) == 1:
        return True
    return False


def temporal_derivative(f, t, deg: int = 1, time_coeff: float = cnst.MILLISECONDS_PER_SECOND) -> np.ndarray:
    if not is_one_dimensional(f) or not is_one_dimensional(t):
        raise ValueError("`f` and `t` must be one-dimensional")
    if len(f) != len(t):
        raise ValueError("`f` and `t` must be of the same length")
    if deg < 0:
        raise ValueError("`deg` must be non-negative")
    if time_coeff <= 0:
        raise ValueError("`time_coeff` must be positive")
    if deg == 0:
        return f
    df = np.concatenate([[0], np.diff(f)])  # first element is 0
    dt = np.concatenate([[np.nan], np.diff(t)])  # first element is NaN
    df_dt = np.divide(df, dt) * time_coeff
    return temporal_derivative(df_dt, t, deg=deg-1, time_coeff=time_coeff)


def extract_column_safe(data: pd.DataFrame, colname: str, warn: bool = True) -> np.ndarray:
    try:
        return data[colname].values
    except KeyError:
        if warn:
            w.warn(f"Column {colname} not found in the given DataFrame")
        return np.full(shape=data.shape[0], fill_value=np.nan)


def get_chunk_indices(arr) -> List[np.ndarray]:
    """
    Given a 1D array with multiple values, returns a list of arrays, where each array contains the indices of
    a different "chunk", i.e. a sequence of the same value.
    """
    if not is_one_dimensional(arr):
        raise ValueError("arr must be one-dimensional")
    indices = np.arange(len(arr))
    split_on = np.nonzero(np.diff(arr))[0] + 1  # +1 because we want to include the last index of each chunk
    chunk_indices = np.split(indices, split_on)
    return chunk_indices


def find_sequences_in_sparse_array(sparse_array: np.ndarray, sequence: np.ndarray) -> List[Tuple[int, int]]:
    """
    Finds all occurrences of the given sequence in the given sparse array, while ignoring intermediate NaN values.
    :param sparse_array: array to search in, may contain NaN values
    :param sequence: sequence to search for
    :return: list of (start_idx, end_idx) tuples for each occurrence of the sequence in the array

    see examples in https://stackoverflow.com/a/76812495/8543025
    """
    from numpy.lib.stride_tricks import sliding_window_view as swv
    if not is_one_dimensional(sparse_array):
        raise ValueError("arr must be one-dimensional")
    n = len(sequence)
    non_nan_idxs = np.where(~np.isnan(sparse_array))[0]
    if len(non_nan_idxs) < n:
        return []
    swv_non_nan_array = swv(sparse_array[non_nan_idxs], n)
    is_sequence = np.all(swv_non_nan_array == sequence, axis=1)
    start_end_idxs = list(zip(non_nan_idxs[:1-n][is_sequence], non_nan_idxs[n-1:][is_sequence]))
    return start_end_idxs

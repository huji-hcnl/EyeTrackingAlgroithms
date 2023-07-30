import numpy as np
from typing import List, Set


def is_one_dimensional(arr) -> bool:
    """ Returns true if the array's shape is (n,) or (1, n) or (n, 1) """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return True
    if arr.ndim == 2 and min(arr.shape) == 1:
        return True
    return False


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


def merge_proximal_chunks(arr, min_chunk_length: int, allow_short_chunks_of: Set = None) -> np.ndarray:
    """
    If two "chunks" of identical values are separated by a short "chunk" of other values, merges the two chunks into one
    chunk by filling the middle chunk with the value of the left chunk.
    Chunks with values specified in `allow_short_chunks_of` are not merged.

    :param arr: 1D array
    :param min_chunk_length: minimum length of a chunk that will not be merged
    :param allow_short_chunks_of: values that are allowed to be short
    :return: array with merged chunks
    """
    if not is_one_dimensional(arr):
        raise ValueError("arr must be one-dimensional")
    if min_chunk_length <= 0:
        raise ValueError("argument `min_chunk_length` must be >= 1")
    if allow_short_chunks_of is None or len(allow_short_chunks_of) == 0:
        allow_short_chunks_of = set()

    arr_copy = np.asarray(arr).copy()
    chunk_indices = get_chunk_indices(arr)
    for i, middle_chunk in enumerate(chunk_indices):
        if i == 0 or i == len(chunk_indices) - 1:
            # ignore the first and last chunk
            continue
        if len(middle_chunk) >= min_chunk_length:
            # ignore chunks that are long enough
            continue
        middle_chunk_value = arr_copy[middle_chunk[0]]
        if middle_chunk_value in allow_short_chunks_of:
            # ignore chunks of the specified types
            continue
        left_chunk_value = arr_copy[chunk_indices[i - 1][0]]
        right_chunk_value = arr_copy[chunk_indices[i + 1][0]]
        if left_chunk_value != right_chunk_value:
            # ignore middle chunks if the left and right chunks are not identical
            continue

        # reached here if the middle chunk is short, its value is not allowed to be short, and left and right chunks are
        # identical. merge the left and right chunks by filling the middle chunk with the value of the left chunk.
        arr_copy[middle_chunk] = left_chunk_value
    return arr_copy


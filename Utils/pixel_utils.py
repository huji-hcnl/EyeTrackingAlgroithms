import numpy as np


def calculate_euclidean_distances(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Calculates the Euclidean distance between subsequent pixels in the given x and y coordinates.
    :param xs: 1D array of x coordinates
    :param ys: 1D array of y coordinates
    :return: distance (in pixel units) between subsequent pixels
    """
    assert len(xs) == len(ys), "x-array and y-array must be of the same length"
    x_diff = np.diff(xs)
    y_diff = np.diff(ys)
    dist = np.sqrt(np.power(x_diff, 2) + np.power(y_diff, 2))
    return dist


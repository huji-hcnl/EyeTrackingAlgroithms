import numpy as np
from typing import Tuple

import constants as cnst
import Utils.array_utils as arr_utils


def calculate_euclidean_distances(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """
    Calculates the Euclidean distance between each pixel and the previous one, in the given x and y coordinates.
    :param xs: 1D array of x coordinates
    :param ys: 1D array of y coordinates
    :return: distance (in pixel units) between subsequent pixels
    """
    assert len(xs) == len(ys), "x-array and y-array must be of the same length"
    x_diff = np.diff(xs)
    y_diff = np.diff(ys)
    dist = np.sqrt(np.power(x_diff, 2) + np.power(y_diff, 2))
    return np.concatenate(([0], dist))  # first distance is 0


def calculate_velocities(xs: np.ndarray, ys: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    """
    Calculates the velocity of the gaze in pixels per milliseconds.
    :param xs: 1D array of x coordinates
    :param ys: 1D array of y coordinates
    :param timestamps: 1D array of timestamps (in milliseconds)
    :return: velocity (pixel / second) between subsequent pixels
    """
    assert len(xs) == len(ys) == len(timestamps), "x-array, y-array and timestamps-array must be of the same length"
    dist = calculate_euclidean_distances(xs, ys)
    cum_dist = np.cumsum(dist)
    velocities = arr_utils.temporal_derivative(cum_dist, timestamps, deg=1, time_coeff=cnst.MILLISECONDS_PER_SECOND)
    return velocities


def calculate_azimuth(p1: Tuple[float, float],
                      p2: Tuple[float, float],
                      zero_direction: str = 'E',
                      use_radians: bool = False) -> float:
    """
    Calculates the counter-clockwise angle between the line starting from p1 and ending at p2, and the line starting
    from p1 and pointing in the direction of `zero_direction`.

    :param p1: the (x,y) coordinates of the starting point of the line
    :param p2: the (x,y) coordinates of the ending point of the line
    :param zero_direction: the direction of the zero angle. Must be one of 'E', 'W', 'S', 'N' (case insensitive).
    :param use_radians: if True, returns the angle in radians. Otherwise, returns the angle in degrees.

    :return: the angle between the two lines, in range [0, 2*pi) or [0, 360), or np.nan if either p1 or p2 is invalid.

    :raises ValueError: if zero_direction is not one of 'E', 'W', 'S', 'N' (case insensitive).
    """
    # verify inputs
    zero_direction = zero_direction.upper()
    valid_directions = ['E', 'W', 'S', 'N']
    if zero_direction not in valid_directions:
        raise ValueError(f"zero_direction must be one of {valid_directions}")

    # exit early for invalid pixels:
    if not _is_valid_pixel(p1) or not _is_valid_pixel(p2):
        return np.nan

    # calc angle & adjust to the desired zero direction
    x1, y1 = p1
    x2, y2 = p2
    angle_rad = np.arctan2(y1 - y2, x2 - x1)  # counter-clockwise angle between line (p1, p2) and right-facing x-axis
    if zero_direction == 'W':
        angle_rad += np.pi
    elif zero_direction == 'S':
        angle_rad += np.pi / 2
    elif zero_direction == 'N':
        angle_rad -= np.pi / 2

    # make sure the angle is in range [0, 2*pi), and return
    angle_rad = angle_rad % (2 * np.pi)
    if use_radians:
        return angle_rad
    return np.rad2deg(angle_rad)


def is_in_rectangle(p: Tuple[float, float], top_left: Tuple[float, float], bottom_right: Tuple[float, float]) -> bool:
    if not _is_valid_pixel(p):
        raise ValueError(f"argument `p` must be a valid pixel: {p}")
    if not _is_valid_pixel(top_left) or not _is_valid_pixel(bottom_right):
        raise ValueError(f"arguments `top_left` and `bottom_right` must be valid pixels: {top_left}, {bottom_right}")
    if top_left[0] > bottom_right[0] or top_left[1] > bottom_right[1]:
        raise ValueError(f"argument `top_left` must be above and to the left of `bottom_right`: {top_left}, {bottom_right}")
    return top_left[0] <= p[0] <= bottom_right[0] and top_left[1] <= p[1] <= bottom_right[1]


def is_in_circle(p: Tuple[float, float], center: Tuple[float, float], radius: float) -> bool:
    if not _is_valid_pixel(p):
        raise ValueError(f"argument `p` must be a valid pixel: {p}")
    if not _is_valid_pixel(center):
        raise ValueError(f"argument `center` must be a valid pixel: {center}")
    if not np.isfinite(radius) or radius < 0:
        raise ValueError(f"argument `radius` must be a finite non-negative number: {radius}")
    return np.sqrt(np.power(p[0] - center[0], 2) + np.power(p[1] - center[1], 2)) <= radius


def _is_valid_pixel(p: Tuple[float, float]) -> bool:
    return np.isfinite(p[0]) and np.isfinite(p[1])
